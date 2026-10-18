"""Offline evaluation CLI for :mod:`vision_ml.classical_detector`.

Runs the classical detector over one or more rectified inspection images, snaps
each endpoint against an optional calibrated hole LUT, renders a diagnostic
image, and -- when a ground-truth file is supplied -- reports per-image
metrics (missed, extra, classification errors, endpoint-to-hole accuracy,
uncertain detections). Hungarian/greedy matching never relies on detector IDs
equalling ground-truth IDs.

The CLI deliberately does NOT re-implement engine snapping policies. It uses
``engine.snapping.snap_detections`` as the authority whenever a hole LUT is
provided; its own nearest-hole helper is only a fallback used when the LUT is
supplied as a bare JSON dict without the engine's :class:`BoardSpec` metadata.

------------------------------------------------------------------------------
JSON formats
------------------------------------------------------------------------------

Hole LUT (``--lut``)::

    {
        "a1": {"x": 70.0, "y": 100.0},
        "a2": {"x": 90.0, "y": 100.0},
        ...
    }

Each key is a breadboard hole id understood by the engine. Values are the
rectified pixel coordinates of the hole centre.

Ground truth (``--ground-truth``)::

    {
        "images/photo1.png": {
            "notes": "optional free-form string",
            "components": [
                {"id": "r1", "type": "resistor", "endpoints": ["d20", "d25"]},
                {"id": "led1", "type": "led", "endpoints": ["h40", "h42"]}
            ]
        },
        "images/photo2.png": {"components": []}
    }

Image keys are matched by basename, so you can keep paths relative to any
dataset root. ``id`` is informational only; matching uses the endpoint hole
ids as an unordered set and prefers detections of the same ``type``.

------------------------------------------------------------------------------
Usage
------------------------------------------------------------------------------

    python -m vision_ml.evaluate_classical \\
        --baseline empty_baseline.png \\
        --image inspection1.png \\
        --image inspection2.png \\
        --output eval_run/ \\
        [--lut hole_lut.json] \\
        [--ground-truth gt.json] \\
        [--anode-direction left] \\
        [--max-snap-px 8.0]

The output directory receives one annotated PNG per inspection image and a
``report.json`` with per-image metrics plus an aggregate summary.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vision_ml.classical_detector import (  # noqa: E402
    BaselineMismatchError,
    ClassicalBreadboardDetector,
)

LOGGER = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# JSON loaders
# -----------------------------------------------------------------------------

def load_hole_lut(path: str | Path) -> dict[str, tuple[float, float]]:
    """Load a flat JSON hole LUT.

    Accepted shapes:
      * ``{"a1": {"x": 70, "y": 100}, ...}`` (preferred)
      * ``{"a1": [70, 100], ...}`` (two-element list/tuple)
    """
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict) or not data:
        raise ValueError(f'Hole LUT at {path} must be a non-empty JSON object')
    out: dict[str, tuple[float, float]] = {}
    for hole_id, value in data.items():
        if isinstance(value, dict):
            try:
                x, y = float(value['x']), float(value['y'])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f'Hole {hole_id!r} entry must have x/y fields') from exc
        elif isinstance(value, (list, tuple)) and len(value) == 2:
            x, y = float(value[0]), float(value[1])
        else:
            raise ValueError(f'Hole {hole_id!r} entry must be a dict or 2-tuple')
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError(f'Hole {hole_id!r} coordinates must be finite')
        out[hole_id] = (x, y)
    return out


def load_ground_truth(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load the GT JSON. Returns the dict keyed by image basename.

    The on-disk key may be a path (relative or absolute); we re-key by
    :func:`os.path.basename` so a caller can locate images independently.
    """
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError(f'Ground truth file {path} must contain a JSON object')
    rekeyed: dict[str, dict[str, Any]] = {}
    for key, value in raw.items():
        if not isinstance(value, dict):
            raise ValueError(f'GT entry for {key!r} must be a JSON object')
        components = value.get('components', [])
        normalised: list[dict[str, Any]] = []
        for comp in components:
            if not isinstance(comp, dict):
                raise ValueError(f'GT component in {key!r} must be a JSON object')
            if 'type' not in comp or 'endpoints' not in comp:
                raise ValueError(f'GT component in {key!r} needs type and endpoints')
            endpoints = list(comp['endpoints'])
            if len(endpoints) != 2:
                raise ValueError(f'GT component in {key!r} must list exactly two endpoints')
            normalised.append({
                'id': comp.get('id'),
                'type': str(comp['type']),
                'endpoints': endpoints,
            })
        basename = Path(key).name or key
        rekeyed[basename] = {'notes': value.get('notes'), 'components': normalised}
    return rekeyed


# -----------------------------------------------------------------------------
# Endpoint snapping
# -----------------------------------------------------------------------------

@dataclass
class ProposedPin:
    """One endpoint snapped to a hole id."""

    pin_name: str
    hole_id: str | None
    distance_px: float
    confidence: float
    ambiguous: bool


@dataclass
class DetectionWithPins:
    detection_id: str
    kind: str
    confidence: float
    bbox: tuple[float, float, float, float]
    endpoints_px: list[tuple[float, float]]
    pins: dict[str, str]  # pin_name -> hole_id (only non-ambiguous entries)
    pin_details: list[ProposedPin]
    diagnostic: dict[str, Any]
    uncertain: bool

    def to_json(self) -> dict[str, Any]:
        return {
            'detection_id': self.detection_id,
            'type': self.kind,
            'conf': self.confidence,
            'bbox': list(self.bbox),
            'endpoints_px': [list(p) for p in self.endpoints_px],
            'pins': dict(self.pins),
            'pin_details': [p.__dict__ for p in self.pin_details],
            'uncertain': self.uncertain,
            'diagnostic': self.diagnostic,
        }


_POLAR_PINS = {'led': ('anode', 'cathode'), 'cap_electrolytic': ('anode', 'cathode')}


def pin_names_for(kind: str) -> tuple[str, str]:
    return _POLAR_PINS.get(kind, ('0', '1'))


def _nearest_hole(x: float, y: float, lut: dict[str, tuple[float, float]]):
    best_id: str | None = None
    best_d2 = math.inf
    for hid, (hx, hy) in lut.items():
        d2 = (x - hx) ** 2 + (y - hy) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best_id = hid
    return best_id, math.sqrt(best_d2) if best_id is not None else math.inf


def attach_proposed_pins(
    detections: Iterable[dict[str, Any]],
    diagnostics: Iterable[dict[str, Any]],
    *,
    lut: dict[str, tuple[float, float]] | None = None,
    max_snap_px: float = 8.0,
    min_kpt_conf: float = 0.50,
) -> list[DetectionWithPins]:
    """Attach proposed hole ids to each detection using the engine's policy.

    When ``lut`` is None, no pins are proposed; the detection is still returned
    with uncertainty flagged from the keypoint confidences themselves.
    """
    diag_by_id = {d.get('detection_id'): d for d in diagnostics}
    results: list[DetectionWithPins] = []
    for det in detections:
        kind = det['type']
        pin_names = pin_names_for(kind)
        pin_details: list[ProposedPin] = []
        pins: dict[str, str] = {}
        uncertain = False
        for pin_name, kp in zip(pin_names, det['keypoints']):
            hole_id: str | None = None
            distance = math.inf
            ambiguous = kp['conf'] < min_kpt_conf
            if lut is not None:
                hole_id, distance = _nearest_hole(kp['x'], kp['y'], lut)
                if distance > max_snap_px:
                    ambiguous = True
            if ambiguous:
                uncertain = True
            else:
                if hole_id is not None:
                    pins[pin_name] = hole_id
            pin_details.append(
                ProposedPin(
                    pin_name=pin_name,
                    hole_id=hole_id,
                    distance_px=float(distance) if math.isfinite(distance) else -1.0,
                    confidence=float(kp['conf']),
                    ambiguous=ambiguous,
                )
            )
        results.append(
            DetectionWithPins(
                detection_id=det['id'],
                kind=kind,
                confidence=float(det['conf']),
                bbox=tuple(float(v) for v in det['bbox']),
                endpoints_px=[(float(kp['x']), float(kp['y'])) for kp in det['keypoints']],
                pins=pins,
                pin_details=pin_details,
                diagnostic=diag_by_id.get(det['id'], {}),
                uncertain=uncertain,
            )
        )
    return results


# -----------------------------------------------------------------------------
# Ground-truth matching
# -----------------------------------------------------------------------------

@dataclass
class MatchResult:
    gt_index: int
    detection_index: int
    gt: dict[str, Any]
    detection: DetectionWithPins
    hole_overlap: int  # 0, 1, or 2 — how many GT endpoints were also proposed
    type_match: bool


@dataclass
class ImageEvaluation:
    image_path: str
    detections: list[DetectionWithPins]
    matches: list[MatchResult] = field(default_factory=list)
    missed: list[dict[str, Any]] = field(default_factory=list)
    extra: list[DetectionWithPins] = field(default_factory=list)
    error: str | None = None

    def summary(self) -> dict[str, Any]:
        total_endpoints = 2 * len(self.matches)
        correct_endpoints = sum(m.hole_overlap for m in self.matches)
        classification_errors = sum(1 for m in self.matches if not m.type_match)
        uncertain = sum(1 for d in self.detections if d.uncertain)
        return {
            'image': self.image_path,
            'num_detections': len(self.detections),
            'num_ground_truth': len(self.matches) + len(self.missed),
            'num_matches': len(self.matches),
            'num_missed': len(self.missed),
            'num_extra': len(self.extra),
            'classification_errors': classification_errors,
            'uncertain_detections': uncertain,
            'endpoint_hole_accuracy': (
                correct_endpoints / total_endpoints if total_endpoints else None
            ),
            'error': self.error,
        }


def _hole_overlap(gt_endpoints: list[str], detection_pins: dict[str, str]) -> int:
    """Count how many ground-truth endpoints appear in the detection's pin set."""
    proposed = set(detection_pins.values())
    return sum(1 for h in gt_endpoints if h in proposed)


def match_detections_to_gt(
    detections: list[DetectionWithPins],
    gt_components: list[dict[str, Any]],
) -> tuple[list[MatchResult], list[dict[str, Any]], list[DetectionWithPins]]:
    """Greedy matching by (hole_overlap, type_match, bbox proximity).

    IDs are never used; matching relies on proposed hole ids and (as a
    tie-breaker) the pixel distance between a detection bbox centre and the
    midpoint of the ground-truth hole positions when available.
    """
    detection_used = [False] * len(detections)
    gt_used = [False] * len(gt_components)

    # Rank every (gt, det) pair by (hole_overlap, type_match, -centre_distance).
    # The centre distance is a soft tie-breaker that collapses to zero when
    # overlap is 2, which is the fully-correct case.
    ranked: list[tuple[int, int, float, int, int]] = []
    for gi, gt in enumerate(gt_components):
        gt_endpoints: list[str] = gt['endpoints']
        for di, det in enumerate(detections):
            overlap = _hole_overlap(gt_endpoints, det.pins)
            if overlap == 0 and det.pins:
                # Detection snapped somewhere else — don't match on noise.
                continue
            type_match = 1 if det.kind == gt['type'] else 0
            # Use detection bbox centre; GT doesn't carry pixel coordinates.
            cx = 0.5 * (det.bbox[0] + det.bbox[2])
            cy = 0.5 * (det.bbox[1] + det.bbox[3])
            tie = -float(math.hypot(cx, cy))
            ranked.append((overlap, type_match, tie, gi, di))
    ranked.sort(reverse=True)

    matches: list[MatchResult] = []
    for overlap, type_match_int, _tie, gi, di in ranked:
        if gt_used[gi] or detection_used[di]:
            continue
        matches.append(
            MatchResult(
                gt_index=gi,
                detection_index=di,
                gt=gt_components[gi],
                detection=detections[di],
                hole_overlap=overlap,
                type_match=bool(type_match_int),
            )
        )
        gt_used[gi] = True
        detection_used[di] = True

    missed = [gt for gi, gt in enumerate(gt_components) if not gt_used[gi]]
    extra = [det for di, det in enumerate(detections) if not detection_used[di]]
    return matches, missed, extra


# -----------------------------------------------------------------------------
# Image-level evaluation
# -----------------------------------------------------------------------------

def evaluate_image(
    image_path: Path,
    detector: ClassicalBreadboardDetector,
    *,
    lut: dict[str, tuple[float, float]] | None,
    max_snap_px: float,
    ground_truth: list[dict[str, Any]] | None,
) -> ImageEvaluation:
    image = cv2.imread(str(image_path))
    if image is None:
        return ImageEvaluation(image_path=str(image_path), detections=[],
                               error=f'cannot read image {image_path}')
    try:
        detections = detector.detect(image)
    except (ValueError, BaselineMismatchError) as exc:
        return ImageEvaluation(image_path=str(image_path), detections=[],
                               error=f'{type(exc).__name__}: {exc}')
    enriched = attach_proposed_pins(
        detections,
        detector.last_diagnostics,
        lut=lut,
        max_snap_px=max_snap_px,
    )
    evaluation = ImageEvaluation(image_path=str(image_path), detections=enriched)
    if ground_truth is None:
        return evaluation
    matches, missed, extra = match_detections_to_gt(enriched, ground_truth)
    evaluation.matches = matches
    evaluation.missed = missed
    evaluation.extra = extra
    return evaluation


# -----------------------------------------------------------------------------
# Rendering
# -----------------------------------------------------------------------------

def render_diagnostic(
    detector: ClassicalBreadboardDetector,
    image: np.ndarray,
    evaluation: ImageEvaluation,
) -> np.ndarray:
    detection_dicts = [
        {
            'id': d.detection_id,
            'type': d.kind,
            'conf': d.confidence,
            'bbox': d.bbox,
            'keypoints': [
                {'x': px, 'y': py,
                 'conf': d.pin_details[i].confidence if i < len(d.pin_details) else 0.0,
                 'vis': 2}
                for i, (px, py) in enumerate(d.endpoints_px)
            ],
        }
        for d in evaluation.detections
    ]
    proposed = [
        {'detection_id': d.detection_id, 'pins': d.pins}
        for d in evaluation.detections
    ]
    return detector.annotate(
        image,
        detections=detection_dicts,
        diagnostics=[d.diagnostic for d in evaluation.detections],
        proposed_pins=proposed,
    )


# -----------------------------------------------------------------------------
# Aggregate report
# -----------------------------------------------------------------------------

def aggregate_summary(evaluations: list[ImageEvaluation]) -> dict[str, Any]:
    totals = Counter()
    endpoint_hits = 0
    endpoint_total = 0
    per_image = []
    for ev in evaluations:
        summary = ev.summary()
        per_image.append(summary)
        if summary['error']:
            continue
        totals['detections'] += summary['num_detections']
        totals['ground_truth'] += summary['num_ground_truth']
        totals['matches'] += summary['num_matches']
        totals['missed'] += summary['num_missed']
        totals['extra'] += summary['num_extra']
        totals['classification_errors'] += summary['classification_errors']
        totals['uncertain_detections'] += summary['uncertain_detections']
        endpoint_hits += 2 * summary['num_matches'] * (summary['endpoint_hole_accuracy'] or 0.0)
        endpoint_total += 2 * summary['num_matches']
    return {
        'per_image': per_image,
        'totals': dict(totals),
        'aggregate_endpoint_hole_accuracy': (
            endpoint_hits / endpoint_total if endpoint_total else None
        ),
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='evaluate_classical',
        description='Offline evaluation of ClassicalBreadboardDetector.',
    )
    parser.add_argument('--baseline', required=True, type=Path,
                        help='Empty-board rectified baseline PNG.')
    parser.add_argument('--image', action='append', dest='images', required=True,
                        type=Path, help='Rectified inspection image (repeatable).')
    parser.add_argument('--output', required=True, type=Path,
                        help='Directory for diagnostic PNGs and report.json.')
    parser.add_argument('--lut', type=Path, default=None,
                        help='Hole LUT JSON (see module docstring for format).')
    parser.add_argument('--ground-truth', type=Path, default=None,
                        help='Ground truth JSON (see module docstring for format).')
    parser.add_argument('--anode-direction', default=None,
                        choices=(None, 'left', 'right', 'top', 'bottom'),
                        help='LED anode direction assumption; omit if unknown.')
    parser.add_argument('--max-snap-px', type=float, default=8.0,
                        help='Max px between endpoint and nearest hole before UNCERTAIN.')
    parser.add_argument('--min-difference', type=int, default=20,
                        help='Minimum per-pixel foreground delta (see detector).')
    parser.add_argument('--min-area', type=int, default=35,
                        help='Minimum connected-component area for detection.')
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')
    baseline_img = cv2.imread(str(args.baseline))
    if baseline_img is None:
        raise SystemExit(f'Cannot read baseline image at {args.baseline}')
    detector = ClassicalBreadboardDetector(
        baseline_img,
        min_difference=args.min_difference,
        min_area=args.min_area,
        led_anode_direction=args.anode_direction,
    )
    lut = load_hole_lut(args.lut) if args.lut else None
    gt_all = load_ground_truth(args.ground_truth) if args.ground_truth else None

    args.output.mkdir(parents=True, exist_ok=True)
    evaluations: list[ImageEvaluation] = []
    for image_path in args.images:
        gt_components = None
        if gt_all is not None:
            gt_components = gt_all.get(image_path.name, {}).get('components')
            if gt_components is None:
                LOGGER.warning('No ground truth entry for %s (looked up by basename)',
                               image_path.name)
                gt_components = []
        evaluation = evaluate_image(
            image_path, detector, lut=lut,
            max_snap_px=args.max_snap_px,
            ground_truth=gt_components,
        )
        evaluations.append(evaluation)
        if evaluation.error:
            LOGGER.error('skip %s: %s', image_path, evaluation.error)
            continue
        image = cv2.imread(str(image_path))
        annotated = render_diagnostic(detector, image, evaluation)
        out_path = args.output / f'{image_path.stem}_annotated.png'
        if not cv2.imwrite(str(out_path), annotated):
            LOGGER.error('Failed to write diagnostic image %s', out_path)

    report = {
        'baseline': str(args.baseline),
        'lut': str(args.lut) if args.lut else None,
        'ground_truth': str(args.ground_truth) if args.ground_truth else None,
        'anode_direction': args.anode_direction,
        'images': [
            {
                **ev.summary(),
                'detections': [d.to_json() for d in ev.detections],
                'matches': [
                    {
                        'gt_id': m.gt.get('id'),
                        'gt_type': m.gt['type'],
                        'gt_endpoints': m.gt['endpoints'],
                        'detection_id': m.detection.detection_id,
                        'detection_type': m.detection.kind,
                        'detection_pins': m.detection.pins,
                        'hole_overlap': m.hole_overlap,
                        'type_match': m.type_match,
                    }
                    for m in ev.matches
                ],
                'missed': [{'gt_id': m.get('id'), 'type': m['type'],
                            'endpoints': m['endpoints']} for m in ev.missed],
                'extra': [{'detection_id': d.detection_id, 'type': d.kind,
                           'pins': d.pins} for d in ev.extra],
            }
            for ev in evaluations
        ],
        'aggregate': aggregate_summary(evaluations),
    }
    report_path = args.output / 'report.json'
    report_path.write_text(json.dumps(report, indent=2))
    LOGGER.info('Wrote %s', report_path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
