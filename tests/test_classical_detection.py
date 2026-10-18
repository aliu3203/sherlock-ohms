"""Regression tests for the classical breadboard detector extensions.

These tests are additive to ``tests/test_vision_ml.py``; they exercise the
diagnostic side-channel and the offline evaluation CLI without touching the
pre-existing contract tests.

Synthetic images here establish *code behaviour*, not real-camera accuracy —
real accuracy still requires rectified captures with manual ground truth, as
documented in the ``evaluate_classical`` CLI help.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine import BoardSpec, HoleLUT
from vision_ml.classical_detector import (
    BaselineMismatchError,
    ClassicalBreadboardDetector,
)
from vision_ml import evaluate_classical


# -----------------------------------------------------------------------------
# Helpers / fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    return tmp_path


def _build_scene():
    """Synthetic 1400x520 scene: a wire, a resistor, a known-polarity LED."""
    lut = HoleLUT.generate_synthetic(BoardSpec.full_830())
    baseline = np.full((520, 1400, 3), 205, np.uint8)
    for x, y in lut.holes.values():
        cv2.circle(baseline, (round(x), round(y)), 2, (75, 75, 75), -1)
    image = baseline.copy()
    expected = {'wire': ('a5', 'a10'), 'resistor': ('d20', 'd25'), 'led': ('h40', 'h42')}
    for kind, holes in expected.items():
        p, q = (tuple(np.rint(lut.holes[hole]).astype(int)) for hole in holes)
        if kind == 'wire':
            cv2.line(image, p, q, (220, 45, 40), 5)
        else:
            cv2.line(image, p, q, (35, 35, 35), 3)
            cx, cy = (p[0] + q[0]) // 2, (p[1] + q[1]) // 2
            if kind == 'resistor':
                cv2.rectangle(image, (cx - 22, cy - 8), (cx + 22, cy + 8), (110, 165, 195), -1)
                for dx in (-12, 0, 12):
                    cv2.line(image, (cx + dx, cy - 8), (cx + dx, cy + 8), (30, 50, 70), 3)
            else:
                cv2.circle(image, (cx, cy), 10, (35, 35, 230), -1)
    return baseline, image, lut, expected


def _lut_to_simple_dict(lut: HoleLUT) -> dict[str, tuple[float, float]]:
    return {hid: (float(x), float(y)) for hid, (x, y) in lut.holes.items()}


# -----------------------------------------------------------------------------
# Diagnostic side-channel
# -----------------------------------------------------------------------------

def test_last_diagnostics_populated_per_detection():
    baseline, image, _lut, _expected = _build_scene()
    detector = ClassicalBreadboardDetector(baseline, led_anode_direction='left')
    detections = detector.detect(image)

    assert len(detector.last_diagnostics) == len(detections) == 3
    for det, diag in zip(detections, detector.last_diagnostics):
        assert diag['detection_id'] == det['id']
        assert diag['kind'] == det['type']
        assert diag['bbox'] == det['bbox']
        assert set(diag) >= {
            'detection_id', 'kind', 'confidence', 'keypoint_confidence',
            'endpoint_source', 'notes', 'bbox', 'aspect_ratio',
            'polarity_assumption', 'polarity_resolved',
        }
        assert diag['endpoint_source'] in {'skeleton', 'pca_extrema'}
        assert diag['keypoint_confidence'] == pytest.approx(det['keypoints'][0]['conf'])


def test_led_polarity_assumption_surfaced_in_diagnostic():
    baseline, image, _lut, _expected = _build_scene()
    known = ClassicalBreadboardDetector(baseline, led_anode_direction='left')
    known.detect(image)
    led_diag = next(d for d in known.last_diagnostics if d['kind'] == 'led')
    assert led_diag['polarity_assumption'] == 'left'
    assert led_diag['polarity_resolved'] is True

    unknown = ClassicalBreadboardDetector(baseline)
    unknown.detect(image)
    led_diag = next(d for d in unknown.last_diagnostics if d['kind'] == 'led')
    assert led_diag['polarity_assumption'] == 'unknown'
    assert led_diag['polarity_resolved'] is False
    assert 'polarity_unknown' in led_diag['notes']
    assert led_diag['keypoint_confidence'] < 0.50


def test_boundary_cropped_detection_is_flagged():
    """A wire touching the image edge should be flagged and low-confidence."""
    baseline = np.full((520, 1400, 3), 205, np.uint8)
    image = baseline.copy()
    # A near-horizontal wire that butts into column 0:
    cv2.line(image, (0, 260), (120, 260), (220, 45, 40), 5)

    detector = ClassicalBreadboardDetector(baseline)
    detections = detector.detect(image)
    assert detections, 'edge-touching wire should still be detected'
    diag = detector.last_diagnostics[0]
    assert diag['cropped_at_image_boundary'] is True
    assert 'cropped_at_image_boundary' in diag['notes']
    assert detections[0]['keypoints'][0]['conf'] <= 0.35


def test_rejections_record_small_blob_reason():
    """A tiny speck above min_area but below the shape threshold is rejected."""
    baseline = np.full((520, 1400, 3), 205, np.uint8)
    image = baseline.copy()
    # Compact circular smudge: large enough to beat min_area but round, so
    # ``aspect < 2`` triggers the "blob_short_or_compact" rejection branch.
    cv2.circle(image, (700, 260), 6, (10, 10, 10), -1)

    detector = ClassicalBreadboardDetector(baseline)
    detections = detector.detect(image)
    assert detections == []
    assert detector.last_rejections, 'rejected blob must be surfaced'
    assert all(r['reason'] == 'blob_short_or_compact' for r in detector.last_rejections)


def test_annotate_returns_image_with_overlays():
    baseline, image, _lut, _expected = _build_scene()
    detector = ClassicalBreadboardDetector(baseline, led_anode_direction='left')
    annotated = detector.annotate(image)
    assert annotated.shape == image.shape
    assert annotated.dtype == np.uint8
    # Overlay must have changed at least the pixels where boxes/text were drawn.
    assert not np.array_equal(annotated, image)


def test_detect_is_still_deterministic():
    """The diagnostic refactor must not break the existing determinism contract."""
    baseline, image, _lut, _expected = _build_scene()
    a = ClassicalBreadboardDetector(baseline, led_anode_direction='left').detect(image)
    b = ClassicalBreadboardDetector(baseline, led_anode_direction='left').detect(image)
    assert a == b


# -----------------------------------------------------------------------------
# evaluate_classical JSON loaders
# -----------------------------------------------------------------------------

def test_load_hole_lut_accepts_dict_and_tuple_forms(workdir: Path):
    path = workdir / 'lut.json'
    path.write_text(json.dumps({
        'a1': {'x': 70, 'y': 100},
        'a2': [90.0, 100.0],
    }))
    lut = evaluate_classical.load_hole_lut(path)
    assert lut == {'a1': (70.0, 100.0), 'a2': (90.0, 100.0)}


def test_load_hole_lut_rejects_malformed(workdir: Path):
    bad = workdir / 'bad.json'
    bad.write_text(json.dumps({'a1': 'nope'}))
    with pytest.raises(ValueError):
        evaluate_classical.load_hole_lut(bad)
    empty = workdir / 'empty.json'
    empty.write_text('{}')
    with pytest.raises(ValueError):
        evaluate_classical.load_hole_lut(empty)


def test_load_ground_truth_rekeys_by_basename(workdir: Path):
    path = workdir / 'gt.json'
    path.write_text(json.dumps({
        'subdir/scene_a.png': {
            'notes': 'manual',
            'components': [
                {'id': 'r1', 'type': 'resistor', 'endpoints': ['d20', 'd25']},
            ],
        }
    }))
    gt = evaluate_classical.load_ground_truth(path)
    assert set(gt) == {'scene_a.png'}
    assert gt['scene_a.png']['components'][0]['endpoints'] == ['d20', 'd25']


def test_load_ground_truth_requires_two_endpoints(workdir: Path):
    bad = workdir / 'gt.json'
    bad.write_text(json.dumps({'img.png': {'components': [{'type': 'wire', 'endpoints': ['a1']}]}}))
    with pytest.raises(ValueError):
        evaluate_classical.load_ground_truth(bad)


# -----------------------------------------------------------------------------
# attach_proposed_pins + match_detections_to_gt
# -----------------------------------------------------------------------------

def test_attach_proposed_pins_snaps_endpoints_to_hole_ids():
    baseline, image, lut, expected = _build_scene()
    detector = ClassicalBreadboardDetector(baseline, led_anode_direction='left')
    detections = detector.detect(image)
    enriched = evaluate_classical.attach_proposed_pins(
        detections, detector.last_diagnostics, lut=_lut_to_simple_dict(lut),
    )
    assert len(enriched) == 3
    by_kind = {d.kind: d for d in enriched}
    assert by_kind['wire'].pins == {'0': 'a5', '1': 'a10'} or \
           by_kind['wire'].pins == {'0': 'a10', '1': 'a5'}
    assert by_kind['resistor'].pins.values().__class__ is type({}.values())
    assert set(by_kind['resistor'].pins.values()) == set(expected['resistor'])
    led = by_kind['led']
    assert led.pins == {'anode': 'h40', 'cathode': 'h42'}
    assert not led.uncertain


def test_attach_proposed_pins_marks_far_endpoints_uncertain():
    baseline = np.full((520, 1400, 3), 205, np.uint8)
    detections = [{
        'id': 'det_1', 'type': 'resistor', 'conf': 0.9,
        'bbox': (10.0, 10.0, 60.0, 20.0),
        'keypoints': [
            {'x': 15.0, 'y': 15.0, 'conf': 0.9, 'vis': 2},  # near the only hole
            {'x': 500.0, 'y': 500.0, 'conf': 0.9, 'vis': 2},  # miles away
        ],
    }]
    lut = {'a1': (12.0, 15.0)}
    enriched = evaluate_classical.attach_proposed_pins(
        detections, diagnostics=[], lut=lut, max_snap_px=8.0,
    )
    assert len(enriched) == 1
    pin_a, pin_b = enriched[0].pin_details
    assert pin_a.ambiguous is False and pin_a.hole_id == 'a1'
    assert pin_b.ambiguous is True
    assert enriched[0].pins == {'0': 'a1'}
    assert enriched[0].uncertain is True


def test_match_detections_ignores_ids_and_prefers_hole_overlap():
    baseline, image, lut, expected = _build_scene()
    detector = ClassicalBreadboardDetector(baseline, led_anode_direction='left')
    detections = detector.detect(image)
    enriched = evaluate_classical.attach_proposed_pins(
        detections, detector.last_diagnostics, lut=_lut_to_simple_dict(lut),
    )
    gt_components = [
        {'id': 'wire_user_label', 'type': 'wire', 'endpoints': list(expected['wire'])},
        {'id': 'res_user_label', 'type': 'resistor', 'endpoints': list(expected['resistor'])},
        {'id': 'led_user_label', 'type': 'led', 'endpoints': list(expected['led'])},
    ]
    matches, missed, extra = evaluate_classical.match_detections_to_gt(
        enriched, gt_components,
    )
    assert missed == [] and extra == []
    assert len(matches) == 3
    assert all(m.hole_overlap == 2 and m.type_match for m in matches)


def test_match_detections_reports_missed_and_extra():
    baseline, image, lut, expected = _build_scene()
    detector = ClassicalBreadboardDetector(baseline, led_anode_direction='left')
    detections = detector.detect(image)
    enriched = evaluate_classical.attach_proposed_pins(
        detections, detector.last_diagnostics, lut=_lut_to_simple_dict(lut),
    )
    # Ground truth claims a nonexistent extra wire; omit resistor so detector's
    # resistor becomes "extra" from the GT point of view.
    gt_components = [
        {'id': 'wire_a', 'type': 'wire', 'endpoints': list(expected['wire'])},
        {'id': 'wire_phantom', 'type': 'wire', 'endpoints': ['a30', 'a35']},
        {'id': 'led_a', 'type': 'led', 'endpoints': list(expected['led'])},
    ]
    matches, missed, extra = evaluate_classical.match_detections_to_gt(
        enriched, gt_components,
    )
    # The phantom wire doesn't share any proposed hole, so it stays missed.
    missed_ids = {m.get('id') for m in missed}
    assert 'wire_phantom' in missed_ids
    extra_kinds = {d.kind for d in extra}
    assert 'resistor' in extra_kinds


# -----------------------------------------------------------------------------
# CLI smoke test
# -----------------------------------------------------------------------------

def test_evaluate_cli_runs_end_to_end(workdir: Path):
    baseline, image, lut, expected = _build_scene()
    baseline_path = workdir / 'baseline.png'
    image_path = workdir / 'inspection.png'
    cv2.imwrite(str(baseline_path), baseline)
    cv2.imwrite(str(image_path), image)
    lut_path = workdir / 'lut.json'
    lut_path.write_text(json.dumps(
        {hid: {'x': float(x), 'y': float(y)} for hid, (x, y) in lut.holes.items()}
    ))
    gt_path = workdir / 'gt.json'
    gt_path.write_text(json.dumps({
        image_path.name: {
            'components': [
                {'id': 'wire1', 'type': 'wire', 'endpoints': list(expected['wire'])},
                {'id': 'res1', 'type': 'resistor', 'endpoints': list(expected['resistor'])},
                {'id': 'led1', 'type': 'led', 'endpoints': list(expected['led'])},
            ]
        }
    }))
    out = workdir / 'eval_out'

    exit_code = evaluate_classical.main([
        '--baseline', str(baseline_path),
        '--image', str(image_path),
        '--output', str(out),
        '--lut', str(lut_path),
        '--ground-truth', str(gt_path),
        '--anode-direction', 'left',
    ])
    assert exit_code == 0
    report_path = out / 'report.json'
    assert report_path.is_file(), 'CLI must emit report.json'
    report = json.loads(report_path.read_text())
    assert report['anode_direction'] == 'left'
    assert len(report['images']) == 1
    img_report = report['images'][0]
    assert img_report['num_matches'] == 3
    assert img_report['num_missed'] == 0 and img_report['num_extra'] == 0
    assert img_report['endpoint_hole_accuracy'] == 1.0
    annotated = out / 'inspection_annotated.png'
    assert annotated.is_file(), 'CLI must emit diagnostic PNG per image'
    assert cv2.imread(str(annotated)).shape == (520, 1400, 3)


def test_evaluate_cli_rejects_missing_baseline(workdir: Path):
    with pytest.raises(SystemExit):
        evaluate_classical.main([
            '--baseline', str(workdir / 'does_not_exist.png'),
            '--image', str(workdir / 'any.png'),
            '--output', str(workdir / 'out'),
        ])


def test_evaluate_handles_image_without_baseline_match(workdir: Path):
    """An image whose lighting shifts drastically must surface the mismatch."""
    baseline, _, _lut, _expected = _build_scene()
    baseline_path = workdir / 'baseline.png'
    cv2.imwrite(str(baseline_path), baseline)
    # Darken the inspection to blow past the lighting-compensation threshold.
    dark = np.clip(baseline.astype(np.int16) - 60, 0, 255).astype(np.uint8)
    image_path = workdir / 'dark.png'
    cv2.imwrite(str(image_path), dark)
    out = workdir / 'out'
    detector = ClassicalBreadboardDetector(baseline)
    evaluation = evaluate_classical.evaluate_image(
        image_path, detector, lut=None, max_snap_px=8.0, ground_truth=None,
    )
    assert evaluation.error and 'BaselineMismatchError' in evaluation.error
    # And the invariant still holds — the exception didn't throw past us.
    with pytest.raises(BaselineMismatchError):
        detector.detect(dark)
