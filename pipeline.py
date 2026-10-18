"""Camera-to-verdict integration for Sherlock Ohms.

Pass a ReferenceCircuit whose pins name holes (including golden jumper wires)
to derive both placement and connectivity references. Alternatively pass an
abstract net reference plus placement_reference for dual-mode verification.

    pipeline = SherlockOhmsPipeline(golden_layout, model_path='best_ncnn_model')
    result = pipeline.inspect(raw_bgr_image, mode='connectivity')
    # result.passed, result.verdicts, result.quality, result.annotated_image

inspect(None) captures from an injected camera, or generates a synthetic ArUco
scene for demos. Injected detections are already in rectified pixel coordinates.
The synthetic scene tests registration, not the accuracy of a trained detector.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import textwrap
from typing import Iterable, Literal

import cv2
import numpy as np
from pydantic import Field

from engine.board import BoardSpec, HoleLUT
from engine.netlist import build_netlist
from engine.schemas import Finding, InspectionVerdict, RawDetection, ReferenceCircuit, SnappedPart
from engine.snapping import pin_names_for, snap_detections
from engine.verifier import verify_circuit
from vision_ml.detector import BreadboardPoseDetector
from vision_ml.pi_hardware import BreadboardCamera, TrafficLight
from vision_ml.registration import (
    BreadboardRegistration, QualityGateError, QualityReport, RegistrationResult,
    generate_synthetic_aruco_board_image,
)

Mode = Literal['placement', 'connectivity']
GREEN = (0, 200, 0)
RED = (0, 0, 230)
AMBER = (0, 191, 255)


class PipelineVerdict(InspectionVerdict):
    """Engine verdict retaining original keypoints for stateless rendering."""
    raw_detections: list[RawDetection] = Field(default_factory=list, exclude=True)


@dataclass
class PipelineResult:
    mode: Mode
    registration: RegistrationResult
    verdict: PipelineVerdict | None = None
    verdicts: dict[str, PipelineVerdict] = field(default_factory=dict)
    detections: list[RawDetection] = field(default_factory=list)
    annotated_image: np.ndarray | None = None

    @property
    def aborted(self) -> bool:
        return self.verdict is None

    @property
    def passed(self) -> bool:
        return self.quality.passed and self.verdict is not None and self.verdict.passed

    @property
    def light_color(self) -> str:
        return 'GREEN' if self.passed else 'RED'

    @property
    def quality(self) -> QualityReport:
        return self.registration.quality

    @property
    def rectified_image(self) -> np.ndarray | None:
        return self.registration.rectified

    @property
    def findings(self) -> list[Finding]:
        return self.verdict.findings if self.verdict is not None else []


class SherlockOhmsPipeline:
    def __init__(self, reference: ReferenceCircuit, *,
                 placement_reference: ReferenceCircuit | None = None,
                 board: BoardSpec | None = None, lut: HoleLUT | None = None,
                 registration: BreadboardRegistration | None = None,
                 model_path: str | Path | None = None,
                 detector: BreadboardPoseDetector | None = None,
                 traffic_light: TrafficLight | None = None,
                 camera: BreadboardCamera | None = None,
                 device: str = 'cpu', max_snap_px: float = 8.0):
        self.board = board or (lut.board if lut is not None else BoardSpec.full_830())
        self.lut = lut if lut is not None else HoleLUT.generate_synthetic(self.board)
        self.registration = registration if registration is not None else BreadboardRegistration()
        if self.lut.board != self.board or not self.lut.holes:
            raise ValueError('A nonempty LUT for the selected board is required')
        if self.registration.output_size != (1400, 520):
            raise ValueError('Pipeline registration must produce 1400x520 images')
        if not np.isfinite(max_snap_px) or max_snap_px <= 0:
            raise ValueError('max_snap_px must be positive')
        if detector is not None and model_path is not None:
            raise ValueError('Supply either a detector or model_path')
        self.max_snap_px = max_snap_px
        self.camera = camera
        # Engine rail_bindings are physical rail -> electrical label, opposite
        # to ReferenceCircuit.rails (label -> physical rail).
        self.rail_bindings = {rail: label for label, rail in reference.rails.items()}
        if len(self.rail_bindings) != len(reference.rails):
            raise ValueError('Reference rail labels must bind distinct physical rails')
        values = [hole for part in reference.parts for hole in part['pins'].values()]
        if values and all(hole in self.lut for hole in values):
            if placement_reference is not None:
                raise ValueError('The supplied reference already defines placement')
            self.placement_reference = reference
            self.connectivity_reference = self._connectivity_from_layout(reference)
        else:
            if any(hole in self.lut for hole in values):
                raise ValueError('Reference mixes hole IDs and abstract net names')
            self.connectivity_reference = reference
            self.placement_reference = placement_reference
        if self.placement_reference is not None:
            if self.placement_reference.rails != reference.rails:
                raise ValueError('Placement and connectivity rail bindings must agree')
            for part in self.placement_reference.parts:
                if any(hole not in self.lut for hole in part['pins'].values()):
                    raise ValueError('Placement references must contain known LUT hole IDs')
        self.detector = detector if detector is not None else BreadboardPoseDetector(model_path, device=device)
        self.traffic_light = traffic_light if traffic_light is not None else TrafficLight()

    def _connectivity_from_layout(self, layout: ReferenceCircuit) -> ReferenceCircuit:
        parts = [SnappedPart(id=p['id'], type=p['type'], pins=p['pins'],
                             bbox=(0, 0, 0, 0), conf=1.0) for p in layout.parts]
        nl = build_netlist(parts, self.board, self.rail_bindings)
        names = {}
        for root, members in nl.nets.items():
            rails = [label for label in ('VCC', 'GND') if label in members]
            if len(rails) > 1:
                raise ValueError('Golden layout shorts VCC to GND')
            names[root] = rails[0] if rails else f'net:{root}'
        return ReferenceCircuit(name=layout.name, rails=layout.rails, parts=[
            {'id': p.id, 'type': p.type,
             'pins': {pin: names[nl.pin_to_net[(p.id, pin)]] for pin in p.pins}}
            for p in parts if p.type != 'wire'
        ])

    @staticmethod
    def _raw_detections(values: Iterable[dict | RawDetection]) -> list[RawDetection]:
        detections = [RawDetection.model_validate(value).model_copy(deep=True) for value in values]
        if len({d.id for d in detections}) != len(detections):
            raise ValueError('Detection IDs must be unique within an inspection')
        for det in detections:
            numbers = [*det.bbox, det.conf, *[v for k in det.keypoints for v in (k.x, k.y, k.conf)]]
            if not np.isfinite(numbers).all():
                raise ValueError(f'Detection {det.id} contains nonfinite values')
            if det.bbox[0] >= det.bbox[2] or det.bbox[1] >= det.bbox[3]:
                raise ValueError(f'Detection {det.id} has an invalid bounding box')
            if not 0 <= det.conf <= 1 or any(not 0 <= k.conf <= 1 for k in det.keypoints):
                raise ValueError('Detection confidences must be within [0, 1]')
            # The engine uses confidence for uncertainty; missing/occluded
            # endpoints must not be treated as confidently visible snaps.
            for k in det.keypoints:
                if k.vis not in (0, 1, 2):
                    raise ValueError('Keypoint visibility must be 0, 1, or 2')
            det.keypoints = tuple(k.model_copy(update={'conf': 0.0}) if k.vis == 0 else k
                                  for k in det.keypoints)
        return detections

    def inspect(self, raw_image: np.ndarray | None = None, *,
                mode: Mode = 'connectivity',
                detections: Iterable[dict | RawDetection] | None = None) -> PipelineResult:
        """Register, infer once, snap once, verify available modes, and render.

        Quality failure is an aborted result (verdict=None). Unexpected errors
        set RED and propagate. None detections runs inference; [] means no parts.
        Warnings retain the engine's semantics: only critical findings fail.
        """
        self.traffic_light.set('yellow')
        try:
            if mode not in ('placement', 'connectivity'):
                raise ValueError('mode must be placement or connectivity')
            if mode == 'placement' and self.placement_reference is None:
                raise ValueError('Placement mode requires a hole-based placement_reference')
            if raw_image is None:
                raw_image = (self.camera.capture_burst() if self.camera is not None
                             else generate_synthetic_aruco_board_image())
            if raw_image.ndim == 2:
                raw_image = cv2.cvtColor(raw_image, cv2.COLOR_GRAY2BGR)
            try:
                registration = self.registration.register(raw_image)
            except QualityGateError as exc:
                result = PipelineResult(mode=mode, registration=exc.result)
                if result.rectified_image is not None:
                    result.annotated_image = result.rectified_image.copy()
                    self._draw_text(result.annotated_image,
                                    ['QUALITY GATE: ' + ', '.join(result.quality.failures)], RED)
                self.traffic_light.set('red')
                return result
            rectified = registration.rectified
            raw = self._raw_detections(self.detector.detect(rectified) if detections is None else detections)
            snapped, findings = snap_detections(raw, self.lut, max_snap_px=self.max_snap_px)
            references = {'connectivity': self.connectivity_reference}
            if self.placement_reference is not None:
                references['placement'] = self.placement_reference
            verdicts = {}
            for name, reference in references.items():
                verdict = verify_circuit(snapped, reference=reference, board=self.board,
                                         rail_bindings=self.rail_bindings, mode=name,
                                         extra_findings=findings)
                verdicts[name] = PipelineVerdict(**verdict.model_dump(), raw_detections=raw)
            verdict = verdicts[mode]
            annotated = self.render_annotated_image(rectified, verdict, self.lut)
            result = PipelineResult(mode, registration, verdict, verdicts, raw, annotated)
            self.traffic_light.set(result.light_color.lower())
            return result
        except Exception:
            self.traffic_light.set('red')
            raise

    def inspect_frame(self, raw_image: np.ndarray | None = None, *,
                      mode: Mode = 'connectivity',
                      detections: Iterable[dict | RawDetection] | None = None) -> PipelineResult:
        """Frame-oriented public entry point; identical semantics to inspect()."""
        return self.inspect(raw_image, mode=mode, detections=detections)

    @staticmethod
    def _draw_text(image: np.ndarray, lines: list[str], color: tuple) -> None:
        # Keep text bounded even for large finding lists; findings remain intact
        # in the structured verdict when the visual summary needs truncation.
        lines = [wrapped for line in lines for wrapped in textwrap.wrap(line, width=110)]
        max_lines = (image.shape[0] - 12) // 19
        if len(lines) > max_lines:
            lines = lines[:max_lines-1] + ['... more findings in structured verdict']
        for index, line in enumerate(lines):
            origin = (10, 19 + 19*index)
            cv2.putText(image, line, origin, cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(image, line, origin, cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1, cv2.LINE_AA)

    def render_annotated_image(self, rectified_img: np.ndarray,
                               verdict: InspectionVerdict, lut: HoleLUT) -> np.ndarray:
        """Copy and annotate; PipelineVerdict retains raw points across runs.

        Global faults without a part location color all boxes conservatively.
        Ordinary engine verdicts can render boxes/findings, but contain no raw
        keypoints from which to draw snap vectors.
        """
        if rectified_img.shape != (520, 1400, 3) or rectified_img.dtype != np.uint8:
            raise ValueError('Expected a uint8 1400x520 BGR rectified image')
        image = rectified_img.copy()

        def pixel(xy):
            return tuple(np.rint(np.clip(xy, [0, 0], [1399, 519])).astype(int))

        for point in lut.holes.values():
            cv2.circle(image, pixel(point), 1, (150, 150, 150), -1)
        raw = {d.id: d for d in getattr(verdict, 'raw_detections', [])}
        for part in verdict.snapped_parts:
            issues = [f for f in verdict.findings if f.part_id == part.id or f.part_id is None]
            color = (RED if any(f.severity == 'critical' for f in issues)
                     else AMBER if issues else GREEN)
            cv2.rectangle(image, pixel(part.bbox[:2]), pixel(part.bbox[2:]), color, 2)
            cv2.putText(image, f'{part.id} {part.type}', pixel((part.bbox[0], part.bbox[1]-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, .4, color, 1, cv2.LINE_AA)
            detection = raw.get(part.id)
            if detection is not None:
                for name, keypoint in zip(pin_names_for(part.type), detection.keypoints):
                    target = lut.get(part.pins.get(name, ''))
                    if target is not None:
                        start, end = pixel((keypoint.x, keypoint.y)), pixel(target)
                        cv2.arrowedLine(image, start, end, color, 1, cv2.LINE_AA, tipLength=.3)
                        cv2.circle(image, start, 3, color, 1)
                        cv2.drawMarker(image, end, color, cv2.MARKER_CROSS, 5, 1)
        for finding in verdict.findings:
            color = RED if finding.severity == 'critical' else AMBER
            for hole in finding.holes + finding.expected_holes:
                point = lut.get(hole)
                if point is not None:
                    cv2.circle(image, pixel(point), 6, color, 1)
        self._draw_text(image, ['PASS' if verdict.passed else 'FAIL'] +
                        [f'{f.code}: {f.message}' for f in verdict.findings],
                        GREEN if verdict.passed else RED)
        return image

    def close(self):
        self.traffic_light.close()
        if self.camera is not None:
            self.camera.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
