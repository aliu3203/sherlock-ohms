"""Zero-training foreground detector for separated, visible breadboard parts.

Use the same registered 1400x520 geometry and locked camera settings for baseline
and inspection. A median baseline must contain EMPTY-board frames. Touching or
crossing components cannot reliably be separated by foreground contours.

LED polarity is not observable from color or silhouette alone. Set
led_anode_direction only for a known demo orientation; otherwise LED endpoint
confidence remains below the engine's 0.50 acceptance threshold.

Each call to :meth:`ClassicalBreadboardDetector.detect` also populates two
diagnostic attributes:

* ``last_diagnostics`` -- one dict per returned detection, including the
  endpoint source (skeleton vs. PCA fallback), classification confidence,
  polarity assumption, and free-form notes such as ``cropped_at_image_boundary``
  or ``leads_not_visible``.
* ``last_rejections`` -- one dict per blob considered but discarded (for
  example too compact to carry two visible leads), carrying the bounding box
  and the reason. Nothing in ``last_rejections`` is returned to callers; it
  exists so operators and the ``evaluate_classical`` CLI can explain why a
  foreground region produced no detection.

Diagnostics are side-channel data: the returned detection dicts remain
byte-for-byte compatible with the engine's :class:`RawDetection` schema and
with earlier releases of this detector.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


class BaselineMismatchError(ValueError):
    """Too much of the board changed for safe component segmentation."""


def _validate_image(image):
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
            or image.shape != (520, 1400, 3)):
        raise ValueError('Expected a uint8 BGR rectified image with shape (520, 1400, 3)')
    return image


def _thin(mask):
    """Zhang-Suen thinning on a small blob ROI; no opencv-contrib dependency."""
    skeleton = np.pad((mask > 0).astype(np.uint8), 1)
    while True:
        changed = False
        for phase in (0, 1):
            p = skeleton[1:-1, 1:-1]
            n = [skeleton[:-2, 1:-1], skeleton[:-2, 2:], skeleton[1:-1, 2:],
                 skeleton[2:, 2:], skeleton[2:, 1:-1], skeleton[2:, :-2],
                 skeleton[1:-1, :-2], skeleton[:-2, :-2]]
            count = sum(n)
            transitions = sum(((n[i] == 0) & (n[(i+1) % 8] == 1)).astype(np.uint8)
                              for i in range(8))
            if phase == 0:
                clear = (n[0]*n[2]*n[4] == 0) & (n[2]*n[4]*n[6] == 0)
            else:
                clear = (n[0]*n[2]*n[6] == 0) & (n[0]*n[4]*n[6] == 0)
            remove = (p == 1) & (count >= 2) & (count <= 6) & (transitions == 1) & clear
            if np.any(remove):
                p[remove] = 0
                changed = True
        if not changed:
            return skeleton[1:-1, 1:-1]


@dataclass
class BlobDescription:
    """Internal: everything we learn about one foreground blob.

    Only ``kind``/``confidence``/``endpoints``/``keypoint_confidence`` feed the
    returned RawDetection; the rest is diagnostic.
    """

    kind: str
    confidence: float
    endpoints: np.ndarray  # shape (2, 2), (x, y) rows, pixel coords with offset applied
    keypoint_confidence: float
    endpoint_source: str  # "skeleton" | "pca_extrema"
    polarity_assumption: str | None  # None, "unknown", "left"/"right"/"top"/"bottom"
    polarity_resolved: bool  # True iff a non-None assumption produced ordered endpoints
    body_present: bool
    aspect_ratio: float
    long_side: float
    short_side: float
    notes: list[str] = field(default_factory=list)


class ClassicalBreadboardDetector:
    """Drop-in detector: detect(BGR image) -> list of RawDetection dictionaries.

    Pixel dimensions assume 8 px/mm. Tune min_difference/min_area and
    body_radius for the camera. conf values are heuristic scores, not learned
    probabilities. IDs are spatially sorted and restart at det_1 for each frame.

    After each :meth:`detect` call, :attr:`last_diagnostics` and
    :attr:`last_rejections` describe how each foreground region was handled.
    """
    def __init__(self, baseline_img=None, *, baseline_frames: Iterable[np.ndarray] | None = None,
                 baseline_path='empty_baseline.png', min_difference=20, min_area=35,
                 max_blob_area=12000, adaptive_block_size=31, body_radius=5.0,
                 max_foreground_fraction=.20, compensate_lighting=True,
                 led_anode_direction=None):
        if baseline_img is not None and baseline_frames is not None:
            raise ValueError('Supply baseline_img or baseline_frames, not both')
        if not 0 < min_difference <= 255 or not 0 < min_area < max_blob_area:
            raise ValueError('Invalid difference threshold or blob area limits')
        if adaptive_block_size < 3 or adaptive_block_size % 2 != 1:
            raise ValueError('adaptive_block_size must be an odd integer >= 3')
        if not 0 < max_foreground_fraction < 1 or not np.isfinite(body_radius) or body_radius <= 1:
            raise ValueError('Invalid foreground fraction or body radius')
        if led_anode_direction not in (None, 'left', 'right', 'top', 'bottom'):
            raise ValueError('led_anode_direction must be left/right/top/bottom or None')
        if baseline_frames is not None:
            baseline_img = self.median_baseline(baseline_frames)
        elif baseline_img is None:
            baseline_img = baseline_path
        if isinstance(baseline_img, (str, Path)):
            path = Path(baseline_img)
            if not path.is_file():
                raise FileNotFoundError(f'Empty baseline missing: {path}. Supply an image or >=3 empty frames.')
            baseline_img = cv2.imread(str(path))
            if baseline_img is None:
                raise ValueError(f'Cannot decode baseline: {path}')
        self.baseline_img = _validate_image(baseline_img).copy()
        self.min_difference, self.min_area = min_difference, min_area
        self.max_blob_area, self.adaptive_block_size = max_blob_area, adaptive_block_size
        self.body_radius = float(body_radius)
        self.max_foreground_fraction = max_foreground_fraction
        self.compensate_lighting = compensate_lighting
        self.led_anode_direction = led_anode_direction
        # Diagnostics populated on each detect() call.
        self.last_diagnostics: list[dict[str, Any]] = []
        self.last_rejections: list[dict[str, Any]] = []

    @staticmethod
    def median_baseline(frames: Iterable[np.ndarray]) -> np.ndarray:
        """Median of >=3 explicitly supplied, already rectified empty-board frames.

        Copy each frame because camera buffers may be reused by a generator.
        No inspection frame is ever automatically added to this baseline.
        """
        frames = [_validate_image(frame).copy() for frame in frames]
        if len(frames) < 3:
            raise ValueError('Median baseline needs at least three empty-board frames')
        return np.median(np.stack(frames), axis=0).astype(np.uint8)

    def save_baseline(self, filename='empty_baseline.png') -> Path:
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), self.baseline_img):
            raise OSError(f'Cannot write baseline: {path}')
        return path

    def foreground_mask(self, rectified_img):
        """Absolute color difference plus adaptive seeds and connected regions."""
        image = _validate_image(rectified_img)
        difference = cv2.absdiff(image, self.baseline_img)
        if self.compensate_lighting:
            # A small global per-channel exposure offset is estimated robustly
            # from the mostly empty board. Large changes require recapture.
            offset = np.median(image.astype(np.int16)-self.baseline_img.astype(np.int16), axis=(0, 1))
            if np.max(np.abs(offset)) > 15:
                raise BaselineMismatchError('Lighting differs substantially from empty baseline')
            if np.any(offset):
                corrected = np.clip(image.astype(np.float32)-offset, 0, 255).astype(np.uint8)
                difference = cv2.absdiff(corrected, self.baseline_img)
        # Max channel difference preserves isoluminant colored components.
        strength = difference.max(axis=2)
        absolute = np.uint8(strength >= self.min_difference)*255
        if np.mean(absolute > 0) > self.max_foreground_fraction:
            raise BaselineMismatchError('Too much foreground: check registration, lighting, or occlusion')
        adaptive = cv2.adaptiveThreshold(strength, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, self.adaptive_block_size, -3)
        seeds = cv2.bitwise_and(absolute, adaptive)
        # Grow adaptive seeds only within the absolute foreground so uniform
        # resistor bodies remain filled, while preserving fine component leads.
        count, labels, stats, _ = cv2.connectedComponentsWithStats(absolute, connectivity=8)
        seeded = np.zeros(count, dtype=bool)
        seeded[np.unique(labels[seeds > 0])] = True
        seeded[0] = False
        seeded &= stats[:, cv2.CC_STAT_AREA] >= self.min_area
        mask = np.uint8(seeded[labels])*255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        return mask, strength

    def _describe_blob(self, image, mask, strength, offset) -> BlobDescription | None:
        yx = np.argwhere(mask > 0)
        points = yx[:, ::-1].astype(np.float64)
        center = points.mean(axis=0)
        # PCA gives the principal axis and deterministic ordering of extrema.
        _, _, axes = np.linalg.svd(points-center, full_matrices=False)
        axis = axes[0]
        if axis[np.argmax(np.abs(axis))] < 0:
            axis = -axis
        projection = (points-center) @ axis
        extrema = np.array([points[projection <= projection.min()+1.5].mean(axis=0),
                            points[projection >= projection.max()-1.5].mean(axis=0)])
        contour, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rect = cv2.minAreaRect(max(contour, key=cv2.contourArea))
        long_side, short_side = sorted(rect[1], reverse=True)
        if long_side < 10:
            return None
        aspect = long_side/max(short_side, 1)
        # Erode geometrically using a distance transform to separate broad
        # component bodies from thin wire/lead segments.
        distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        body = np.uint8(distance >= self.body_radius)*255
        body_present = bool(np.count_nonzero(body) >= 12)
        kind = 'wire'
        notes: list[str] = []
        if body_present:
            bodies, _ = cv2.findContours(body, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            body_contour = max(bodies, key=cv2.contourArea)
            a, b = sorted(cv2.minAreaRect(body_contour)[1], reverse=True)
            body_aspect = a/max(b, 1)
            perimeter = cv2.arcLength(body_contour, True)
            circularity = 4*np.pi*cv2.contourArea(body_contour)/max(perimeter**2, 1)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            saturated = np.mean((hsv[:, :, 1][body > 0] >= 110) & (hsv[:, :, 2][body > 0] >= 70))
            if body_aspect >= 5 and a/max(long_side, 1) >= .7:
                kind = 'wire'  # thick insulation, uniform along most of the span
            else:
                kind = 'led' if saturated >= .55 and body_aspect < 1.85 and circularity > .60 else 'resistor'
        elif aspect < 2:
            # A tiny compact speck is not a component with two visible leads.
            return None
        skeleton = _thin(mask)
        neighbors = cv2.filter2D(skeleton, cv2.CV_16S, np.ones((3, 3), np.int16))-skeleton
        ends = (skeleton > 0) & (neighbors == 1)
        if kind != 'wire':
            radius = int(np.ceil(self.body_radius)) + 2
            body_extent = cv2.dilate(body, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                                   (2*radius+1, 2*radius+1)))
            ends &= body_extent == 0
        tips = np.argwhere(ends)[:, ::-1].astype(float)
        reliable = len(tips) == 2
        if reliable:
            endpoints = tips
            endpoint_source = 'skeleton'
        else:
            endpoints = extrema
            endpoint_source = 'pca_extrema'
            if kind != 'wire':
                notes.append('leads_not_visible')
            else:
                notes.append('skeleton_endpoints_unreliable')
        endpoints = endpoints[np.argsort((endpoints-center) @ axis)]
        # Extra branches suggest touching parts; avoid asserting reliable pins.
        keypoint_conf = .85 if reliable else .35
        polarity_assumption: str | None = None
        polarity_resolved = False
        if kind == 'led':
            if self.led_anode_direction is None:
                keypoint_conf = min(keypoint_conf, .49)
                polarity_assumption = 'unknown'
                notes.append('polarity_unknown')
            else:
                polarity_assumption = self.led_anode_direction
                dimension = 0 if self.led_anode_direction in ('left', 'right') else 1
                ordering = np.argsort(endpoints[:, dimension])
                if self.led_anode_direction in ('right', 'bottom'):
                    ordering = ordering[::-1]
                endpoints = endpoints[ordering]
                if abs(endpoints[0, dimension]-endpoints[1, dimension]) < 3:
                    keypoint_conf = min(keypoint_conf, .49)
                    notes.append('polarity_axis_too_short')
                else:
                    polarity_resolved = True
        signal = float(np.median(strength[mask > 0])) / 255
        confidence = float(np.clip(.60+.30*signal, .60, .90))
        endpoints = endpoints + np.asarray(offset)
        return BlobDescription(
            kind=kind,
            confidence=confidence,
            endpoints=endpoints,
            keypoint_confidence=keypoint_conf,
            endpoint_source=endpoint_source,
            polarity_assumption=polarity_assumption,
            polarity_resolved=polarity_resolved,
            body_present=body_present,
            aspect_ratio=float(aspect),
            long_side=float(long_side),
            short_side=float(short_side),
            notes=notes,
        )

    def detect(self, rectified_img):
        image = _validate_image(rectified_img)
        mask, strength = self.foreground_mask(image)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: list[dict[str, Any]] = []
        diagnostics: list[dict[str, Any]] = []
        rejections: list[dict[str, Any]] = []
        for contour in sorted(contours, key=lambda c: (cv2.boundingRect(c)[1], cv2.boundingRect(c)[0])):
            x, y, w, h = cv2.boundingRect(contour)
            if cv2.contourArea(contour) > self.max_blob_area:
                raise BaselineMismatchError('Oversized foreground blob: occlusion or merged components')
            # Pad each local mask so distance transforms/thinning see a true
            # background boundary even when the bounding rectangle is filled.
            local = np.zeros((h+2, w+2), np.uint8)
            shifted = contour - np.array([[[x-1, y-1]]], dtype=contour.dtype)
            cv2.drawContours(local, [shifted], -1, 255, cv2.FILLED)
            crop = cv2.copyMakeBorder(image[y:y+h, x:x+w], 1, 1, 1, 1, cv2.BORDER_REPLICATE)
            energy = np.pad(strength[y:y+h, x:x+w], 1)
            description = self._describe_blob(crop, local, energy, (x-1, y-1))
            bbox = (float(x), float(y), float(x+w), float(y+h))
            if description is None:
                rejections.append({
                    'bbox': bbox,
                    'reason': 'blob_short_or_compact',
                    'area': float(cv2.contourArea(contour)),
                })
                continue
            cropped_boundary = bool(x == 0 or y == 0 or x+w == 1400 or y+h == 520)
            if cropped_boundary:
                description.keypoint_confidence = min(description.keypoint_confidence, .35)
                description.notes.append('cropped_at_image_boundary')
            det_id = f'det_{len(detections)+1}'
            detection = {
                'id': det_id,
                'type': description.kind,
                'conf': description.confidence,
                'bbox': bbox,
                'keypoints': [
                    {'x': float(px), 'y': float(py),
                     'conf': description.keypoint_confidence, 'vis': 2}
                    for px, py in description.endpoints
                ],
            }
            detections.append(detection)
            diagnostics.append({
                'detection_id': det_id,
                'kind': description.kind,
                'confidence': description.confidence,
                'keypoint_confidence': description.keypoint_confidence,
                'endpoint_source': description.endpoint_source,
                'polarity_assumption': description.polarity_assumption,
                'polarity_resolved': description.polarity_resolved,
                'body_present': description.body_present,
                'aspect_ratio': description.aspect_ratio,
                'long_side': description.long_side,
                'short_side': description.short_side,
                'cropped_at_image_boundary': cropped_boundary,
                'notes': list(description.notes),
                'endpoints': [
                    {'x': float(px), 'y': float(py)} for px, py in description.endpoints
                ],
                'bbox': bbox,
            })
        self.last_diagnostics = diagnostics
        self.last_rejections = rejections
        return detections

    def annotate(
        self,
        rectified_img: np.ndarray,
        detections: list[dict] | None = None,
        *,
        diagnostics: list[dict] | None = None,
        proposed_pins: list[dict] | None = None,
        show_rejections: bool = True,
    ) -> np.ndarray:
        """Draw a diagnostic overlay on top of ``rectified_img``.

        Parameters mirror :meth:`detect`'s outputs. When ``detections`` is None,
        the detector re-runs on ``rectified_img`` and uses its own last
        diagnostics; otherwise the provided lists are visualised untouched.

        ``proposed_pins`` is an optional list, aligned to ``detections``, with
        shape ``[{"detection_id": str, "pins": {"anode":"d14",...}, ...}]`` as
        produced by :mod:`vision_ml.evaluate_classical`; when supplied the
        proposed hole id is drawn next to each endpoint.
        """
        canvas = _validate_image(rectified_img).copy()
        if detections is None:
            detections = self.detect(canvas)
            diagnostics = self.last_diagnostics
            rejections = self.last_rejections if show_rejections else []
        else:
            diagnostics = diagnostics or []
            rejections = self.last_rejections if show_rejections else []
        pins_by_id = {}
        if proposed_pins:
            for entry in proposed_pins:
                pins_by_id[entry.get('detection_id')] = entry
        colors = {'wire': (220, 120, 20), 'resistor': (20, 140, 220), 'led': (20, 20, 220)}
        for detection in detections:
            diag = next(
                (d for d in diagnostics if d.get('detection_id') == detection['id']), {}
            )
            uncertain = any(kp['conf'] < .50 for kp in detection['keypoints'])
            color = (0, 215, 255) if uncertain else colors.get(detection['type'], (255, 255, 255))
            x1, y1, x2, y2 = (int(round(v)) for v in detection['bbox'])
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            label = f"{detection['id']} {detection['type']} conf={detection['conf']:.2f}"
            cv2.putText(canvas, label, (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1, cv2.LINE_AA)
            if diag.get('notes'):
                cv2.putText(canvas, ','.join(diag['notes'][:3]), (x1, min(519, y2 + 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, .4, color, 1, cv2.LINE_AA)
            pin_entry = pins_by_id.get(detection['id'], {})
            pin_map = pin_entry.get('pins') or {}
            # Preserve a stable pin-name ordering so the UI reads anode/cathode first.
            pin_names = list(pin_map.keys())
            for idx, kp in enumerate(detection['keypoints']):
                cx, cy = int(round(kp['x'])), int(round(kp['y']))
                cv2.drawMarker(canvas, (cx, cy), color, cv2.MARKER_CROSS, 10, 2)
                tag = f"({cx},{cy}) kp={kp['conf']:.2f}"
                if idx < len(pin_names):
                    pin_name = pin_names[idx]
                    tag = f"{pin_name}->{pin_map[pin_name]} " + tag
                cv2.putText(canvas, tag, (cx + 4, cy - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, .35, color, 1, cv2.LINE_AA)
        for rej in rejections:
            x1, y1, x2, y2 = (int(round(v)) for v in rej['bbox'])
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (120, 120, 120), 1)
            cv2.putText(canvas, f"rejected: {rej['reason']}", (x1, max(0, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, .35, (120, 120, 120), 1, cv2.LINE_AA)
        return canvas

    __call__ = detect
