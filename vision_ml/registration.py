"""ArUco registration and deterministic pre-inference quality gates.

Marker IDs run clockwise: 0=top left, 1=top right, 2=bottom right,
3=bottom left. Marker *centers* define the rectified pixel-center rectangle.
The physical marker-center positions can be supplied in millimetres.
"""
from dataclasses import dataclass
import hashlib
import json

import cv2
import numpy as np

PIXELS_PER_MM = 8.0
BOARD_SIZE = (1400, 520)  # width, height


def _gray(image):
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ValueError('Expected a uint8 BGR or grayscale image')
    if image.size == 0 or image.ndim not in (2, 3):
        raise ValueError('Expected a nonempty image')
    if image.ndim == 3 and image.shape[2] != 3:
        raise ValueError('Expected three BGR channels')
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image


@dataclass(frozen=True)
class QualityReport:
    markers_ok: bool
    sharpness: float
    brightness: float
    glare_fraction: float
    failures: tuple[str, ...]

    @property
    def passed(self):
        return not self.failures


@dataclass
class RegistrationResult:
    rectified: np.ndarray | None
    homography: np.ndarray | None
    marker_centers: dict
    quality: QualityReport


class QualityGateError(ValueError):
    def __init__(self, result):
        self.result = result
        super().__init__('Quality gate failed: ' + ', '.join(result.quality.failures))


class BreadboardRegistration:
    def __init__(self, brightness_baseline=180.0, sharpness_threshold=100.0,
                 pixels_per_mm=PIXELS_PER_MM, output_size=BOARD_SIZE,
                 marker_positions_mm=None):
        if (not np.isfinite([brightness_baseline, sharpness_threshold]).all()
                or not 0 < brightness_baseline <= 255 or sharpness_threshold < 0):
            raise ValueError('Invalid brightness baseline or sharpness threshold')
        if (not np.isfinite(pixels_per_mm) or pixels_per_mm <= 0
                or len(output_size) != 2 or not np.isfinite(output_size).all()
                or min(output_size) < 2 or any(int(n) != n for n in output_size)):
            raise ValueError('Invalid metric scale or output size')
        self.brightness_baseline = float(brightness_baseline)
        self.sharpness_threshold = float(sharpness_threshold)
        self.pixels_per_mm = float(pixels_per_mm)
        self.output_size = tuple(map(int, output_size))
        w, h = self.output_size
        if marker_positions_mm is None:
            marker_positions_mm = {0: (0, 0), 1: ((w-1)/pixels_per_mm, 0),
                                   2: ((w-1)/pixels_per_mm, (h-1)/pixels_per_mm),
                                   3: (0, (h-1)/pixels_per_mm)}
        if set(marker_positions_mm) != {0, 1, 2, 3}:
            raise ValueError('Marker positions must contain exactly IDs 0, 1, 2, 3')
        self.destination = np.asarray([marker_positions_mm[i] for i in range(4)],
                                      dtype=np.float32) * pixels_per_mm
        if self.destination.shape != (4, 2) or not np.isfinite(self.destination).all():
            raise ValueError('Four finite marker positions are required')
        if (not cv2.isContourConvex(self.destination)
                or cv2.contourArea(self.destination, oriented=True) <= 1e-6):
            raise ValueError('Marker positions must form a convex clockwise quadrilateral')
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())

    @classmethod
    def from_calibration(cls, filename):
        """Load the calibration CLI output as quality/geometry configuration."""
        import json
        from pathlib import Path
        data = json.loads(Path(filename).read_text())
        positions = data.get('marker_positions_mm')
        if positions is not None:
            positions = {int(key): value for key, value in positions.items()}
        return cls(brightness_baseline=data['brightness_baseline'],
                   sharpness_threshold=data.get('sharpness_threshold', 100.0),
                   pixels_per_mm=data.get('pixels_per_mm', PIXELS_PER_MM),
                   output_size=tuple(data.get('rectified_size', BOARD_SIZE)),
                   marker_positions_mm=positions)

    def detect_markers(self, image):
        corners, ids, _ = self.detector.detectMarkers(_gray(image))
        centers = {}
        if ids is not None:
            for marker, marker_id in zip(corners, ids.ravel()):
                marker_id = int(marker_id)
                if marker_id in range(4):
                    if marker_id in centers:
                        raise ValueError(f'Duplicate registration marker ID {marker_id}')
                    centers[marker_id] = np.asarray(marker).reshape(4, 2).mean(axis=0)
        return centers

    def quality_report(self, rectified, markers_ok=True):
        gray = _gray(rectified)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(gray.mean())
        glare = float(np.mean(gray >= 250))
        failures = []
        if not markers_ok:
            failures.append('markers')
        if sharpness < self.sharpness_threshold:
            failures.append('sharpness')
        if abs(brightness - self.brightness_baseline) > self.brightness_baseline * .15 + 1e-9:
            failures.append('brightness')
        if glare >= .02:
            failures.append('glare')
        return QualityReport(markers_ok, sharpness, brightness, glare, tuple(failures))

    def register(self, image, *, enforce_quality=True):
        _gray(image)
        centers = self.detect_markers(image)
        homography = rectified = None
        if set(centers) == {0, 1, 2, 3}:
            source = np.asarray([centers[i] for i in range(4)], dtype=np.float32)
            if (not cv2.isContourConvex(source)
                    or cv2.contourArea(source, oriented=True) <= 1e-6):
                raise ValueError('Marker arrangement must be clockwise and nondegenerate')
            homography = cv2.getPerspectiveTransform(source, self.destination)
            if not np.isfinite(homography).all() or abs(np.linalg.det(homography)) < 1e-12:
                raise ValueError('Degenerate registration homography')
            rectified = cv2.warpPerspective(image, homography, self.output_size)
            quality = self.quality_report(rectified)
        else:
            # Board geometry is unknown: do not report misleading whole-frame metrics.
            quality = QualityReport(False, float('nan'), float('nan'), float('nan'), ('markers',))
        result = RegistrationResult(rectified, homography, centers, quality)
        if enforce_quality and not quality.passed:
            raise QualityGateError(result)
        return result



def content_fingerprint(value):
    """Stable SHA-256 of finite JSON data, independent of key insertion order."""
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(encoded.encode('utf-8')).hexdigest()


def calibration_identity(calibration, hole_lut):
    """Bind setup, registration, thresholds, topology, and actual hole positions.

    Works for legacy calibration dictionaries too; no identity is inferred from
    a filename. Changing any calibration field or LUT coordinate changes the ID.
    """
    payload = {key: value for key, value in calibration.items() if key != 'calibration_id'}
    return content_fingerprint({'calibration': payload, 'hole_lut': hole_lut})


def generate_synthetic_aruco_board_image():
    """Return a BGR test scene with real markers and a 63-column breadboard.

    Marker centers have 1399x519 pixel separation, matching the default mapping.
    A light-gray margin supplies each marker with a detection quiet zone.
    """
    image = np.full((648, 1528, 3), 180, dtype=np.uint8)
    for row in range(10):
        y = round(180 + row * 30)
        for column in range(63):
            x = round(135 + column * 20.32)
            cv2.circle(image, (x, y), 3, (45, 45, 45), -1)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    for marker_id, (x, y) in enumerate([(64, 64), (1463, 64), (1463, 583), (64, 583)]):
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, 64)
        image[y-32:y+32, x-32:x+32] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    return image
