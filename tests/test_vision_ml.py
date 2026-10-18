"""Vision-only tests. All temporary artifacts stay under vision_ml/."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vision_ml import (BreadboardRegistration, BreadboardPoseDetector,
                       QualityGateError, generate_synthetic_aruco_board_image)
from vision_ml.calibrate_cli import save_marker_sheet, save_calibration, create_hole_lut
from vision_ml.detector import letterbox
from vision_ml.pi_hardware import BreadboardCamera, TrafficLight
from vision_ml.prelabel import pose_label, prelabel_images
from vision_ml.split_dataset import build_id, split_by_build, split_dataset


@pytest.fixture
def workdir():
    with tempfile.TemporaryDirectory(prefix='test_', dir=ROOT / 'vision_ml') as directory:
        yield Path(directory)


@pytest.fixture
def scene():
    return generate_synthetic_aruco_board_image()


def test_real_synthetic_aruco_registration(scene):
    registration = BreadboardRegistration()
    result = registration.register(scene)
    assert set(result.marker_centers) == {0, 1, 2, 3}
    assert result.rectified.shape == (520, 1400, 3)
    assert result.quality.passed
    points = np.float32([[result.marker_centers[i] for i in range(4)]])
    mapped = cv2.perspectiveTransform(points, result.homography)[0]
    np.testing.assert_allclose(mapped, [[0, 0], [1399, 0], [1399, 519], [0, 519]], atol=.001)
    assert registration.pixels_per_mm == 8.0


def test_perspective_registration(scene):
    h, w = scene.shape[:2]
    transform = cv2.getPerspectiveTransform(np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]]),
                                            np.float32([[45, 30], [w-50, 5], [w-10, h-35], [10, h-10]]))
    image = cv2.warpPerspective(scene, transform, (w, h), borderValue=(180, 180, 180))
    result = BreadboardRegistration().register(image)
    assert result.rectified.shape == (520, 1400, 3)
    assert result.quality.passed


def test_missing_marker_rejected(scene):
    scene[20:110, 20:110] = 180
    with pytest.raises(QualityGateError) as exc:
        BreadboardRegistration().register(scene)
    assert exc.value.result.quality.failures == ('markers',)
    assert exc.value.result.rectified is None


@pytest.mark.parametrize('defect', ['sharpness', 'brightness', 'glare'])
def test_quality_gate_rejects_defect(scene, defect):
    if defect == 'sharpness':
        scene = cv2.GaussianBlur(scene, (13, 13), 2.5)
    elif defect == 'brightness':
        scene = (scene.astype(float)*.5).astype(np.uint8)
    else:
        scene[220:320, 400:900] = 255
    with pytest.raises(QualityGateError) as exc:
        BreadboardRegistration().register(scene)
    assert defect in exc.value.result.quality.failures


def test_gate_boundaries():
    registration = BreadboardRegistration(brightness_baseline=100, sharpness_threshold=0)
    assert registration.quality_report(np.full((100, 100), 85, np.uint8)).passed
    assert registration.quality_report(np.full((100, 100), 115, np.uint8)).passed
    assert 'brightness' in registration.quality_report(np.full((100, 100), 84, np.uint8)).failures
    image = np.full((100, 100), 100, np.uint8)
    image.ravel()[:199] = 250
    assert registration.quality_report(image).passed
    image.ravel()[199] = 250
    assert 'glare' in registration.quality_report(image).failures


def test_exact_letterbox_roundtrip():
    padded, transform = letterbox(np.zeros((520, 1400, 3), np.uint8))
    assert padded.shape == (480, 1280, 3)
    assert transform.resized_size == (1280, 475)
    assert transform.padding == (0, 2)
    points = np.vstack(([[0, 0], [1399, 519], [500.25, 201.75]],
                        np.random.default_rng(12).random((1000, 2))*[1399, 519]))
    restored = transform.unmap_points(transform.map_points(points))
    assert np.abs(restored-points).max() < .01
    np.testing.assert_array_equal(padded[:2], 114)
    np.testing.assert_array_equal(padded[-3:], 114)


class FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value, dtype=float)

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class FakeBoxes:
    def __init__(self, value):
        self.data = FakeTensor(value)

    def __len__(self):
        return len(self.data.value)


def test_detector_filters_and_unmaps():
    image = np.zeros((520, 1400, 3), np.uint8)
    _, transform = letterbox(image)
    original = np.array([[123.456, 87.654], [1234.567, 456.789]])
    mapped = transform.map_points(original)
    boxes = [[*mapped.ravel(), .92, 2], [*mapped.ravel(), .39, 0],
             [*mapped.ravel(), .95, 1], [*mapped.ravel(), .95, 1]]
    points = [[[ *mapped[0], .9], [*mapped[1], .8]],
              [[ *mapped[0], .9], [*mapped[1], .8]],
              [[ *mapped[0], .49], [*mapped[1], .8]],
              [[ *mapped[0], .9], [*mapped[1], float('nan')]]]
    calls = []

    def predict(source, **kwargs):
        calls.append((source.shape, kwargs))
        return [SimpleNamespace(boxes=FakeBoxes(boxes),
                                keypoints=SimpleNamespace(data=FakeTensor(points)),
                                names={0: 'wire', 1: 'resistor', 2: 'led'})]

    detector = BreadboardPoseDetector()
    detector.model = SimpleNamespace(predict=predict)
    detections = detector(image)
    assert len(detections) == 1
    det = detections[0]
    assert set(det) == {'id', 'type', 'conf', 'bbox', 'keypoints'}
    assert det['type'] == 'led' and det['id'] == 'det_1'
    assert isinstance(det['bbox'], tuple)
    np.testing.assert_allclose(det['bbox'], original.ravel(), atol=.01)
    np.testing.assert_allclose([[p['x'], p['y']] for p in det['keypoints']], original, atol=.01)
    assert all(p['vis'] == 2 for p in det['keypoints'])
    assert calls[0][1]['conf'] == .4 and calls[0][1]['iou'] == .5
    assert calls[0][1]['imgsz'] == (480, 1280) and not calls[0][1]['rect']
    label = pose_label(det, 1400, 520).split()
    assert len(label) == 11 and label[0] == '2'
    assert float(label[7]) == float(label[10]) == 2
    assert abs(float(label[5])-original[0, 0]/1400) < 1e-8


def test_mock_and_invalid_model(workdir):
    detector = BreadboardPoseDetector()
    assert detector.mock
    assert detector(np.zeros((520, 1400, 3), np.uint8)) == []
    with pytest.raises(FileNotFoundError):
        BreadboardPoseDetector(workdir / 'missing.pt')
    with pytest.raises(ValueError):
        BreadboardPoseDetector(kpt_conf=1.1)


def test_build_grouped_split():
    images = [Path(f'build{build:02d}__photo{photo}.png')
              for build in range(10) for photo in range(3)]
    splits = split_by_build(images, seed=7)
    assert [len(splits[name]) for name in ('train', 'val', 'test')] == [21, 6, 3]
    memberships = [{build_id(p) for p in paths} for paths in splits.values()]
    assert all(not memberships[i] & memberships[j] for i in range(3) for j in range(i))
    assert set(sum(splits.values(), [])) == set(images)
    assert splits == split_by_build(reversed(images), seed=7)
    assert splits != split_by_build(images, seed=8)
    assert all(split_by_build(images[::12]).values())
    assert build_id('build_001__photo.png', r'^(?P<build>.+)__') == 'build_001'
    with pytest.raises(ValueError):
        build_id('unnamed.png')


def test_dataset_copy_and_missing_labels(workdir):
    images, labels, output = [workdir / name for name in ('images', 'labels', 'dataset')]
    images.mkdir()
    labels.mkdir()
    for i in range(10):
        (images / f'b{i}__p.png').write_bytes(b'image')
        (labels / f'b{i}__p.txt').write_text('')
    split_dataset(images, labels, output)
    assert len(list((output / 'images').rglob('*.png'))) == 10
    assert len(list((output / 'labels').rglob('*.txt'))) == 10
    assert set(json.loads((output / 'split_manifest.json').read_text())) == {'train', 'val', 'test'}
    with pytest.raises(FileExistsError):
        split_dataset(images, labels, output)
    (labels / 'b0__p.txt').unlink()
    with pytest.raises(FileNotFoundError):
        split_dataset(images, labels, workdir / 'new')
    assert not (workdir / 'new').exists()


def test_marker_print_and_calibration(workdir, scene):
    sheet = save_marker_sheet(workdir / 'markers.png', marker_size=120)
    registration = BreadboardRegistration()
    assert set(registration.detect_markers(cv2.imread(str(sheet)))) == {0, 1, 2, 3}
    points = {'a1': (70, 100), 'a63': (1330, 100), 'j1': (70, 420), 'j63': (1330, 420)}
    image = registration.register(scene).rectified
    calibration, lut = save_calibration(image, points, workdir)
    assert len(lut) == 630
    for name, xy in points.items():
        np.testing.assert_allclose([lut[name]['x'], lut[name]['y']], xy, atol=.001)
    assert lut['f1']['y']-lut['e1']['y'] == pytest.approx(3*(lut['b1']['y']-lut['a1']['y']), abs=.001)
    assert calibration['brightness_baseline'] == pytest.approx(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).mean())
    assert (workdir / 'calibration.json').is_file()
    assert (workdir / 'hole_lut.json').is_file()
    loaded = BreadboardRegistration.from_calibration(workdir / 'calibration.json')
    assert loaded.brightness_baseline == calibration['brightness_baseline']
    assert loaded.register(scene).quality.passed
    with pytest.raises(ValueError):
        create_hole_lut(dict.fromkeys(points, (10, 10)))


def test_prelabel_preserves_existing(workdir, monkeypatch):
    from vision_ml import prelabel
    images, output = workdir / 'images', workdir / 'labels'
    images.mkdir()
    output.mkdir()
    cv2.imwrite(str(images / 'one.png'), np.zeros((520, 1400, 3), np.uint8))
    cv2.imwrite(str(images / 'two.png'), np.zeros((520, 1400, 3), np.uint8))
    (output / 'one.txt').write_text('human label\n')
    monkeypatch.setattr(prelabel, 'BreadboardPoseDetector', lambda *a, **kw: lambda image: [])
    counts = prelabel_images('checkpoint.pt', images, output)
    assert counts == {'written': 1, 'skipped': 1}
    assert (output / 'one.txt').read_text() == 'human label\n'
    assert (output / 'two.txt').read_text() == ''


def test_local_camera_sharpest_burst_and_gpio(workdir, capsys):
    sharp = np.zeros((100, 100, 3), np.uint8)
    sharp[::2] = 200
    frames = [cv2.GaussianBlur(sharp, (9, 9), 3), sharp, np.full_like(sharp, 100)]
    paths = []
    for i, frame in enumerate(frames):
        path = workdir / f'{i}.png'
        cv2.imwrite(str(path), frame)
        paths.append(path)
    with BreadboardCamera(paths, force_fallback=True) as camera:
        np.testing.assert_array_equal(camera.capture_burst(), sharp)
    with TrafficLight(force_fallback=True) as lights:
        lights.set('green')
        assert lights.state == 'green'
        assert '[GPIO] GREEN' in capsys.readouterr().out
        with pytest.raises(ValueError):
            lights.set('blue')


def test_training_settings_and_best_export(workdir, monkeypatch):
    from vision_ml.train_pose import train_and_export
    best = workdir / 'best.pt'
    best.touch()
    calls = []

    class YOLO:
        def __init__(self, source):
            calls.append(('load', source))
            self.trainer = SimpleNamespace(best=best)

        def train(self, **kwargs):
            calls.append(('train', kwargs))

        def export(self, **kwargs):
            calls.append(('export', kwargs))
            return 'ncnn_folder'

    monkeypatch.setitem(sys.modules, 'ultralytics', SimpleNamespace(YOLO=YOLO))
    assert train_and_export() == (best, 'ncnn_folder')
    args = calls[1][1]
    for key, value in dict(imgsz=1280, epochs=100, batch=8, close_mosaic=10,
                           degrees=5, translate=.05, scale=.1,
                           hsv_v=.3, hsv_s=.4, hsv_h=.01).items():
        assert args[key] == value
    assert calls[2] == ('load', str(best))
    assert calls[3] == ('export', {'format': 'ncnn', 'imgsz': (480, 1280)})


def test_pi_locks_controls_and_falls_back_on_capture_failure(workdir, monkeypatch):
    frame = np.full((32, 32, 3), 120, np.uint8)
    fallback = workdir / 'fallback.png'
    cv2.imwrite(str(fallback), frame)

    class Pi:
        def __init__(self):
            self.camera_controls = dict.fromkeys(['AfMode', 'AeEnable', 'AwbEnable'])
            self.count = 0
            self.closed = False

        def create_still_configuration(self, **kwargs):
            self.config = kwargs
            return kwargs

        def configure(self, config):
            pass

        def start(self):
            pass

        def capture_metadata(self):
            return {'ExposureTime': 10000, 'AnalogueGain': 1.5, 'ColourGains': (1.2, 1.1)}

        def set_controls(self, controls):
            self.controls = controls

        def capture_array(self, name):
            self.count += 1
            if self.count > 3:
                raise OSError('camera disconnected')
            return frame.copy()

        def stop(self):
            pass

        def close(self):
            self.closed = True

    pi = Pi()
    monkeypatch.setitem(sys.modules, 'picamera2', SimpleNamespace(Picamera2=lambda: pi))
    with BreadboardCamera([fallback], warmup_seconds=0) as camera:
        assert not pi.controls['AeEnable'] and not pi.controls['AwbEnable']
        assert pi.controls['AfMode'] == 0
        assert pi.controls['ExposureTime'] == 10000
        assert pi.controls['AnalogueGain'] == 1.5
        assert pi.count == 3  # queued unlocked frames drained
        np.testing.assert_array_equal(camera.capture_burst(), frame)
        assert camera.camera is None and pi.closed


def test_gpio_pin_mapping_and_exclusive_lights(monkeypatch):
    leds = {}

    class LED:
        def __init__(self, pin):
            self.on_state = False
            self.closed = False
            leds[pin] = self

        def on(self):
            self.on_state = True

        def off(self):
            self.on_state = False

        def close(self):
            self.closed = True

    monkeypatch.setitem(sys.modules, 'gpiozero', SimpleNamespace(LED=LED))
    with TrafficLight() as traffic:
        traffic.set('red')
        assert leds[17].on_state
        traffic.set('green')
        assert leds[22].on_state and not leds[17].on_state and not leds[27].on_state
    assert all(led.closed for led in leds.values())


def _classical_scene():
    from engine import BoardSpec, HoleLUT
    lut = HoleLUT.generate_synthetic(BoardSpec.full_830())
    baseline = np.full((520, 1400, 3), 205, np.uint8)
    for x, y in lut.holes.values():
        cv2.circle(baseline, (round(x), round(y)), 2, (75, 75, 75), -1)
    image = baseline.copy()
    expected = {'wire': ('a5', 'a10'), 'resistor': ('d20', 'd25'), 'led': ('h40', 'h42')}
    for kind, holes in expected.items():
        p, q = [tuple(np.rint(lut.holes[hole]).astype(int)) for hole in holes]
        center = tuple(np.rint((np.asarray(p)+q)/2).astype(int))
        if kind == 'wire':
            cv2.line(image, p, q, (220, 45, 40), 5)
        else:
            cv2.line(image, p, q, (35, 35, 35), 3)
            if kind == 'resistor':
                x, y = center
                cv2.rectangle(image, (x-22, y-8), (x+22, y+8), (110, 165, 195), -1)
                for dx in (-12, 0, 12):
                    cv2.line(image, (x+dx, y-8), (x+dx, y+8), (30, 50, 70), 3)
            else:
                cv2.circle(image, center, 10, (35, 35, 230), -1)
    return baseline, image, lut, expected


def test_classical_components_detect_and_snap():
    from engine import RawDetection, snap_detections
    from vision_ml.classical_detector import ClassicalBreadboardDetector
    baseline, image, lut, expected = _classical_scene()
    before = image.copy()
    detector = ClassicalBreadboardDetector(baseline, led_anode_direction='left')
    detections = detector.detect(image)
    assert len(detections) == 3
    assert {d['type'] for d in detections} == set(expected)
    raw = [RawDetection.model_validate(d) for d in detections]
    snapped, findings = snap_detections(raw, lut)
    assert findings == []
    for part in snapped:
        assert tuple(part.pins.values()) == expected[part.type]
        assert part.ambiguous_pins == []
    assert detector.detect(baseline) == []
    assert detector.detect(image) == detections  # deterministic order and scores
    np.testing.assert_array_equal(image, before)


def test_classical_diagonal_and_vertical_wires_snap():
    from engine import RawDetection, snap_detections
    from vision_ml.classical_detector import ClassicalBreadboardDetector
    baseline, _, lut, _ = _classical_scene()
    image = baseline.copy()
    expected = [('a10', 'e10'), ('f20', 'j25')]
    for a, b in expected:
        cv2.line(image, tuple(np.rint(lut.holes[a]).astype(int)),
                 tuple(np.rint(lut.holes[b]).astype(int)), (20, 180, 20), 5)
    detections = ClassicalBreadboardDetector(baseline).detect(image)
    assert len(detections) == 2 and all(d['type'] == 'wire' for d in detections)
    parts, findings = snap_detections([RawDetection.model_validate(d) for d in detections], lut)
    assert findings == []
    assert {frozenset(p.pins.values()) for p in parts} == {frozenset(p) for p in expected}


def test_classical_led_parallel_leads_and_unknown_polarity():
    from engine import RawDetection, snap_detections
    from vision_ml.classical_detector import ClassicalBreadboardDetector
    baseline, _, lut, _ = _classical_scene()
    image = baseline.copy()
    a, b = [np.rint(lut.holes[h]).astype(int) for h in ('h40', 'h41')]
    center = ((a[0]+b[0])//2, a[1]-35)
    for tip in (a, b):
        cv2.line(image, (tip[0], center[1]), tuple(tip), (35, 35, 35), 3)
    cv2.circle(image, center, 12, (25, 25, 230), -1)
    known = ClassicalBreadboardDetector(baseline, led_anode_direction='left').detect(image)
    assert len(known) == 1 and known[0]['type'] == 'led'
    parts, findings = snap_detections([RawDetection.model_validate(d) for d in known], lut)
    assert not findings
    assert parts[0].pins == {'anode': 'h40', 'cathode': 'h41'}
    unknown = ClassicalBreadboardDetector(baseline).detect(image)
    _, findings = snap_detections([RawDetection.model_validate(d) for d in unknown], lut)
    assert any(f.code == 'UNCERTAIN' for f in findings)
    assert all(k['conf'] < .50 for k in unknown[0]['keypoints'])


def test_classical_median_baseline_save_load_and_noise(workdir):
    from vision_ml.classical_detector import ClassicalBreadboardDetector
    baseline, image, _, _ = _classical_scene()
    contaminated = baseline.copy()
    cv2.line(contaminated, (100, 100), (400, 100), (0, 0, 0), 5)
    detector = ClassicalBreadboardDetector(baseline_frames=[baseline, contaminated, baseline],
                                           led_anode_direction='left')
    np.testing.assert_array_equal(detector.baseline_img, baseline)
    filename = detector.save_baseline(workdir / 'empty_baseline.png')
    loaded = ClassicalBreadboardDetector(baseline_path=filename, led_anode_direction='left')
    assert loaded.detect(image) == detector.detect(image)
    rng = np.random.default_rng(42)
    noise = rng.integers(-4, 5, baseline.shape)
    noisy = np.clip(baseline.astype(int)+noise+7, 0, 255).astype(np.uint8)
    # Isolated high-contrast sensor speckles are below minimum component area.
    noisy[490, 1300] = 0
    assert loaded.detect(noisy) == []


def test_classical_rejects_invalid_baseline_and_occlusion(workdir):
    from vision_ml.classical_detector import ClassicalBreadboardDetector, BaselineMismatchError
    baseline, _, _, _ = _classical_scene()
    with pytest.raises(FileNotFoundError):
        ClassicalBreadboardDetector(baseline_path=workdir / 'missing.png')
    with pytest.raises(ValueError, match='three'):
        ClassicalBreadboardDetector(baseline_frames=[baseline])
    detector = ClassicalBreadboardDetector(baseline)
    with pytest.raises(ValueError, match='shape'):
        detector.detect(np.zeros((100, 100, 3), np.uint8))
    occluded = baseline.copy()
    occluded[100:450, 100:1100] = 0
    with pytest.raises(BaselineMismatchError):
        detector.detect(occluded)
    with pytest.raises(BaselineMismatchError):
        detector.detect((baseline*.5).astype(np.uint8))


def test_classical_thick_wire_and_rotated_resistor():
    from engine import RawDetection, snap_detections
    from vision_ml.classical_detector import ClassicalBreadboardDetector
    baseline, _, lut, _ = _classical_scene()
    image = baseline.copy()
    pairs = [('a5', 'a15'), ('f35', 'j40')]
    for kind, pair in zip(('wire', 'resistor'), pairs):
        p, q = [np.rint(lut.holes[hole]).astype(int) for hole in pair]
        cv2.line(image, tuple(p), tuple(q), (220, 20, 20) if kind == 'wire' else (30, 30, 30),
                 13 if kind == 'wire' else 3)
        if kind == 'resistor':
            center = (p+q)/2
            angle = float(np.degrees(np.arctan2(*(q-p)[::-1])))
            body = cv2.boxPoints((tuple(map(float, center)), (44., 17.), angle))
            cv2.fillConvexPoly(image, np.rint(body).astype(np.int32), (90, 150, 180))
    detections = ClassicalBreadboardDetector(baseline).detect(image)
    assert [d['type'] for d in detections] == ['wire', 'resistor']
    parts, findings = snap_detections([RawDetection.model_validate(d) for d in detections], lut)
    assert findings == []
    assert {frozenset(part.pins.values()) for part in parts} == {frozenset(pair) for pair in pairs}
