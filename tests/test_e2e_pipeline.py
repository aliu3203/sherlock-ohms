"""End-to-end registration, pose adapter, engine/VF2 and traffic-light checks."""
from copy import deepcopy
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from engine.board import BoardSpec, HoleLUT
from engine.schemas import RawDetection, ReferenceCircuit
from pipeline import AMBER, GREEN, RED, SherlockOhmsPipeline
from vision_ml.detector import BreadboardPoseDetector, letterbox
from vision_ml.pi_hardware import TrafficLight
from vision_ml.registration import generate_synthetic_aruco_board_image


class RecordingLight(TrafficLight):
    """Use the actual laptop GPIO fallback and record transitions."""
    def __init__(self):
        super().__init__(force_fallback=True)
        self.history = []

    def set(self, color):
        super().set(color)
        self.history.append(self.state.upper())


def make_detection(lut, part_id, kind, holes, jitter=(0, 0)):
    xy = np.asarray([lut.holes[h] for h in holes]) + jitter
    return {'id': part_id, 'type': kind, 'conf': .99,
            'bbox': (*np.min(xy, axis=0)-5, *np.max(xy, axis=0)+5),
            'keypoints': [{'x': float(x), 'y': float(y), 'conf': .99, 'vis': 2} for x, y in xy]}


@pytest.fixture
def setup():
    board = BoardSpec.full_830()
    lut = HoleLUT.generate_synthetic(board)
    reference = ReferenceCircuit(
        name='golden LED and resistor', rails={'VCC': 'top_plus', 'GND': 'top_minus'},
        parts=[
            {'id': 'W_VCC', 'type': 'wire', 'pins': {'0': 'top_plus_5', '1': 'a5'}},
            {'id': 'R1', 'type': 'resistor', 'pins': {'0': 'a5', '1': 'a10'}},
            {'id': 'LED1', 'type': 'led', 'pins': {'anode': 'd10', 'cathode': 'd15'}},
            {'id': 'W_GND', 'type': 'wire', 'pins': {'0': 'a15', '1': 'top_minus_15'}},
        ])
    detections = [make_detection(lut, p['id'], p['type'], list(p['pins'].values()), jitter=(1.5, -1))
                  for p in reference.parts]
    return reference, lut, detections


def make_pipeline(setup, **kwargs):
    reference, lut, _ = setup
    light = RecordingLight()
    return SherlockOhmsPipeline(reference, lut=lut, traffic_light=light, **kwargs), light


class Tensor:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float64)

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class Boxes:
    def __init__(self, values):
        self.data = Tensor(values)

    def __len__(self):
        return len(self.data.values)


class GoldenPoseBackend:
    """Deterministic model output; real detector letterboxing/unmapping still runs."""
    def __init__(self, detections):
        self.detections = detections
        self.calls = []

    def predict(self, image, **kwargs):
        self.calls.append((image.shape, kwargs))
        _, transform = letterbox(np.zeros((520, 1400, 3), np.uint8))
        classes = {'wire': 0, 'resistor': 1, 'led': 2}
        boxes, keypoints = [], []
        for det in self.detections:
            bbox = transform.map_points(np.array(det['bbox']).reshape(2, 2)).ravel()
            xy = transform.map_points([[p['x'], p['y']] for p in det['keypoints']])
            boxes.append([*bbox, det['conf'], classes[det['type']]])
            keypoints.append([[*p, k['conf']] for p, k in zip(xy, det['keypoints'])])
        return [SimpleNamespace(boxes=Boxes(boxes), keypoints=SimpleNamespace(data=Tensor(keypoints)),
                                names={0: 'wire', 1: 'resistor', 2: 'led'})]


def test_e2e_exact_match_green(setup, monkeypatch):
    import engine.verifier as verifier
    calls = []
    original = verifier.GraphMatcher

    def graph_matcher(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(verifier, 'GraphMatcher', graph_matcher)
    detector = BreadboardPoseDetector()
    detector.model = GoldenPoseBackend(setup[2])
    pipeline, light = make_pipeline(setup, detector=detector)
    raw = generate_synthetic_aruco_board_image()
    untouched = raw.copy()
    result = pipeline.inspect(raw)
    assert result.passed is True and result.verdict.passed is True
    assert result.light_color == 'GREEN' and light.state.upper() == 'GREEN'
    assert light.history == ['YELLOW', 'GREEN']
    assert set(result.verdicts) == {'placement', 'connectivity'}
    assert all(v.passed for v in result.verdicts.values())
    assert calls  # real NetworkX VF2 was reached
    assert all(isinstance(d, RawDetection) for d in result.detections)
    assert result.quality.passed and not result.aborted
    assert result.annotated_image.shape == (520, 1400, 3)
    assert result.annotated_image.dtype == np.uint8
    assert not np.shares_memory(result.annotated_image, result.rectified_image)
    assert np.any(result.annotated_image != result.rectified_image)
    np.testing.assert_array_equal(raw, untouched)
    assert detector.model.calls[0][0] == (480, 1280, 3)
    assert detector.model.calls[0][1]['iou'] == .5
    for original_det, restored in zip(setup[2], result.detections):
        np.testing.assert_allclose(restored.bbox, original_det['bbox'], atol=.01)
    assert np.any(np.all(result.annotated_image == GREEN, axis=-1))


def test_e2e_equivalent_layout_green(setup):
    _, lut, _ = setup
    # Shift every column +15, move the LED across the trench, and bridge its
    # anode net with an extra e25 -> f25 jumper. This is a real center-gap bridge.
    shifted = [
        make_detection(lut, 'W_VCC', 'wire', ['top_plus_20', 'a20']),
        make_detection(lut, 'R1', 'resistor', ['a20', 'a25']),
        make_detection(lut, 'W_BRIDGE', 'wire', ['e25', 'f25']),
        make_detection(lut, 'LED1', 'led', ['i25', 'i30']),
        make_detection(lut, 'W_GND', 'wire', ['j30', 'top_minus_30']),
    ]
    pipeline, light = make_pipeline(setup)
    placement = pipeline.inspect(mode='placement', detections=shifted)
    assert placement.passed is False and light.state.upper() == 'RED'
    assert 'MISPLACED_PIN' in {f.code for f in placement.findings}
    assert placement.verdicts['connectivity'].passed
    connectivity = pipeline.inspect(mode='connectivity', detections=shifted)
    assert connectivity.passed is True and light.state.upper() == 'GREEN'
    assert connectivity.findings == []
    assert light.history == ['YELLOW', 'RED', 'YELLOW', 'GREEN']


def test_e2e_off_by_one_row_red(setup):
    _, lut, detections = setup
    detections = deepcopy(detections)
    detections[2] = make_detection(lut, 'LED1', 'led', ['d10', 'd16'])
    pipeline, light = make_pipeline(setup)
    result = pipeline.inspect(detections=detections)
    assert result.passed is False
    assert result.light_color == 'RED' and light.state.upper() == 'RED'
    assert 'OPEN_CONNECTION' in {f.code for f in result.findings}
    assert np.any(np.all(result.annotated_image == RED, axis=-1))


def test_e2e_power_short_red(setup):
    _, lut, detections = setup
    shorted = detections + [make_detection(lut, 'W_SHORT', 'wire', ['top_plus_40', 'top_minus_40'])]
    pipeline, light = make_pipeline(setup)
    result = pipeline.inspect(detections=shorted)
    assert result.passed is False and light.state.upper() == 'RED'
    assert 'POWER_SHORT' in {f.code for f in result.findings}
    assert not result.verdicts['placement'].passed


@pytest.mark.parametrize('defect', ['blur', 'dark', 'missing_marker', 'glare'])
def test_e2e_quality_gate_abort(setup, defect):
    class NeverInfer:
        def detect(self, image):
            pytest.fail('Inference must not run after failed quality gates')

    pipeline, light = make_pipeline(setup, detector=NeverInfer())
    # A failed frame following a successful frame must not preserve GREEN.
    assert pipeline.inspect(detections=setup[2]).passed
    image = generate_synthetic_aruco_board_image()
    expected = {'blur': 'sharpness', 'dark': 'brightness', 'missing_marker': 'markers', 'glare': 'glare'}
    if defect == 'blur':
        image = cv2.GaussianBlur(image, (13, 13), 2.5)
    elif defect == 'dark':
        image = (image.astype(float)*.5).astype(np.uint8)
    elif defect == 'missing_marker':
        image[20:110, 20:110] = 180
    else:
        image[220:320, 400:900] = 255
    result = pipeline.inspect(image)
    assert result.aborted and result.passed is False
    assert result.verdict is None and result.verdicts == {} and result.detections == []
    assert expected[defect] in result.quality.failures
    assert light.history[-2:] == ['YELLOW', 'RED']


def test_e2e_warning_render_and_snap_vectors(setup, monkeypatch):
    _, lut, detections = setup
    detections = detections + [make_detection(lut, 'W_LOOSE', 'wire', ['a50', 'a51'])]
    pipeline, _ = make_pipeline(setup)
    vectors = []
    original = cv2.arrowedLine

    def draw(image, start, end, *args, **kwargs):
        vectors.append((start, end))
        return original(image, start, end, *args, **kwargs)

    monkeypatch.setattr(cv2, 'arrowedLine', draw)
    result = pipeline.inspect(detections=detections)
    assert result.passed  # engine warnings do not fail the circuit
    assert 'DANGLING_WIRE' in {f.code for f in result.findings}
    assert np.any(np.all(result.annotated_image == AMBER, axis=-1))
    assert len(vectors) == 2*len(detections)
    assert any(start != end for start, end in vectors)
    # Rendering an older verdict remains correct after inspecting another frame.
    pipeline.inspect(detections=[])
    rerender = pipeline.render_annotated_image(result.rectified_image, result.verdict, lut)
    np.testing.assert_array_equal(rerender, result.annotated_image)


def test_e2e_runtime_error_sets_red(setup):
    class BrokenDetector:
        def detect(self, image):
            raise RuntimeError('backend failed')

    pipeline, light = make_pipeline(setup, detector=BrokenDetector())
    with pytest.raises(RuntimeError, match='backend failed'):
        pipeline.inspect()
    assert light.history == ['YELLOW', 'RED']


def test_e2e_injected_camera_and_empty_detections(setup):
    calls = []

    class Camera:
        def capture_burst(self):
            calls.append(True)
            return generate_synthetic_aruco_board_image()

    pipeline, light = make_pipeline(setup, camera=Camera())
    result = pipeline.inspect(detections=[])
    assert calls and not result.aborted
    assert not result.passed and light.state.upper() == 'RED'
    assert 'MISSING_PART' in {f.code for f in result.findings}


def test_e2e_invalid_detections_do_not_pass(setup):
    pipeline, light = make_pipeline(setup)
    with pytest.raises(ValueError, match='unique'):
        pipeline.inspect(detections=[setup[2][0], setup[2][0]])
    assert light.state.upper() == 'RED'
    invalid = deepcopy(setup[2])
    invalid[0]['keypoints'][0]['x'] = float('nan')
    with pytest.raises(ValueError, match='nonfinite'):
        pipeline.inspect(detections=invalid)
    assert light.state.upper() == 'RED'


def test_e2e_abstract_reference_with_explicit_placement(setup):
    golden, lut, detections = setup
    abstract = ReferenceCircuit(name='abstract', rails=golden.rails, parts=[
        {'id': 'R1', 'type': 'resistor', 'pins': {'0': 'VCC', '1': 'bridge'}},
        {'id': 'LED1', 'type': 'led', 'pins': {'anode': 'bridge', 'cathode': 'GND'}},
    ])
    pipeline = SherlockOhmsPipeline(abstract, placement_reference=golden, lut=lut,
                                    traffic_light=RecordingLight())
    result = pipeline.inspect(detections=detections)
    assert all(v.passed for v in result.verdicts.values())
