"""Safety and equivalence regressions; synthetic images are not model accuracy tests."""
import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from engine.live_verifier import LiveVerifier
from vision_ml.hole_grid import HoleGrid
from vision_ml.live_inspector import LiveInspector, calibrate


@pytest.fixture
def grid():
    return HoleGrid.from_anchors((800, 400), 30,
                                {'a1': [70, 80], 'a30': [650, 80], 'j30': [650, 300], 'j1': [70, 300]})


def part(pid, kind, h0, h1):
    return {'id': pid, 'type': kind, 'pins': {'0': h0, '1': h1}}


def detection(grid, pid, kind, h0, h1):
    return {'id': pid, 'type': kind, 'conf': .95, 'bbox': [0, 0, 400, 400],
            'keypoints': [{'x': grid.holes[h][0], 'y': grid.holes[h][1], 'conf': .95} for h in (h0, h1)]}


def settled(verifier, detections):
    for _ in range(3):
        result = verifier.inspect(detections)
    return result


def test_grid_requires_explicit_rails_and_honors_center_gap(grid):
    assert grid.holes['f1'][1] - grid.holes['e1'][1] == pytest.approx(60)
    assert grid.node('a1') == grid.node('e1') != grid.node('f1')
    assert grid.associate(*grid.holes['a1']).hole == 'a1'
    x, y = grid.holes['a1']
    assert grid.associate(x + 10, y).hole is None
    assert grid.associate(x, y, confidence=.5).hole is None
    with pytest.raises(ValueError):
        HoleGrid({**grid.holes, 'p1': (25, 25)}, image_size=grid.image_size, columns=30)


def test_explicit_split_rails(grid):
    measured = HoleGrid({**grid.holes, 'p1': (70, 30), 'p2': (90, 30), 'p3': (110, 30)},
                        image_size=grid.image_size, columns=30,
                        rail_segments={'positive_left': ['p1', 'p2'], 'positive_right': ['p3']})
    assert measured.node('p1') == measured.node('p2') != measured.node('p3')


def test_refiner_cannot_move_to_a_different_hole_and_can_defer_to_baseline(grid):
    x,y=grid.holes['a1']
    point={'x':x+20,'y':y,'conf':.95,'coarse_x':x,'coarse_y':y,'coarse_conf':.95}
    assert grid.associate_keypoint(point).reason=='refinement_disagrees_with_coarse_hole'
    point.update(x=x,y=y,conf=.59)
    assert grid.associate_keypoint(point).hole=='a1' # uninformative refinement, baseline already passes
    point.update(coarse_x=x+10,conf=.95)
    assert grid.associate_keypoint(point).hole is None # cannot resolve a coarse boundary ambiguity
    point.update(coarse_x=x+6)
    assert grid.associate_keypoint(point).hole=='a1' # precision improved within the same unambiguous cell


def test_hole_observation_refinement_and_occlusion(grid):
    image = np.full((400, 800, 3), 180, np.uint8)
    for x, y in grid.xy:
        cv2.circle(image, (round(x + 1), round(y + 1)), 3, (30, 30, 30), -1)
    assert grid.observe(image)['passed']
    refined = grid.refined(image)
    assert np.median(np.linalg.norm(refined.xy - (grid.xy + 1), axis=1)) < .1
    image[180:] = 180
    assert not grid.observe(image)['passed']


def test_wire_only_wrong_location_cannot_pass_and_same_strip_can(grid):
    verifier = LiveVerifier(grid, {'parts': [part('W1', 'wire', 'a1', 'a5')]})
    assert settled(verifier, [detection(grid, 'D1', 'wire', 'e1', 'c5')])['passed']
    wrong = settled(verifier, [detection(grid, 'D1', 'wire', 'a2', 'a6')])
    assert wrong['status'] == 'INCORRECT'


def test_equivalent_wire_route_can_have_more_wires(grid):
    verifier = LiveVerifier(grid, {'parts': [part('W1', 'wire', 'a1', 'a5')]})
    routed = [detection(grid, 'D1', 'wire', 'c1', 'c8'), detection(grid, 'D2', 'wire', 'b8', 'e5')]
    assert settled(verifier, routed)['passed']


def test_resistor_topology_accepts_relocation_and_rejects_changed_graph(grid):
    ref = {'parts': [part('R1', 'resistor', 'a1', 'a5'), part('R2', 'resistor', 'b5', 'a9')]}
    verifier = LiveVerifier(grid, ref)
    equivalent = [detection(grid, 'D1', 'resistor', 'f12', 'f16'), detection(grid, 'D2', 'resistor', 'h16', 'j20')]
    assert settled(verifier, equivalent)['passed']
    equivalent[1] = detection(grid, 'D2', 'resistor', 'h17', 'j20')
    assert settled(verifier, equivalent)['status'] == 'INCORRECT'


def test_external_power_terminals_are_not_permutable(grid):
    ref = {'terminals': {'VCC': 'a1', 'GND': 'a30'},
           'parts': [part('R1', 'resistor', 'b1', 'b30')]}
    verifier = LiveVerifier(grid, ref)
    assert settled(verifier, [detection(grid, 'D1', 'resistor', 'e1', 'd30')])['passed']
    assert settled(verifier, [detection(grid, 'D1', 'resistor', 'e2', 'd30')])['status'] == 'INCORRECT'
    short = settled(verifier, [detection(grid, 'W1', 'wire', 'a1', 'a30')])
    assert short['status'] == 'INCORRECT'
    assert short['findings'][0]['code'] == 'POWER_SHORT'


def test_uncertainty_resets_previous_green_and_never_silently_drops_proposals(grid):
    verifier = LiveVerifier(grid, {'parts': [part('W1', 'wire', 'a1', 'a5')]})
    ds = [detection(grid, 'D1', 'wire', 'e1', 'c5')]
    assert settled(verifier, ds)['passed']
    assert verifier.inspect(ds, quality_ok=False)['status'] == 'UNCERTAIN'
    assert verifier.inspect(ds)['status'] == 'UNCERTAIN'
    assert settled(verifier, ds)['passed']
    ds.append(detection(grid, 'D2', 'wire', 'a10', 'a14'))
    ds[-1]['conf'] = .2
    assert verifier.inspect(ds)['status'] == 'UNCERTAIN'
    assert verifier.inspect([])['passed'] is False


def test_jitter_duplicate_holes_and_missing_components(grid):
    verifier = LiveVerifier(grid, {'parts': [part('R1', 'resistor', 'a1', 'a5')]})
    correct = [detection(grid, 'D1', 'resistor', 'e1', 'c5')]
    wrong = [detection(grid, 'D1', 'resistor', 'a1', 'b1')]
    for i in range(8):
        assert verifier.inspect(correct if i % 2 else wrong)['status'] == 'UNCERTAIN'
    duplicate = correct + [detection(grid, 'D2', 'wire', 'e1', 'c8')]
    assert verifier.inspect(duplicate)['status'] == 'UNCERTAIN'
    assert settled(verifier, [detection(grid, 'D1', 'wire', 'a1', 'a5')])['status'] == 'INCOMPLETE'


def test_invalid_reference_and_nan_predictions_fail_closed(grid):
    with pytest.raises(ValueError):
        LiveVerifier(grid, {'parts': []})
    with pytest.raises(ValueError):
        LiveVerifier(grid, {'parts': [part('L1', 'led', 'a1', 'a5')]})
    verifier = LiveVerifier(grid, {'parts': [part('W1', 'wire', 'a1', 'a5')]})
    ds = [detection(grid, 'D1', 'wire', 'a1', 'a5')]
    ds[0]['keypoints'][0]['x'] = float('nan')
    result = verifier.inspect(ds)
    assert result['status'] == 'UNCERTAIN'
    json.dumps(result, allow_nan=False)


def test_refiner_pixel_geometry_and_uncertain_heatmaps():
    import torch
    from vision_ml.endpoint_refiner import EndpointRefiner, patch_input
    image = np.zeros((120,120,3),np.uint8)
    sample,origin = patch_input(image,(40,40),(80,40))
    assert sample.shape == (6,64,64) and origin.tolist() == [8.5,8.5]
    class Heatmap:
        def __init__(self,diffuse=False,large_shift=False):
            yy,xx=torch.meshgrid(torch.arange(64),torch.arange(64),indexing='ij')
            self.logits=(-(xx-(60 if large_shift else 34.5))**2-(yy-28.5)**2)/4.5
            if diffuse:self.logits[:]=0
        def __call__(self,x):return self.logits.flatten()[None].repeat(len(x),1)
    refiner=EndpointRefiner.__new__(EndpointRefiner)
    refiner.device='cpu';refiner.model=Heatmap()
    detections=[{'id':'W1','type':'wire','conf':.9,'keypoints':[
        {'x':40.,'y':40.,'conf':.99},{'x':80.,'y':40.,'conf':.99}]}]
    point=refiner.refine(image,detections)[0]['keypoints'][0]
    assert point['x']==pytest.approx(43,abs=.12) and point['y']==pytest.approx(37,abs=.12)
    assert detections[0]['keypoints'][0]['x']==40 # caller's raw predictions remain available
    for heat in (Heatmap(diffuse=True),Heatmap(large_shift=True)):
        refiner.model=heat
        point=refiner.refine(image,detections)[0]['keypoints'][0]
        assert point['x']==40 and point['conf']<.6


def test_marker_calibration_to_live_verdict_and_tamper_check(grid, tmp_path, monkeypatch):
    # This exercises real ArUco detection/rectification and hole fitting, but uses
    # known predictions. It is intentionally NOT a neural accuracy measurement.
    image = np.full((528, 928, 3), 180, np.uint8)
    for x, y in grid.xy:
        cv2.circle(image, (round(x + 64), round(y + 64)), 3, (30, 30, 30), -1)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    for i, (x, y) in enumerate([(64, 64), (863, 64), (863, 463), (64, 463)]):
        marker = cv2.aruco.generateImageMarker(dictionary, i, 64)
        image[y-32:y+32, x-32:x+32] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    path = tmp_path / 'empty.png'
    cv2.imwrite(str(path), image)
    anchors = tmp_path / 'anchors.json'
    anchors.write_text(json.dumps({'anchors': {k: grid.holes[k] for k in ('a1', 'a30', 'j30', 'j1')}}))
    out = tmp_path / 'setup'
    calibrate(SimpleNamespace(image=path, output=out, registration=None, size=(800, 400), columns=30, anchors=anchors))
    class Detector:
        def __init__(self, *a, **kw):
            pass
        def detect(self, frame):
            return [detection(grid, 'D1', 'wire', 'a1', 'a5')]
    monkeypatch.setattr('vision_ml.live_inspector.EndpointDetector', Detector)
    inspector = LiveInspector(setup=out / 'setup.json', weights='test-injected',
                              reference={'parts': [part('W1', 'wire', 'a1', 'a5')]})
    for _ in range(3):
        verdict, overlay = inspector.inspect(image)
    assert verdict['passed'] and overlay.shape == (400, 800, 3)
    # Camera motion should be absorbed by marker registration, with the original
    # hole map retained. Do not refit holes to each new set of predictions.
    motion = np.array([[.98,.035,7],[-.015,1.01,10],[.000015,.00003,1]], dtype=float)
    moved_camera = cv2.warpPerspective(image, motion, image.shape[1::-1])
    moved_result, _ = inspector.inspect(moved_camera)
    assert moved_result['passed']
    # Moving the markers relative to the board invalidates that calibration.
    for dx, dy in ((0, 8), (0, 20), (20, 0)):
        moved_markers = image.copy()
        for i, (x, y) in enumerate([(64, 64), (863, 64), (863, 463), (64, 463)]):
            moved_markers[y-32:y+32+dy, x-32:x+32+dx] = 180
            marker = cv2.aruco.generateImageMarker(dictionary, i, 64)
            moved_markers[y-32+dy:y+32+dy, x-32+dx:x+32+dx] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
        assert inspector.inspect(moved_markers)[0]['status'] == 'UNCERTAIN'
    blank = np.full_like(image, 180)
    assert inspector.inspect(blank)[0]['status'] == 'UNCERTAIN'
    with monkeypatch.context() as timed:
        times = iter((0., 3.))
        timed.setattr('vision_ml.live_inspector.time.perf_counter', lambda: next(times))
        stale, _ = inspector.inspect(image)
        assert stale['status'] == 'UNCERTAIN' and stale['elapsed_ms'] == 3000
    setup = json.loads((out / 'setup.json').read_text())
    setup['grid']['holes']['a1'][0] += 20
    (out / 'setup.json').write_text(json.dumps(setup))
    with pytest.raises(ValueError, match='changed'):
        LiveInspector(setup=out / 'setup.json', weights='test-injected', reference={'parts': []})
