"""Offline geometry/provenance tests, not a claim of physical alignment."""
import json
import sys
import types

import cv2
import numpy as np
import pytest

from engine.board import BoardSpec, hole_to_node
from vision_ml.calibrate_cli import create_hole_lut, save_calibration, render_calibration_overlay
from vision_ml.capture_dataset import CaptureSession, main
from vision_ml.pi_hardware import BreadboardCamera
from vision_ml.registration import BreadboardRegistration, generate_synthetic_aruco_board_image

POINTS = {'a1': (70, 100), 'a63': (1330, 100), 'j1': (70, 420), 'j63': (1330, 420)}


def geometry():
    # Irregular spacing and a gap; numeric IDs reflect engine electrical groups.
    return {'split_rails': True, 'rails': {
        'top_plus': {'holes': {'top_plus_1': {'x': 42, 'y': 31},
                              'top_plus_2': {'x': 65, 'y': 32},
                              'top_plus_32': {'x': 800, 'y': 34}},
                     'segments': [['top_plus_1', 'top_plus_2'], ['top_plus_32']]}}}


@pytest.fixture
def calibrated(tmp_path):
    raw = generate_synthetic_aruco_board_image()
    rectified = BreadboardRegistration().register(raw).rectified
    save_calibration(rectified, POINTS, tmp_path / 'cal', geometry=geometry(), setup_id='bench-v1')
    return raw, tmp_path / 'cal' / 'calibration.json'


def test_measured_rails_and_legacy_geometry(calibrated):
    raw, path = calibrated
    lut = json.loads(path.with_name('hole_lut.json').read_text())
    assert len(lut) == 633
    assert lut['top_plus_32'] == {'x': 800., 'y': 34.}
    assert lut['top_plus_2']['x'] != lut['a2']['x']
    board = BoardSpec.full_830(split_rails=True)
    assert hole_to_node('top_plus_1', board) == hole_to_node('top_plus_2', board)
    assert hole_to_node('top_plus_32', board) != hole_to_node('top_plus_2', board)
    assert len(create_hole_lut(POINTS)) == 630
    image = BreadboardRegistration().register(raw).rectified
    original = image.copy()
    overlay = render_calibration_overlay(image, lut)
    assert overlay.shape == (520, 1400, 3)
    assert np.array_equal(image, original)
    assert tuple(overlay[34, 800]) == (0, 190, 255)
    assert path.with_name('calibration_overlay.png').is_file()


@pytest.mark.parametrize('mutation', [
    lambda g: g['rails']['top_plus']['holes']['top_plus_1'].update(x=float('nan')),
    lambda g: g['rails']['top_plus']['holes']['top_plus_1'].update(y=520),
    lambda g: g['rails']['top_plus'].update(segments=[['top_plus_1', 'top_plus_2', 'top_plus_32']]),
    lambda g: g['rails']['top_plus'].update(segments=[['top_plus_1'], ['top_plus_1', 'top_plus_32']]),
    lambda g: g.update(terminal_column_positions_mm=[0]*63),
    lambda g: g.update(split_rails='false'),
])
def test_invalid_geometry(mutation):
    g = geometry()
    mutation(g)
    with pytest.raises(ValueError):
        create_hole_lut(POINTS, g)


def test_invalid_references_and_markers():
    with pytest.raises(ValueError):
        create_hole_lut({name: [10, 10] for name in POINTS})
    with pytest.raises(ValueError):
        BreadboardRegistration(marker_positions_mm={0: [0, 0]})
    with pytest.raises(ValueError):
        BreadboardRegistration(brightness_baseline=float('nan'))


def provenance(index):
    return {'source': 'local', 'timestamp': index+1, 'local_path': f'/empty/frame{index}.png'}


def test_baseline_explicit_quality_identity_and_immutable(calibrated, tmp_path):
    raw, path = calibrated
    session = CaptureSession(path, tmp_path/'capture', setup_id='bench-v1', source='local')
    populated = session.save_frame(raw, provenance(99))
    assert populated['quality']['passed']
    assert not (session.root/'empty_baseline.png').exists()
    with pytest.raises(ValueError, match='three'):
        session.create_empty_baseline([populated['id']])
    records = [session.save_frame(raw, provenance(i), empty_board=True) for i in range(3)]
    with pytest.raises(ValueError, match='EMPTY'):
        session.create_empty_baseline([populated['id'], records[0]['id'], records[1]['id']])
    output = session.create_empty_baseline([r['id'] for r in records])
    assert output.exists()
    expected = cv2.imread(str(session.root/records[0]['id']/'rectified.png'))
    assert np.array_equal(session.load_baseline(), expected)
    session.save_frame(raw, provenance(100))
    assert np.array_equal(session.load_baseline(), expected)
    with pytest.raises(FileExistsError):
        session.create_empty_baseline([r['id'] for r in records])
    with pytest.raises(ValueError, match='setup'):
        CaptureSession(path, session.root, setup_id='other', source='local')
    with pytest.raises(ValueError, match='different'):
        CaptureSession(path, session.root, setup_id='bench-v1', source='pi')
    lut_path = path.with_name('hole_lut.json')
    lut = json.loads(lut_path.read_text())
    lut['a1']['x'] += 1
    lut_path.write_text(json.dumps(lut))
    with pytest.raises(ValueError, match='fingerprint'):
        CaptureSession(path, tmp_path/'other', setup_id='bench-v1', source='local')


def test_rejected_quality_and_replayed_sources(calibrated, tmp_path):
    raw, path = calibrated
    session = CaptureSession(path, tmp_path/'capture', setup_id='bench-v1', source='local')
    dark = session.save_frame((raw*.4).astype(np.uint8), provenance(0), empty_board=True)
    assert not dark['quality']['passed']
    missing = session.save_frame(np.zeros_like(raw), provenance(1), empty_board=True)
    assert missing['quality']['brightness'] is None
    assert (session.root/missing['id']/'raw.png').exists()
    records = [session.save_frame(raw, provenance(2), empty_board=True) for _ in range(3)]
    with pytest.raises(ValueError, match='separately'):
        session.create_empty_baseline([r['id'] for r in records])
    with pytest.raises(ValueError, match='quality'):
        session.create_empty_baseline([dark['id'], records[0]['id'], records[1]['id']])
    assert not (session.root/'empty_baseline.png').exists()


def test_explicit_camera_failure_and_visible_fallback(monkeypatch, tmp_path):
    def unavailable():
        raise OSError('No camera attached')
    monkeypatch.setitem(sys.modules, 'picamera2', types.SimpleNamespace(Picamera2=unavailable))
    image = np.full((30, 30, 3), 100, np.uint8)
    path = tmp_path/'local.png'
    cv2.imwrite(str(path), image)
    with pytest.raises(RuntimeError, match='live'):
        BreadboardCamera([path], require_live=True, warmup_seconds=0)
    with BreadboardCamera([path], warmup_seconds=0) as camera:
        assert np.array_equal(camera.capture(), image)
        assert camera.last_capture['source'] == 'local'
        assert camera.last_capture['local_path'] == str(path)
        assert camera.last_capture['fallback_reason'] == 'No camera attached'


def test_live_capture_failure_never_uses_recording(monkeypatch, tmp_path):
    class Camera:
        camera_controls = {}
        def create_still_configuration(self, **kwargs): return {}
        def configure(self, config): pass
        def start(self): pass
        def capture_metadata(self): return {}
        def set_controls(self, controls): pass
        def stop(self): pass
        def close(self): pass
        def capture_array(self, name):
            self.count = getattr(self, 'count', 0)+1
            if self.count > 3:
                raise OSError('Disconnected')
            return np.zeros((20, 20, 3), np.uint8)
    monkeypatch.setitem(sys.modules, 'picamera2', types.SimpleNamespace(Picamera2=Camera))
    with BreadboardCamera([tmp_path/'recorded.png'], require_live=True, warmup_seconds=0) as camera:
        assert camera.source == 'pi'
        with pytest.raises(RuntimeError, match='live'):
            camera.capture_frame()
        assert camera.index == 0
        with pytest.raises(RuntimeError):
            camera.capture_frame()


def test_local_cli_empty_baseline(calibrated, tmp_path):
    raw, path = calibrated
    images = []
    for i in range(3):
        image_path = tmp_path/f'empty{i}.png'
        cv2.imwrite(str(image_path), raw)
        images.append(str(image_path))
    args = ['baseline', '--calibration', str(path), '--setup-id', 'bench-v1',
            '--output-dir', str(tmp_path/'cli'), '--source', 'local',
            '--count', '3', '--images', *images]
    with pytest.raises(SystemExit):
        main(args)
    assert main(args+['--confirm-empty-board']) == 0
    meta = json.loads((tmp_path/'cli'/'empty_baseline.json').read_text())
    assert len(meta['frames']) == 3
    assert meta['source'] == 'local'


def test_median_pixels_and_baseline_integrity(calibrated, tmp_path):
    raw, path = calibrated
    session = CaptureSession(path, tmp_path/'median', setup_id='bench-v1', source='local')
    records = []
    rectified = []
    for i, shift in enumerate((0, 3, 8)):
        frame = np.clip(raw.astype(np.int16)+shift, 0, 255).astype(np.uint8)
        record = session.save_frame(frame, provenance(i), empty_board=True)
        records.append(record['id'])
        rectified.append(cv2.imread(str(session.root/record['id']/'rectified.png')))
    session.create_empty_baseline(records)
    expected = np.median(np.stack(rectified), axis=0).astype(np.uint8)
    assert np.array_equal(session.load_baseline(), expected)
    assert not np.array_equal(expected, rectified[0])
    cv2.imwrite(str(session.root/'empty_baseline.png'), np.zeros_like(expected))
    with pytest.raises(ValueError, match='mismatch'):
        session.load_baseline()


def test_changed_calibration_cannot_reuse_session(calibrated, tmp_path):
    raw, path = calibrated
    CaptureSession(path, tmp_path/'session', setup_id='bench-v1', source='local')
    rectified = BreadboardRegistration().register(raw).rectified
    points = dict(POINTS, a1=(71, 100))
    save_calibration(rectified, points, path.parent, setup_id='bench-v1')
    with pytest.raises(ValueError, match='different'):
        CaptureSession(path, tmp_path/'session', setup_id='bench-v1', source='local')


def test_all_four_continuous_rails_with_optical_gaps():
    rails = {}
    for prefix, y in zip(('top_plus', 'top_minus', 'bottom_plus', 'bottom_minus'),
                         (25, 50, 460, 490)):
        holes = {f'{prefix}_{i}': {'x': x, 'y': y}
                 for i, x in ((1, 40), (2, 61), (3, 140), (4, 1200))}
        rails[prefix] = {'holes': holes, 'segments': [list(holes)]}
    lut = create_hole_lut(POINTS, {'split_rails': False, 'rails': rails})
    assert len(lut) == 646
    for prefix in rails:
        assert lut[f'{prefix}_4']['x'] == 1200
        assert hole_to_node(f'{prefix}_1', BoardSpec.full_830()) == hole_to_node(
            f'{prefix}_4', BoardSpec.full_830())


def test_invalid_rail_id_and_empty_setup(calibrated, tmp_path):
    g = geometry()
    g['rails']['top_plus']['holes']['top_plus_64'] = {'x': 1200, 'y': 30}
    with pytest.raises(ValueError, match='ID'):
        create_hole_lut(POINTS, g)
    raw, _ = calibrated
    with pytest.raises(ValueError, match='setup_id'):
        save_calibration(BreadboardRegistration().register(raw).rectified, POINTS,
                         tmp_path/'invalid', setup_id='')
