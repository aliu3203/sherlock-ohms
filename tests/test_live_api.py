"""HTTP contracts and conservative live-state checks, not model accuracy tests."""
import base64
import json
from io import BytesIO
from pathlib import Path
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import create_app
from app.hole_api import decode_image
from app.storage import Storage
from vision_ml.hole_grid import HoleGrid


def encoded(image):
    ok, blob = cv2.imencode('.png', image)
    assert ok
    return base64.b64encode(blob).decode('ascii')


def board_image():
    """Real marker/quality/grid path with known geometry; no neural assertions."""
    grid = HoleGrid.from_anchors((1400, 520), 30,
                                {'a1': [70, 80], 'a30': [650, 80], 'j30': [650, 300], 'j1': [70, 300]})
    image = np.full((648, 1528, 3), 180, np.uint8)
    for x, y in grid.xy:
        cv2.circle(image, (round(x + 64), round(y + 64)), 3, (30, 30, 30), -1)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    for i, (x, y) in enumerate([(64, 64), (1463, 64), (1463, 583), (64, 583)]):
        marker = cv2.aruco.generateImageMarker(dictionary, i, 64)
        image[y-32:y+32, x-32:x+32] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    return image, {name: grid.holes[name] for name in ('a1', 'a30', 'j30', 'j1')}


@pytest.fixture
def client(tmp_path):
    app = create_app(storage=Storage(tmp_path / 'data'))
    with TestClient(app) as test_client:
        yield test_client


def reference(client, **updates):
    payload = {'name': 'Wire bridge', 'parts': [
        {'id': 'W1', 'type': 'wire', 'pins': {'0': 'a1', '1': 'a5'}}], 'terminals': {}}
    payload.update(updates)
    response = client.post('/api/live/references', json=payload)
    assert response.status_code == 201, response.text
    return response.json()


def setup(client):
    image, anchors = board_image()
    response = client.post('/api/live/setups', json={'name': 'Test board', 'columns': 30,
                                                  'anchors': anchors, 'image_base64': encoded(image)})
    assert response.status_code == 201, response.text
    return response.json(), image


def inject_predictions(client, monkeypatch, saved):
    backend = client.app.state.live_api
    raw = json.loads((backend.root / f"{saved['id']}.json").read_text())
    grid = HoleGrid(**raw['grid'])

    class Detector:
        def detect(self, image):
            return [{'id': 'D1', 'type': 'wire', 'conf': .99, 'bbox': [50, 50, 200, 110],
                     'keypoints': [{'x': grid.holes[h][0], 'y': grid.holes[h][1], 'conf': .99}
                                   for h in ('a1', 'a5')]}]

    class Refiner:
        def refine(self, image, detections):
            return detections

    backend.detector, backend.refiner = Detector(), Refiner()
    monkeypatch.setattr(backend, 'ensure_models', lambda: None)
    return backend


def session(client, saved, ref):
    response = client.post('/api/live/sessions', json={'setup_id': saved['id'], 'reference_id': ref['id']})
    assert response.status_code == 201, response.text
    return response.json()['session_id']


def frame(client, identity, image, number, mode='live'):
    distinct = image.copy()
    distinct[10, number % 50, 0] = number % 170  # emulate distinct captures, not accuracy evidence
    return client.post(f'/api/live/sessions/{identity}/frame', json={
        'image_base64': encoded(distinct), 'frame_id': number, 'mode': mode})


def test_status_lazy_and_old_api_is_available(client):
    status = client.get('/api/live/status').json()
    assert status['model']['loaded'] is False
    assert status['model']['mock'] is False
    assert status['model']['supported_types'] == ['wire', 'resistor']
    assert status['setups'] == status['references'] == []
    assert client.get('/api/health').status_code == 200
    assert client.get('/api/references').status_code == 200


def test_reference_validation_and_persistence(client):
    created = reference(client)
    assert client.get('/api/live/references').json()['references'][0]['id'] == created['id']
    with TestClient(create_app(storage=Storage(client.app.state.storage.root))) as reopened:
        assert reopened.get('/api/live/references').json()['references'][0] == created
    for bad in (
        {'parts': []}, {'name': ' '},
        {'parts': [{'id': 'L1', 'type': 'led', 'pins': {'0': 'a1', '1': 'a5'}}]},
        {'parts': [{'id': 'W1', 'type': 'wire', 'pins': {'A': 'a1', 'B': 'a5'}}]},
        {'parts': [{'id': 'W1', 'type': 'wire', 'pins': {'0': 'a1', '1': 'a64'}}]},
        {'parts': [{'id': 'W1', 'type': 'wire', 'pins': {'0': 'a1', '1': 'a1'}}]},
        {'terminals': {'VCC': 'top_plus'}},
    ):
        payload = {'name': 'Bad', 'parts': [{'id': 'W1', 'type': 'wire', 'pins': {'0': 'a1', '1': 'a5'}}]}
        payload.update(bad)
        assert client.post('/api/live/references', json=payload).status_code == 422


def test_image_validation_and_missing_weights_never_mock(client, monkeypatch, tmp_path):
    for invalid in ('not-base64', base64.b64encode(b'not an image').decode()):
        assert client.post('/api/live/preview', json={'image_base64': invalid}).status_code == 400
    assert client.post('/api/live/preview', json={'image_base64': encoded(np.zeros((8, 8, 3), np.uint8))}).status_code == 413
    monkeypatch.setattr(client.app.state.live_api, 'weights', tmp_path / 'missing.pt')
    response = client.post('/api/live/preview', json={'image_base64': encoded(np.full((64, 64, 3), 180, np.uint8))})
    assert response.status_code == 503 and 'missing' in response.json()['detail']
    assert client.get('/api/live/history').json()['inspections'] == []


def test_phone_exif_orientation_is_normalized_before_inspection(client, monkeypatch):
    # A landscape JPEG with EXIF=6 displays as a clockwise portrait image.
    pixels = np.zeros((40, 80, 3), np.uint8)
    pixels[:, :40] = (255, 0, 0)
    pixels[:, 40:] = (0, 0, 255)
    source = Image.fromarray(pixels)
    exif = Image.Exif()
    exif[274] = 6
    blob = BytesIO()
    source.save(blob, format='JPEG', quality=95, exif=exif)
    value = base64.b64encode(blob.getvalue()).decode('ascii')
    decoded = decode_image(value)
    assert decoded.shape == (80, 40, 3)
    assert decoded[10, 20, 2] > 230 and decoded[10, 20, 0] < 20
    assert decoded[70, 20, 0] > 230 and decoded[70, 20, 2] < 20
    # The public image endpoint accepts this common phone format too.
    backend = client.app.state.live_api
    monkeypatch.setattr(backend, 'ensure_models', lambda: None)
    backend.detector = SimpleNamespace(detect=lambda image: [])
    backend.refiner = SimpleNamespace(refine=lambda image, detections: detections)
    response = client.post('/api/live/preview', json={'image_base64': value})
    assert response.status_code == 200, response.text
    assert (response.json()['width'], response.json()['height']) == (40, 80)


def test_duplicate_and_reversed_markers_return_actionable_422(client):
    image, anchors = board_image()
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    duplicate = image.copy()
    marker = cv2.aruco.generateImageMarker(dictionary, 0, 64)
    duplicate[180:244, 1100:1164] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    reversed_markers = image.copy()
    for identity, (x, y) in ((3, (1463, 64)), (1, (64, 583))):
        marker = cv2.aruco.generateImageMarker(dictionary, identity, 64)
        reversed_markers[y-32:y+32, x-32:x+32] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    for bad_image in (duplicate, reversed_markers):
        value = encoded(bad_image)
        preview = client.post('/api/live/calibration/preview', json={'image_base64': value, 'columns': 30})
        assert preview.status_code == 422, preview.text
        assert 'registration rejected' in preview.json()['detail']
        setup = client.post('/api/live/setups', json={
            'name': 'Rejected', 'image_base64': value, 'columns': 30, 'anchors': anchors})
        assert setup.status_code == 422, setup.text
    assert client.get('/api/live/setups').json()['setups'] == []


def test_marker_preview_and_calibration_are_real_and_fingerprinted(client):
    image, anchors = board_image()
    response = client.post('/api/live/calibration/preview', json={'image_base64': encoded(image), 'columns': 30})
    assert response.status_code == 200, response.text
    assert response.json()['width'] == 1400
    assert response.json()['anchor_names'] == ['a1', 'a30', 'j30', 'j1']
    assert response.json()['quality']['passed']
    blank = encoded(np.full_like(image, 180))
    assert client.post('/api/live/calibration/preview', json={'image_base64': blank}).status_code == 422
    saved, _ = setup(client)
    assert saved['observation']['passed'] and saved['image_base64']
    assert client.get('/api/live/setups').json()['setups'][0]['id'] == saved['id']
    payload = {'name': 'Bad', 'columns': 30, 'image_base64': encoded(image), 'anchors': anchors}
    payload['anchors']['a1'] = [float('nan'), 80]
    # Exercise finite validation even for a nonstandard JSON NaN token.
    invalid = client.post('/api/live/setups', content=json.dumps(payload), headers={'Content-Type': 'application/json'})
    assert invalid.status_code == 422
    assert 'NaN' not in invalid.text


def test_uncalibrated_preview_returns_endpoints_but_never_a_pass(client, monkeypatch):
    saved, image = setup(client)
    inject_predictions(client, monkeypatch, saved)
    result = client.post('/api/live/preview', json={'image_base64': encoded(image)}).json()
    assert result['status'] == 'UNCALIBRATED' and not result['passed']
    assert result['parts'] == [] and len(result['detections']) == 1 and result['image_base64']
    history = client.get('/api/live/history').json()['inspections']
    assert len(history) == 1 and history[0]['mode'] == 'preview'
    detail = client.get('/api/live/inspections/' + result['id']).json()
    assert detail['image_base64'] and detail['id'] == result['id']


def test_live_sessions_settle_separately_and_still_never_turns_green(client, monkeypatch):
    saved, image = setup(client)
    backend = inject_predictions(client, monkeypatch, saved)
    ref = reference(client)
    first, second = session(client, saved, ref), session(client, saved, ref)
    for number in (1, 2):
        result = frame(client, first, image, number)
        assert result.status_code == 200, result.text
        assert result.json()['status'] == 'UNCERTAIN'
    assert frame(client, first, image, 3).json()['passed']
    assert frame(client, second, image, 1).json()['status'] == 'UNCERTAIN'
    for number in (4, 5, 6):
        result = frame(client, first, image, number, 'still').json()
        assert not result['passed'] and result['status'] == 'UNCERTAIN'
        assert result['candidate_status'] == 'CORRECT'
    assert frame(client, first, image, 7).json()['status'] == 'UNCERTAIN'
    history = client.get('/api/live/history').json()['inspections']
    assert len([h for h in history if h['mode'] == 'live']) == 2
    assert len([h for h in history if h['mode'] == 'still']) == 3
    assert client.delete('/api/live/sessions/' + first).json() == {'deleted': True}
    assert frame(client, first, image, 8).status_code == 404
    assert first not in backend.sessions


def test_still_preserves_tracking_failure_diagnostics_without_a_candidate(client, monkeypatch):
    saved, image = setup(client)
    inject_predictions(client, monkeypatch, saved)
    identity = session(client, saved, reference(client))
    result = frame(client, identity, np.full_like(image, 180), 1, 'still')
    assert result.status_code == 200, result.text
    result = result.json()
    assert result['status'] == 'UNCERTAIN' and result['light_color'] == 'AMBER' and not result['passed']
    assert result.get('candidate_status') is None
    assert 'markers failed' in result['message'] and 'markers' in result['message']
    assert 'start live capture' not in result['message']


def test_replay_errors_lost_tracking_and_gaps_reset_green(client, monkeypatch):
    saved, image = setup(client)
    backend = inject_predictions(client, monkeypatch, saved)
    identity = session(client, saved, reference(client))
    for number in (1, 2, 3):
        result = frame(client, identity, image, number)
    assert result.json()['passed']
    # The identical frame cannot be replayed to maintain a settled result.
    assert frame(client, identity, image, 3).status_code == 409
    assert frame(client, identity, image, 4).json()['status'] == 'UNCERTAIN'
    for number in (5, 6):
        result = frame(client, identity, image, number)
    assert result.json()['passed']
    backend.sessions[identity].last_at -= 3
    assert frame(client, identity, image, 7).json()['status'] == 'UNCERTAIN'
    assert frame(client, identity, image, 8).json()['status'] == 'UNCERTAIN'
    assert frame(client, identity, image, 9).json()['passed']
    bad = client.post(f'/api/live/sessions/{identity}/frame', json={'image_base64': 'bad', 'mode': 'live'})
    assert bad.status_code == 400
    assert frame(client, identity, image, 10).json()['status'] == 'UNCERTAIN'
    assert frame(client, identity, np.full_like(image, 180), 11).json()['status'] == 'UNCERTAIN'
    assert frame(client, identity, image, 12).json()['status'] == 'UNCERTAIN'
    # Pydantic errors reset state too.
    assert frame(client, identity, image, 13).json()['status'] == 'UNCERTAIN'
    assert frame(client, identity, image, 14).json()['passed']
    bad = client.post(f'/api/live/sessions/{identity}/frame', json={'image_base64': encoded(image), 'mode': 'bad'})
    assert bad.status_code == 422
    assert frame(client, identity, image, 15).json()['status'] == 'UNCERTAIN'


def test_reference_must_fit_actual_grid_and_unchanged_setup(client, monkeypatch):
    saved, _ = setup(client)
    backend = inject_predictions(client, monkeypatch, saved)
    outside = reference(client, parts=[{'id': 'W1', 'type': 'wire', 'pins': {'0': 'a1', '1': 'a63'}}])
    response = client.post('/api/live/sessions', json={'setup_id': saved['id'], 'reference_id': outside['id']})
    assert response.status_code == 422
    valid = reference(client)
    path = backend.root / f"{saved['id']}.json"
    raw = json.loads(path.read_text())
    raw['grid']['holes']['a1'][0] += 20
    path.write_text(json.dumps(raw))
    response = client.post('/api/live/sessions', json={'setup_id': saved['id'], 'reference_id': valid['id']})
    assert response.status_code == 422 and 'changed' in response.json()['detail']


def test_identical_frames_and_slow_requests_cannot_keep_green(client, monkeypatch):
    saved, image = setup(client)
    backend = inject_predictions(client, monkeypatch, saved)
    identity = session(client, saved, reference(client))
    same = encoded(image)
    for number in (1, 2, 3):
        result = client.post(f'/api/live/sessions/{identity}/frame', json={
            'image_base64': same, 'frame_id': number, 'mode': 'live'})
        assert result.json()['status'] == 'UNCERTAIN'
        assert result.json()['stable_frames'] == 1
    assert frame(client, identity, image, 4).json()['status'] == 'UNCERTAIN'
    assert frame(client, identity, image, 5).json()['passed']
    # Account for request queue/decode time as well as neural inference time.
    clock = [time.monotonic()]
    monkeypatch.setattr('app.hole_api.time', SimpleNamespace(
        monotonic=lambda: clock[0], perf_counter=time.perf_counter))
    original = backend.sessions[identity].inspector.inspect

    def delayed(frame):
        result = original(frame)
        clock[0] += 3
        return result

    backend.sessions[identity].inspector.inspect = delayed
    result = frame(client, identity, image, 6).json()
    assert result['status'] == 'UNCERTAIN' and not result['passed']
    assert result['elapsed_ms'] == 3000


def test_cors_requires_explicit_origin(tmp_path, monkeypatch):
    monkeypatch.setenv('SHERLOCK_ALLOWED_ORIGINS', 'https://example.base44.app')
    with TestClient(create_app(storage=Storage(tmp_path / 'cors'))) as client:
        allowed = client.options('/api/live/status', headers={
            'Origin': 'https://example.base44.app', 'Access-Control-Request-Method': 'GET'})
        assert allowed.headers['access-control-allow-origin'] == 'https://example.base44.app'
        denied = client.options('/api/live/status', headers={
            'Origin': 'https://unrelated.example', 'Access-Control-Request-Method': 'GET'})
        assert 'access-control-allow-origin' not in denied.headers
