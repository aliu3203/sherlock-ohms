"""HTTP contracts exercised against real storage, worker, pipeline and renderer."""
from __future__ import annotations

import json
from pathlib import Path
import re
from types import SimpleNamespace
from urllib.parse import urljoin

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from fastapi.responses import FileResponse

from app.main import create_app
from app.storage import Storage
from engine import BoardSpec, HoleLUT
from pipeline import SherlockOhmsPipeline
from verify_simulation import (MockDetections, RecordingTrafficLight, detection,
                               golden_detections, golden_reference, run_verification)


@pytest.fixture
def context(tmp_path):
    storage = Storage(tmp_path / 'data')
    lut = HoleLUT.generate_synthetic(BoardSpec.full_830())
    detector = MockDetections(golden_detections(lut))
    light = RecordingTrafficLight()

    def factory(reference):
        return SherlockOhmsPipeline(reference, lut=lut, detector=detector, traffic_light=light)

    app = create_app(storage=storage, pipeline_factory=factory, traffic_light=light)
    with TestClient(app) as client:
        yield SimpleNamespace(client=client, app=app, storage=storage, detector=detector, light=light, lut=lut)


def inspect(context, reference_id='led_blink', mode='connectivity'):
    response = context.client.post('/api/inspect', json={'reference_id': reference_id, 'mode': mode})
    assert response.status_code == 200, response.text
    receipt = response.json()
    assert receipt['status'] == 'pending'
    assert receipt['reference_id'] == reference_id and receipt['mode'] == mode
    assert isinstance(receipt['inspection_id'], str) and receipt['inspection_id']
    context.app.state.worker.wait(receipt['inspection_id'], timeout=10)
    response = context.client.get(f"/api/inspections/{receipt['inspection_id']}")
    assert response.status_code == 200
    result = response.json()
    assert {'passed', 'light_color', 'findings', 'snapped_parts', 'actual_netlist', 'timestamp'} <= result.keys()
    return result


def sse_events(client, inspection_id):
    with client.stream('GET', f'/api/events/{inspection_id}') as response:
        assert response.status_code == 200
        assert response.headers['content-type'].startswith('text/event-stream')
        lines = list(response.iter_lines())
    events, current = [], {}
    for line in lines + ['']:
        if line.startswith('event: '):
            current['event'] = line[7:]
        elif line.startswith('data: '):
            current['data'] = json.loads(line[6:])
        elif not line and current:
            events.append(current)
            current = {}
    return events


def test_dashboard_root_and_static_dependencies(context):
    route = next(route for route in context.app.routes if getattr(route, 'path', None) == '/')
    assert isinstance(route.endpoint(), FileResponse)
    response = context.client.get('/')
    assert response.status_code == 200 and response.headers['content-type'].startswith('text/html')
    assert 'Sherlock Ohms' in response.text
    assert any(route.path == '/static' for route in context.app.routes)
    assets = re.findall(r'(?:href|src)="([^"]+)"', response.text)
    for asset in assets:
        if asset.startswith('/static/'):
            fetched = context.client.get(urljoin('/', asset))
            assert fetched.status_code == 200, asset
    for asset in ('app.js', 'demo.mjs', 'api.mjs', 'styles.css', 'board-base.svg'):
        fetched = context.client.get(f'/static/{asset}')
        assert fetched.status_code == 200 and fetched.content
        if asset.endswith(('.js', '.mjs')):
            assert 'javascript' in fetched.headers['content-type']
    assert context.client.get('/static/no-such-file.js').status_code == 404
    assert context.client.get('/api/health').json()['status'] == 'ok'


def test_seeded_references_are_correct_and_idempotent(context):
    response = context.client.get('/api/references')
    assert response.status_code == 200
    refs = {r['id']: r for r in response.json()['references']}
    assert set(refs) == {'led_blink', 'series_parallel_leds'}
    for key, name, count in [('led_blink', 'LED Blink', 2), ('series_parallel_leds', 'Series-Parallel LEDs', 4)]:
        assert refs[key]['name'] == name and refs[key]['is_seed'] is True
        assert len(refs[key]['data']['parts']) == count
        assert refs[key]['data']['rails'] == {'VCC': 'top_plus', 'GND': 'top_minus'}
    restarted = Storage(context.storage.root)
    assert restarted.list_references() == context.storage.list_references()


def test_post_inspect_get_full_result_and_history_contract(context):
    result = inspect(context)
    assert result['status'] == 'done' and result['passed'] is True
    assert result['light_color'] == 'GREEN' and result['findings'] == []
    assert result['quality']['passed'] and len(result['snapped_parts']) == 4
    assert context.light.transitions == ['YELLOW', 'GREEN']
    history = context.client.get('/api/inspections').json()['inspections']
    assert len(history) == 1 and history[0]['inspection_id'] == result['inspection_id']
    assert 'snapped_parts' not in history[0]
    assert context.client.get('/api/inspections', params={'limit': 0}).json()['inspections'] == []


@pytest.mark.parametrize('fault,code', [('short', 'POWER_SHORT'), ('open', 'OPEN_CONNECTION')])
def test_actual_faults_survive_http_serialization(context, fault, code):
    if fault == 'short':
        context.detector.values.append(detection(context.lut, 'W_SHORT', 'wire', 'top_plus_40', 'top_minus_40'))
    else:
        context.detector.values[2] = detection(context.lut, 'LED1', 'led', 'd10', 'd16')
    result = inspect(context)
    assert result['passed'] is False and result['light_color'] == 'RED'
    assert code in {f['code'] for f in result['findings']}
    assert context.light.transitions == ['YELLOW', 'RED']


def test_upload_and_reuse_reference_in_both_modes(context):
    definition = golden_reference().model_dump()
    response = context.client.post('/api/references', json=definition)
    assert response.status_code == 201
    ref = response.json()
    assert ref['data'] == definition and not ref['is_seed']
    assert ref in context.client.get('/api/references').json()['references']
    for mode in ('placement', 'connectivity'):
        assert inspect(context, ref['id'], mode)['passed']


def test_save_inspection_reference_preserves_wire_topology_and_rail_bindings(context):
    # Use bottom rails to detect accidental hard-coded top-rail bindings.
    definition = golden_reference().model_dump()
    definition['rails'] = {'VCC': 'bottom_plus', 'GND': 'bottom_minus'}
    for part in definition['parts']:
        part['pins'] = {pin: hole.replace('top_', 'bottom_') for pin, hole in part['pins'].items()}
    context.detector.values = [detection(context.lut, p['id'], p['type'], *p['pins'].values()) for p in definition['parts']]
    source = context.client.post('/api/references', json=definition).json()
    result = inspect(context, source['id'])
    assert result['passed'] is True
    response = context.client.post(f"/api/references/from-inspection/{result['inspection_id']}", json={'name': 'Captured golden board'})
    assert response.status_code == 201
    saved = response.json()
    assert saved['data']['name'] == 'Captured golden board'
    assert saved['data']['parts'] == definition['parts']
    assert saved['data']['rails'] == definition['rails']
    assert inspect(context, saved['id'])['passed'] is True


def test_sse_contract_streams_all_named_stages_in_order(context):
    # Connect immediately so this covers a live job as well as late replay.
    receipt = context.client.post('/api/inspect', json={'reference_id': 'led_blink'}).json()
    events = sse_events(context.client, receipt['inspection_id'])
    assert [e['event'] for e in events] == ['capturing', 'registering', 'detecting', 'checking', 'done']
    assert all(e['data']['inspection_id'] == receipt['inspection_id'] for e in events)
    assert all(e['data']['stage'] == e['event'] for e in events)
    assert events[-1]['data']['passed'] is True and events[-1]['data']['light_color'] == 'GREEN'
    context.app.state.worker.wait(receipt['inspection_id'], timeout=10)
    assert sse_events(context.client, receipt['inspection_id']) == events


def test_sse_persisted_terminal_record_survives_broker_restart(context):
    result = inspect(context)
    restarted_app = create_app(storage=context.storage)
    with TestClient(restarted_app) as client:
        events = sse_events(client, result['inspection_id'])
    assert [e['event'] for e in events] == ['done']
    assert events[0]['data']['passed'] is True


def test_images_calibration_and_preview_contracts(context):
    result = inspect(context)
    for layer in ('raw', 'rectified', 'annotated'):
        response = context.client.get(f"/api/inspections/{result['inspection_id']}/image", params={'layer': layer})
        assert response.status_code == 200 and response.headers['content-type'] == 'image/png'
        decoded = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
        assert decoded is not None
        if layer != 'raw':
            assert decoded.shape == (520, 1400, 3)
    assert context.client.get('/api/calibration').status_code == 404
    response = context.client.post('/api/calibrate')
    assert response.status_code == 200
    calibration = response.json()
    assert calibration['data']['has_homography'] is True
    assert set(calibration['data']['marker_centers']) == {'0', '1', '2', '3'}
    assert calibration['data']['quality']['passed'] is True
    assert context.client.get('/api/calibration').json() == calibration
    preview = context.client.get('/api/preview')
    assert preview.status_code == 200 and preview.headers['content-type'] == 'image/png'
    assert cv2.imdecode(np.frombuffer(preview.content, np.uint8), 1).shape[1] == 640


def test_quality_abort_contract_skips_detector(context):
    from vision_ml.registration import generate_synthetic_aruco_board_image
    class DarkCamera:
        def capture_burst(self):
            return (generate_synthetic_aruco_board_image()*.5).astype(np.uint8)
    original = context.app.state.pipeline_factory
    def factory(reference):
        pipeline = original(reference)
        pipeline.camera = DarkCamera()
        return pipeline
    context.app.state.pipeline_factory = factory
    result = inspect(context)
    assert result['status'] == 'aborted' and result['passed'] is False
    assert result['light_color'] == 'RED' and result['findings']
    assert 'brightness' in result['quality']['failures']
    assert context.detector.calls == 0
    events = sse_events(context.client, result['inspection_id'])
    assert [e['event'] for e in events] == ['capturing', 'registering', 'done']
    assert events[-1]['data']['status'] == 'aborted'


def test_backend_error_contract_has_no_false_pass(context):
    def fail(_image):
        raise RuntimeError('test detector unavailable')
    context.detector.detect = fail
    result = inspect(context)
    assert result['status'] == 'error' and result['passed'] is False
    assert result['light_color'] == 'RED' and result['findings'] == []
    assert result['error'] == 'test detector unavailable'
    events = sse_events(context.client, result['inspection_id'])
    assert events[-1]['event'] == 'error'
    assert context.light.transitions == ['YELLOW', 'RED']


@pytest.mark.parametrize('method,path,payload,status', [
    ('post', '/api/inspect', {'reference_id': 'missing'}, 404),
    ('post', '/api/inspect', {'reference_id': 'led_blink', 'mode': 'bad'}, 422),
    ('post', '/api/references', {'name': 'missing fields'}, 400),
    ('get', '/api/inspections/missing', None, 404),
    ('get', '/api/inspections/missing/image', None, 404),
    ('get', '/api/events/missing', None, 404),
    ('post', '/api/references/from-inspection/missing', {'name': 'capture'}, 404),
])
def test_api_error_status_contracts(context, method, path, payload, status):
    response = context.client.request(method, path, **({'json': payload} if payload is not None else {}))
    assert response.status_code == status
    assert 'detail' in response.json()


def test_incomplete_inspection_cannot_be_saved_and_missing_image_is_404(context):
    context.storage.create_inspection('pending', reference_id='led_blink', mode='connectivity')
    response = context.client.post('/api/references/from-inspection/pending', json={'name': 'not ready'})
    assert response.status_code == 400
    assert context.client.get('/api/inspections/pending/image').status_code == 404
    assert context.client.get('/api/inspections/pending/image?layer=invalid').status_code == 400
    pending = context.client.get('/api/inspections/pending').json()
    assert pending['passed'] is None and pending['findings'] == []


def test_verification_script_generates_decodable_debug_artifacts(tmp_path):
    report = run_verification(tmp_path)
    assert report['checks_passed'] == 9 and report['checks_failed'] == 0
    assert cv2.imread(report['debug_image']).shape == (520, 1400, 3)
    assert len(list((tmp_path / 'simulation_artifacts').glob('*.png'))) == 8
    assert json.loads((tmp_path / 'simulation_report.json').read_text()) == report
