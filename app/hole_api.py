"""Local HTTP bridge to the trained endpoint and measured-hole inspection path.

This API never substitutes synthetic captures or mock detections for a model.
References, calibrations, and bounded inspection history are stored separately
from the original dashboard's data. Models are shared; verifier state is not.
"""
from __future__ import annotations

import base64
import binascii
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import re
import sqlite3
import threading
import time
from typing import Any, Literal
import uuid

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from engine.live_verifier import LiveVerifier
from vision_ml.endpoint_detector import EndpointDetector
from vision_ml.hole_grid import HoleGrid
from vision_ml.live_inspector import LiveInspector, draw_overlay
from vision_ml.registration import BreadboardRegistration, content_fingerprint

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / 'evaluation' / 'hole_precision_20261017'
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_ENCODED = ((MAX_IMAGE_BYTES + 2) // 3) * 4 + 128
MAX_PIXELS = 16_000_000
HOLE = re.compile(r'[a-j](?:[1-9]|[1-5][0-9]|6[0-3])\Z')


def _now():
    return datetime.now(timezone.utc).isoformat()


def _safe(value):
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (str, int, bool, type(None))):
        return value
    return str(value)


class _SafeRoute(APIRoute):
    def get_route_handler(self):
        handler = super().get_route_handler()

        async def safe_handler(request: Request):
            try:
                return await handler(request)
            except RequestValidationError as exc:
                # A rejected frame must not leave a previous green settled.
                backend = getattr(request.app.state, 'live_api', None)
                if backend and request.path_params.get('session_id'):
                    backend.reset_session(request.path_params['session_id'])
                return JSONResponse(status_code=422, content={'detail': _safe(exc.errors())})
        return safe_handler


class StrictBody(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)


class ImageBody(StrictBody):
    image_base64: str = Field(min_length=1, max_length=MAX_ENCODED)


class CalibrationBody(ImageBody):
    columns: Literal[17, 30, 63] = 30


class SetupBody(CalibrationBody):
    name: str = Field(min_length=1, max_length=100)
    anchors: dict[str, tuple[float, float]]

    @model_validator(mode='after')
    def named_anchors(self):
        expected = {'a1', f'a{self.columns}', f'j{self.columns}', 'j1'}
        if set(self.anchors) != expected:
            raise ValueError(f'Provide exactly these rectified hole centers: {sorted(expected)}')
        return self


class ReferencePart(StrictBody):
    id: str = Field(min_length=1, max_length=80)
    type: Literal['wire', 'resistor']
    pins: dict[str, str]

    @field_validator('pins')
    @classmethod
    def pins_valid(cls, pins):
        if set(pins) != {'0', '1'} or any(not HOLE.fullmatch(h) for h in pins.values()):
            raise ValueError('Pins must be 0 and 1, with terminal holes a1 through j63')
        return pins


class ReferenceBody(StrictBody):
    name: str = Field(min_length=1, max_length=100)
    parts: list[ReferencePart] = Field(min_length=1, max_length=100)
    terminals: dict[str, str] = Field(default_factory=dict, max_length=20)

    @model_validator(mode='after')
    def circuit_valid(self):
        ids = [p.id for p in self.parts]
        occupied = [h for p in self.parts for h in p.pins.values()]
        if len(ids) != len(set(ids)) or len(occupied) != len(set(occupied)):
            raise ValueError('Part IDs and occupied pin holes must be unique')
        if any(not k.strip() or len(k) > 80 or not HOLE.fullmatch(h) for k, h in self.terminals.items()):
            raise ValueError('Named terminals must use terminal holes a1 through j63')
        if not self.name.strip():
            raise ValueError('Reference name must not be blank')
        return self


class SessionBody(StrictBody):
    setup_id: str = Field(min_length=1, max_length=80)
    reference_id: str = Field(min_length=1, max_length=80)


class FrameBody(ImageBody):
    mode: Literal['still', 'live'] = 'still'
    frame_id: int | None = Field(default=None, ge=0, strict=True)


def decode_image(value: str) -> np.ndarray:
    if value.startswith('data:'):
        header, separator, value = value.partition(',')
        if not separator or header not in ('data:image/jpeg;base64', 'data:image/png;base64', 'data:image/webp;base64'):
            raise HTTPException(400, 'Use a base64 JPEG, PNG, or WebP image')
    try:
        blob = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error):
        raise HTTPException(400, 'Invalid base64 image') from None
    if not blob or len(blob) > MAX_IMAGE_BYTES:
        raise HTTPException(413, 'Image must be between 1 byte and 8 MB')
    try:
        with Image.open(BytesIO(blob)) as source:
            width, height = source.size
            if source.format not in ('PNG', 'JPEG', 'WEBP'):
                raise HTTPException(400, 'Use a JPEG, PNG, or WebP image')
            if min(width, height) < 16 or max(width, height) > 8192 or width * height > MAX_PIXELS:
                raise HTTPException(413, 'Image dimensions must be at least 16px and at most 16 megapixels')
            # Phone JPEGs often store portrait orientation in EXIF. Normalize it
            # once so calibration clicks, preview images, and inference all use
            # the same pixel coordinate system, including mirrored EXIF modes.
            normalized = ImageOps.exif_transpose(source).convert('RGB')
            width, height = normalized.size
            if min(width, height) < 16 or max(width, height) > 8192 or width * height > MAX_PIXELS:
                raise HTTPException(413, 'Image dimensions must be at least 16px and at most 16 megapixels')
            return cv2.cvtColor(np.asarray(normalized), cv2.COLOR_RGB2BGR)
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError, cv2.error, Image.DecompressionBombError):
        raise HTTPException(400, 'Image could not be decoded') from None


def _jpeg(image):
    ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise HTTPException(500, 'Could not encode inspection image')
    return encoded.tobytes()


def _with_image(result, image):
    result = _safe(result)
    result.update(image_base64=base64.b64encode(_jpeg(image)).decode('ascii'),
                  width=int(image.shape[1]), height=int(image.shape[0]))
    return result


@dataclass
class Session:
    inspector: Any
    setup_id: str
    reference_id: str
    lock: threading.RLock = field(default_factory=threading.RLock)
    last_at: float = 0.
    last_frame_id: int = -1
    last_digest: str | None = None
    created_at: str = field(default_factory=_now)
    last_mode: str | None = None
    closed: bool = False


class LiveBackend:
    def __init__(self, root: Path):
        self.root = Path(root) / 'live'
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / 'live.sqlite3'
        self.lock = threading.RLock()
        self.inference_lock = threading.RLock()
        self.sessions: dict[str, Session] = {}
        self.detector = self.refiner = None
        self.weights = Path(os.environ.get('SHERLOCK_ENDPOINT_WEIGHTS', EXPERIMENT / 'runs/nano_endpoints/weights/best.pt'))
        override = os.environ.get('SHERLOCK_ENDPOINT_REFINER')
        self.refiner_weights = Path(override) if override else EXPERIMENT / 'refiner/best.pt'
        self.device = os.environ.get('SHERLOCK_ENDPOINT_DEVICE', 'cpu')
        if self.device not in ('cpu', 'mps'):
            raise ValueError('SHERLOCK_ENDPOINT_DEVICE must be cpu or mps')
        with self.connect() as conn:
            conn.executescript('''
                CREATE TABLE IF NOT EXISTS objects (
                  id TEXT PRIMARY KEY, kind TEXT NOT NULL, created_at TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS inspections (
                  id TEXT PRIMARY KEY, created_at TEXT NOT NULL, payload TEXT NOT NULL, jpeg BLOB NOT NULL);
            ''')

    def connect(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def objects(self, kind):
        with self.connect() as conn:
            rows = conn.execute('SELECT payload FROM objects WHERE kind=? ORDER BY created_at DESC', (kind,)).fetchall()
        return [json.loads(row['payload']) for row in rows]

    def object(self, kind, identity):
        with self.connect() as conn:
            row = conn.execute('SELECT payload FROM objects WHERE kind=? AND id=?', (kind, identity)).fetchone()
        if row is None:
            raise HTTPException(404, f'{kind.capitalize()} not found')
        return json.loads(row['payload'])

    def save_object(self, kind, payload):
        with self.lock, self.connect() as conn:
            conn.execute('INSERT INTO objects VALUES (?,?,?,?)',
                         (payload['id'], kind, payload['created_at'], json.dumps(payload, allow_nan=False)))

    def history(self, limit=50):
        with self.connect() as conn:
            rows = conn.execute('SELECT payload FROM inspections ORDER BY created_at DESC LIMIT ?', (limit,)).fetchall()
        return [json.loads(row['payload']) for row in rows]

    def persist(self, result, image, mode, identity=None):
        identity = identity or str(uuid.uuid4())
        payload = _safe({**result, 'id': identity, 'created_at': _now(), 'mode': mode})
        payload.pop('image_base64', None)
        with self.lock, self.connect() as conn:
            conn.execute('INSERT OR REPLACE INTO inspections VALUES (?,?,?,?)',
                         (identity, payload['created_at'], json.dumps(payload, allow_nan=False), _jpeg(image)))
            conn.execute('DELETE FROM inspections WHERE id NOT IN (SELECT id FROM inspections ORDER BY created_at DESC LIMIT 100)')
        return payload

    def ensure_models(self):
        with self.inference_lock:
            if self.detector is not None:
                return
            if not self.weights.is_file():
                raise HTTPException(503, 'Trained endpoint weights are missing; configure SHERLOCK_ENDPOINT_WEIGHTS')
            if not self.refiner_weights.is_file():
                raise HTTPException(503, 'Endpoint refiner weights are missing; configure SHERLOCK_ENDPOINT_REFINER')
            try:
                from vision_ml.endpoint_refiner import EndpointRefiner
                detector = EndpointDetector(self.weights, device=self.device)
                refiner = EndpointRefiner(self.refiner_weights, device=self.device)
                self.detector, self.refiner = detector, refiner
            except Exception as exc:
                raise HTTPException(503, f'Trained endpoint model could not be loaded: {exc}') from exc

    def new_inspector(self, setup_path, reference):
        self.ensure_models()
        # Hydrate the same LiveInspector without loading duplicate model copies
        # per browser session. Its normal inspect/quality/verifier path is used.
        inspector = LiveInspector.__new__(LiveInspector)
        setup = json.loads(Path(setup_path).read_text())
        identity = setup.pop('setup_id')
        if content_fingerprint(setup) != identity:
            raise ValueError('Calibration changed; recalibrate the board')
        inspector.setup, inspector.setup_id = setup, identity
        inspector.grid = HoleGrid(**setup['grid'])
        config = dict(setup['registration'])
        config['marker_positions_mm'] = {int(k): v for k, v in config['marker_positions_mm'].items()}
        inspector.registration = BreadboardRegistration(**config)
        if inspector.registration.output_size != inspector.grid.image_size:
            raise ValueError('Calibration image sizes differ')
        if inspector.grid.pitch * min(1, 1280 / max(inspector.grid.image_size)) < 12:
            raise ValueError('Too few image pixels between holes; use a closer view and recalibrate')
        inspector.detector, inspector.refiner = self.detector, self.refiner
        inspector.verifier = LiveVerifier(inspector.grid, reference)
        return inspector

    def reset_session(self, identity):
        with self.lock:
            session = self.sessions.get(identity)
        if session:
            with session.lock:
                session.inspector.verifier.uncertain('Frame rejected; waiting for fresh observations.')
                session.last_at, session.last_digest, session.last_mode = 0., None, None

    def session(self, identity):
        with self.lock:
            session = self.sessions.get(identity)
        if session is None:
            raise HTTPException(404, 'Session not found; start a new inspection session')
        return session

    @staticmethod
    def registration_preview(image, columns):
        # Preserve >=12 model pixels per column, including the 63-column board.
        reg = BreadboardRegistration(output_size=(1400, 520))
        try:
            result = reg.register(image, enforce_quality=False)
        except (ValueError, TypeError, cv2.error) as exc:
            raise HTTPException(422, f'Board registration rejected: {exc}') from exc
        if result.rectified is None:
            raise HTTPException(422, 'Show all four ArUco DICT_4X4_50 markers, IDs 0–3 clockwise')
        reg.brightness_baseline = result.quality.brightness
        quality = reg.quality_report(result.rectified)
        if not quality.passed:
            raise HTTPException(422, 'Empty-board quality failed: ' + ', '.join(quality.failures))
        return reg, result.rectified, _safe({**asdict(quality), 'passed': quality.passed})


def register_live_api(app: FastAPI):
    backend = LiveBackend(app.state.storage.root)
    app.state.live_api = backend
    router = APIRouter(prefix='/api/live', tags=['live hole inspection'], route_class=_SafeRoute)

    @router.get('/status')
    def status():
        available = backend.weights.is_file() and backend.refiner_weights.is_file()
        return {'model': {'available': available, 'loaded': backend.detector is not None,
                          'device': backend.device, 'supported_types': ['wire', 'resistor'],
                          'name': 'YOLO11n endpoints + patch refiner', 'mock': False},
                'setups': backend.objects('setup'), 'references': backend.objects('reference'),
                'markers': {'dictionary': 'DICT_4X4_50', 'ids_clockwise': [0, 1, 2, 3]},
                'required_stable_frames': 3, 'valid_for_ms': 2000,
                'scope': 'visible_wire_resistor_topology',
                'limitations': ['Prototype: real-camera accuracy is not established.',
                               'No electrical contact, resistor-value, LED, IC, or power-rail verification.']}

    @router.get('/references')
    def references():
        return {'references': backend.objects('reference')}

    @router.post('/references', status_code=201)
    def create_reference(body: ReferenceBody):
        payload = body.model_dump()
        payload.update(id=str(uuid.uuid4()), created_at=_now(), name=body.name.strip())
        backend.save_object('reference', payload)
        return payload

    @router.get('/setups')
    def setups():
        return {'setups': backend.objects('setup')}

    @router.post('/calibration/preview')
    def calibration_preview(body: CalibrationBody):
        frame = decode_image(body.image_base64)
        _, image, quality = backend.registration_preview(frame, body.columns)
        return _with_image({'quality': quality, 'columns': body.columns,
                            'anchor_names': ['a1', f'a{body.columns}', f'j{body.columns}', 'j1']}, image)

    @router.post('/setups', status_code=201)
    def create_setup(body: SetupBody):
        frame = decode_image(body.image_base64)
        reg, image, quality = backend.registration_preview(frame, body.columns)
        try:
            grid = HoleGrid.from_anchors(image.shape[1::-1], body.columns, body.anchors).refined(image)
            if grid.pitch * min(1, 1280 / max(grid.image_size)) < 12:
                raise ValueError('Too few pixels between holes: make the board larger in the marker rectangle')
            observation = grid.observe(image)
            if not observation['passed']:
                raise ValueError('Measured grid does not align with enough visible holes; check the four anchors')
            setup = {'format_version': 1,
                     'grid': {'holes': grid.holes, 'columns': grid.columns,
                              'image_size': grid.image_size, 'rail_segments': grid.rail_segments},
                     'registration': {'brightness_baseline': reg.brightness_baseline,
                                      'sharpness_threshold': reg.sharpness_threshold,
                                      'pixels_per_mm': reg.pixels_per_mm, 'output_size': reg.output_size,
                                      'marker_positions_mm': {i: (p / reg.pixels_per_mm).tolist()
                                                              for i, p in enumerate(reg.destination)}},
                     'calibration_observation': {k: v for k, v in observation.items() if k != 'matches'}}
            setup['setup_id'] = content_fingerprint(setup)
            identity = str(uuid.uuid4())
            (backend.root / f'{identity}.json').write_text(json.dumps(setup, allow_nan=False))
            payload = {'id': identity, 'name': body.name.strip() or 'Breadboard', 'columns': body.columns,
                       'created_at': _now(), 'setup_id': setup['setup_id'],
                       'quality': quality, 'observation': setup['calibration_observation']}
            backend.save_object('setup', payload)
            overlay = draw_overlay(image, {'status': 'CALIBRATION', 'light_color': 'AMBER',
                                           'message': 'Review hole labels before use.'}, grid)
            for name in body.anchors:
                point = tuple(np.rint(grid.holes[name]).astype(int))
                cv2.putText(overlay, name, point, cv2.FONT_HERSHEY_SIMPLEX, .65, (20, 20, 220), 2)
            return _with_image(payload, overlay)
        except (ValueError, TypeError, cv2.error) as exc:
            raise HTTPException(422, f'Calibration rejected: {exc}') from exc

    @router.post('/preview')
    def preview(body: ImageBody):
        image = decode_image(body.image_base64)
        started = time.perf_counter()
        with backend.inference_lock:
            backend.ensure_models()
            try:
                detections = backend.detector.detect(image)
                detections = backend.refiner.refine(image, detections)
            except Exception as exc:
                raise HTTPException(503, f'Endpoint inference failed: {exc}') from exc
        overlay = image.copy()
        for detection in detections:
            x1, y1, x2, y2 = map(round, detection['bbox'])
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (230, 180, 30), 2)
            cv2.putText(overlay, f"{detection['type']} {detection['conf']:.2f}", (x1, max(20, y1)),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (230, 180, 30), 1)
            for point in detection['keypoints']:
                cv2.circle(overlay, (round(point['x']), round(point['y'])), 5, (0, 170, 240), 2)
        result = {'status': 'UNCALIBRATED', 'light_color': 'AMBER', 'passed': False,
                  'message': 'Endpoint predictions only. Calibrate the board and choose a reference to check wiring.',
                  'detections': detections, 'parts': [], 'findings': [],
                  'elapsed_ms': round(1000 * (time.perf_counter() - started), 1)}
        saved = backend.persist(result, overlay, 'preview')
        return _with_image(saved, overlay)

    @router.post('/sessions', status_code=201)
    def create_session(body: SessionBody):
        backend.object('setup', body.setup_id)
        reference = backend.object('reference', body.reference_id)
        try:
            with backend.inference_lock:
                inspector = backend.new_inspector(backend.root / f'{body.setup_id}.json', reference)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            raise HTTPException(422, f'Setup/reference could not be used: {exc}') from exc
        with backend.lock:
            # Bound abandoned browser sessions without ever reusing their state.
            if len(backend.sessions) >= 16:
                raise HTTPException(429, 'Too many active sessions; close an existing live session')
            identity = str(uuid.uuid4())
            backend.sessions[identity] = Session(inspector, body.setup_id, body.reference_id)
        return {'session_id': identity, 'setup_id': body.setup_id, 'reference_id': body.reference_id,
                'required_stable_frames': 3, 'valid_for_ms': 2000}

    @router.post('/sessions/{session_id}/frame')
    def inspect_frame(session_id: str, body: FrameBody):
        started = time.monotonic()
        session = backend.session(session_id)
        with session.lock:
            try:
                if session.closed:
                    raise HTTPException(404, 'Session ended; start a new inspection session')
                image = decode_image(body.image_base64)
                digest = hashlib.sha256(image.tobytes()).hexdigest()
                if body.frame_id is not None and body.frame_id <= session.last_frame_id:
                    raise HTTPException(409, 'frame_id must increase for each session frame')
                now = time.monotonic()
                reset = (body.mode != 'live' or session.last_mode != 'live' or
                         now - session.last_at > 2 or digest == session.last_digest)
                if reset:
                    session.inspector.verifier.uncertain('Waiting for fresh consecutive camera frames.')
                with backend.inference_lock:
                    result, overlay = session.inspector.inspect(image)
                total_ms = round(1000 * (time.monotonic() - started), 1)
                result['elapsed_ms'] = total_ms
                if total_ms > 2000:
                    result = {**result, **session.inspector.verifier.uncertain(
                        'Frame processing was too slow for fresh live feedback.',
                        detections=result.get('detections'), parts=result.get('parts'))}
                    overlay = draw_overlay(overlay, result)
                session.last_at, session.last_digest, session.last_mode = time.monotonic(), digest, body.mode
                if body.frame_id is not None:
                    session.last_frame_id = body.frame_id
                if body.mode == 'still':
                    # Still images can expose a candidate, but cannot establish a
                    # temporally confirmed green, even when posted repeatedly.
                    result.update(status='UNCERTAIN', light_color='AMBER', passed=False)
                    if result.get('candidate_status'):
                        result['message'] = 'Still image only; start live capture for a confirmed wiring verdict.'
                    session.inspector.verifier.uncertain('Still image cannot establish a live pass.')
                    overlay = draw_overlay(overlay, result)
                result.update(session_id=session_id, setup_id=session.setup_id,
                              reference_id=session.reference_id, mode=body.mode)
                saved = backend.persist(result, overlay, body.mode,
                                        identity=f'live-{session_id}' if body.mode == 'live' else None)
                return _with_image(saved, overlay)
            except HTTPException:
                backend.reset_session(session_id)
                raise
            except Exception as exc:
                backend.reset_session(session_id)
                raise HTTPException(503, f'Inspection failed; live stability reset: {exc}') from exc

    @router.delete('/sessions/{session_id}')
    def delete_session(session_id: str):
        with backend.lock:
            session = backend.sessions.pop(session_id, None)
        if session is None:
            raise HTTPException(404, 'Session not found')
        with session.lock:
            session.closed = True
            session.inspector.verifier.uncertain('Session ended.')
        return {'deleted': True}

    @router.get('/history')
    def history():
        return {'inspections': backend.history()}

    @router.get('/inspections/{inspection_id}')
    def inspection(inspection_id: str):
        with backend.connect() as conn:
            row = conn.execute('SELECT payload,jpeg FROM inspections WHERE id=?', (inspection_id,)).fetchone()
        if row is None:
            raise HTTPException(404, 'Inspection not found')
        return {**json.loads(row['payload']), 'image_base64': base64.b64encode(row['jpeg']).decode('ascii')}

    app.include_router(router)
