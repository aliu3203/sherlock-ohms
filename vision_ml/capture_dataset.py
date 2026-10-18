"""Explicit raw-frame preparation and EMPTY-board baseline capture.

Examples (run from repository root):
  python -m vision_ml.capture_dataset capture --calibration calib/calibration.json \
    --setup-id bench-v1 --output-dir captures --source local --images raw.png
  python -m vision_ml.capture_dataset baseline --calibration calib/calibration.json \
    --setup-id bench-v1 --output-dir captures --source live --count 3 --confirm-empty-board

A setup ID is the operator's declaration of the physical board, camera position,
lens, and lighting. Change it and recalibrate when any of those change. Local
image origin cannot be inferred from pixels: provenance records the input path.
This module never learns a baseline during ordinary capture or inspection.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import uuid

import cv2
import numpy as np

from .pi_hardware import BreadboardCamera
from .registration import BreadboardRegistration, calibration_identity, _gray


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def _write_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(f'Cannot write {path}')


class CaptureSession:
    """One calibration/setup and capture backend per output directory.

    Images are saved even on quality rejection. Only explicitly declared empty,
    accepted records from this session can be selected for a median baseline.
    Existing baselines are immutable; use a new session to replace one.
    """

    def __init__(self, calibration_path, output_dir, *, setup_id, source):
        if source not in ('local', 'pi'):
            raise ValueError('source must be local or pi')
        if not isinstance(setup_id, str) or not setup_id.strip():
            raise ValueError('An explicit setup_id is required')
        calibration_path = Path(calibration_path)
        calibration = json.loads(calibration_path.read_text())
        lut = json.loads(calibration_path.with_name('hole_lut.json').read_text())
        identity = calibration_identity(calibration, lut)
        if calibration.get('calibration_id', identity) != identity:
            raise ValueError('Calibration/LUT fingerprint mismatch')
        if calibration.get('setup_id', setup_id) != setup_id:
            raise ValueError('Physical setup ID does not match calibration')
        if calibration.get('rectified_size', [1400, 520]) != [1400, 520]:
            raise ValueError('Capture expects 1400x520 calibration')
        self.registration = BreadboardRegistration.from_calibration(calibration_path)
        self.root = Path(output_dir)
        self.binding = dict(calibration_id=identity, setup_id=setup_id, source=source)
        manifest = self.root / 'session.json'
        if manifest.exists():
            if json.loads(manifest.read_text()) != self.binding:
                raise ValueError('Output directory belongs to a different calibration/setup/source')
        else:
            if self.root.exists() and any(self.root.iterdir()):
                raise ValueError('Refusing nonempty directory without session identity')
            self.root.mkdir(parents=True, exist_ok=True)
            _write_json(self.root / 'calibration.json', calibration)
            _write_json(self.root / 'hole_lut.json', lut)
            _write_json(manifest, self.binding)

    def save_frame(self, image, provenance, *, empty_board=False):
        """empty_board is an explicit operator declaration, never an inference."""
        _gray(image)
        if provenance.get('source') != self.binding['source']:
            raise ValueError('Capture source differs from session source')
        if not provenance.get('timestamp'):
            raise ValueError('Capture provenance needs a timestamp')
        if self.binding['source'] == 'local' and not provenance.get('local_path'):
            raise ValueError('Local capture needs its original image path')
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        record_id = uuid.uuid4().hex
        folder = self.root / record_id
        folder.mkdir()
        _write_image(folder / 'raw.png', image)
        record = dict(self.binding, id=record_id, captured_at=time.time(),
                      empty_board=bool(empty_board), provenance=provenance,
                      raw_sha256=_hash(folder / 'raw.png'))
        try:
            result = self.registration.register(image, enforce_quality=False)
            quality = asdict(result.quality)
            quality['passed'] = result.quality.passed
            quality = {k: None if isinstance(v, float) and not np.isfinite(v) else v
                       for k, v in quality.items()}
            record['homography'] = result.homography.tolist() if result.homography is not None else None
            record['marker_centers'] = {str(k): v.tolist() for k, v in result.marker_centers.items()}
            if result.rectified is not None:
                _write_image(folder / 'rectified.png', result.rectified)
                record['rectified_sha256'] = _hash(folder / 'rectified.png')
        except ValueError as exc:
            quality = {'passed': False, 'failures': ['registration'], 'error': str(exc)}
        record['quality'] = quality
        _write_json(folder / 'quality.json', quality)
        _write_json(folder / 'record.json', record)
        return record

    def create_empty_baseline(self, record_ids):
        if len(record_ids) < 3 or len(set(record_ids)) != len(record_ids):
            raise ValueError('At least three distinct EMPTY-board records are required')
        baseline_path = self.root / 'empty_baseline.png'
        metadata_path = self.root / 'empty_baseline.json'
        if baseline_path.exists() or metadata_path.exists():
            raise FileExistsError('Baseline already exists; use a new session to replace it')
        frames, origins, records = [], set(), []
        camera_controls = None
        for record_id in record_ids:
            if not isinstance(record_id, str) or len(record_id) != 32 or any(c not in '0123456789abcdef' for c in record_id):
                raise ValueError('Invalid capture record ID')
            folder = self.root / record_id
            record = json.loads((folder / 'record.json').read_text())
            if any(record.get(k) != v for k, v in self.binding.items()):
                raise ValueError('Baseline frame calibration/setup/source mismatch')
            if not record['empty_board'] or not record['quality']['passed']:
                raise ValueError('Baseline requires explicitly EMPTY quality-approved frames')
            source = record['provenance']
            if self.binding['source'] == 'pi':
                controls = source.get('locked_controls', {})
                if camera_controls is not None and controls != camera_controls:
                    raise ValueError('Baseline camera controls changed between frames')
                camera_controls = controls
            # Prevent cyclic replay of a single local file from counting as three frames.
            origin = (str(Path(source['local_path']).resolve()) if self.binding['source'] == 'local'
                      else (source['timestamp'], source['sequence']))
            if origin in origins:
                raise ValueError('Baseline requires three separately acquired frames')
            origins.add(origin)
            path = folder / 'rectified.png'
            if _hash(path) != record['rectified_sha256'] or _hash(folder / 'raw.png') != record['raw_sha256']:
                raise ValueError('Capture image fingerprint mismatch')
            image = cv2.imread(str(path))
            if image is None or image.shape != (520, 1400, 3):
                raise ValueError('Invalid rectified baseline frame')
            frames.append(image)
            records.append({'id': record_id, 'sha256': record['rectified_sha256']})
        median = np.median(np.stack(frames), axis=0).astype(np.uint8)
        _write_image(baseline_path, median)
        _write_json(metadata_path, dict(self.binding, method='pixel_median',
                    empty_board_confirmed=True, frames=records,
                    locked_controls=camera_controls, image_sha256=_hash(baseline_path)))
        return baseline_path

    def load_baseline(self):
        metadata = json.loads((self.root / 'empty_baseline.json').read_text())
        path = self.root / 'empty_baseline.png'
        if (any(metadata.get(k) != v for k, v in self.binding.items())
                or _hash(path) != metadata['image_sha256']):
            raise ValueError('Baseline calibration/setup/image mismatch')
        return cv2.imread(str(path))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('capture', 'baseline'))
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--setup-id', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--source', choices=('local', 'live'), required=True)
    parser.add_argument('--images', nargs='+', type=Path)
    parser.add_argument('--count', type=int, default=1)
    parser.add_argument('--confirm-empty-board', action='store_true')
    args = parser.parse_args(argv)
    if args.count < 1:
        parser.error('--count must be positive')
    empty = args.operation == 'baseline'
    if empty and (args.count < 3 or not args.confirm_empty_board):
        parser.error('baseline needs --count >= 3 and --confirm-empty-board')
    if args.source == 'live' and args.images:
        parser.error('Live capture does not accept fallback images')
    if args.source == 'local' and (not args.images or len(args.images) < args.count):
        parser.error('Supply at least --count separate local image paths')
    if empty and args.images and len({p.resolve() for p in args.images[:args.count]}) < args.count:
        parser.error('Empty baseline needs distinct source files')
    session = CaptureSession(args.calibration, args.output_dir, setup_id=args.setup_id,
                             source='pi' if args.source == 'live' else 'local')
    records = []
    with BreadboardCamera(args.images, force_fallback=args.source == 'local',
                          require_live=args.source == 'live') as camera:
        print(f'Capture source: {camera.source}; setup: {args.setup_id}')
        for _ in range(args.count):
            frame = camera.capture_frame()
            record = session.save_frame(frame, camera.last_capture, empty_board=empty)
            records.append(record)
            print(f"{record['id']}: source={camera.last_capture['source']} quality={record['quality']}")
    if empty:
        print(session.create_empty_baseline([r['id'] for r in records]))
    return 0 if all(r['quality']['passed'] for r in records) else 1


if __name__ == '__main__':
    raise SystemExit(main())
