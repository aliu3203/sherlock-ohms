"""Marker-tracked endpoint/hole inspection with an explicit uncertainty state.

Run ``python -m vision_ml.live_inspector --help`` for local calibration and video
commands. All outputs go to an explicitly requested path; no web service needed.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import cv2
import numpy as np

from engine.live_verifier import LiveVerifier
from .endpoint_detector import EndpointDetector
from .hole_grid import HoleGrid
from .registration import BreadboardRegistration, content_fingerprint


def _quality(report):
    return {k: (None if isinstance(v, float) and not np.isfinite(v) else v)
            for k, v in asdict(report).items()}


class LiveInspector:
    def __init__(self, *, setup, weights, reference, device='cpu', image_size=1280, refiner_weights=None):
        self.setup = json.loads(Path(setup).read_text())
        identity = self.setup.pop('setup_id')
        if content_fingerprint(self.setup) != identity:
            raise ValueError('Setup contents changed; recalibrate to create a consistent setup')
        self.setup_id = identity
        self.grid = HoleGrid(**self.setup['grid'])
        config = self.setup['registration'].copy()
        if config.get('marker_positions_mm'):
            config['marker_positions_mm'] = {int(k): v for k, v in config['marker_positions_mm'].items()}
        self.registration = BreadboardRegistration(**config)
        if self.registration.output_size != self.grid.image_size:
            raise ValueError('Registration and hole-grid image sizes differ')
        if self.grid.pitch * min(1, image_size / max(self.grid.image_size)) < 12:
            raise ValueError('Too few model pixels between holes; use a larger image_size or closer view')
        self.detector = EndpointDetector(weights, device=device, image_size=image_size)
        self.refiner = None
        if refiner_weights is not None:
            from .endpoint_refiner import EndpointRefiner
            self.refiner = EndpointRefiner(refiner_weights, device=device)
        if isinstance(reference, (str, Path)):
            reference = json.loads(Path(reference).read_text())
        self.verifier = LiveVerifier(self.grid, reference)

    def inspect(self, frame):
        if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or
                frame.ndim != 3 or frame.shape[2] != 3 or not frame.size):
            self.verifier.uncertain('Invalid frame')
            raise ValueError('Expected a nonempty uint8 BGR frame')
        start = time.perf_counter()
        observed_at = round(time.time() * 1000)
        try:
            registration = self.registration.register(frame, enforce_quality=False)
        except ValueError as exc:
            verdict = self.verifier.uncertain(f'Board registration failed: {exc}')
            verdict.update({'observed_at_unix_ms': observed_at, 'valid_for_ms': 2000,
                            'setup_id': self.setup_id, 'quality': None, 'grid_observation': None,
                            'elapsed_ms': round(1000 * (time.perf_counter() - start), 1)})
            return verdict, draw_overlay(frame, verdict)
        rectified = registration.rectified
        quality = _quality(registration.quality)
        observation = None
        if not registration.quality.passed or rectified is None:
            verdict = self.verifier.uncertain('Image quality or markers failed: ' + ', '.join(registration.quality.failures))
        else:
            observation = self.grid.observe(rectified)
            if not observation['passed']:
                verdict = self.verifier.uncertain('Visible holes do not support the saved board calibration.')
            else:
                try:
                    detections = self.detector.detect(rectified)
                    if self.refiner is not None:
                        detections = self.refiner.refine(rectified, detections)
                    verdict = self.verifier.inspect(detections)
                except (ValueError, RuntimeError, TypeError) as exc:
                    verdict = self.verifier.uncertain(f'Endpoint inference failed: {exc}')
        elapsed = round(1000 * (time.perf_counter() - start), 1)
        if elapsed > 2000:
            verdict = self.verifier.uncertain('Inference was too slow for fresh live feedback.')
        verdict.update({'setup_id': self.setup_id, 'quality': quality,
                        'observed_at_unix_ms': observed_at, 'valid_for_ms': 2000,
                        'grid_observation': None if observation is None else {k: v for k, v in observation.items() if k != 'matches'},
                        'elapsed_ms': elapsed})
        overlay = draw_overlay(rectified if rectified is not None else frame, verdict,
                               self.grid if rectified is not None else None)
        return verdict, overlay


def draw_overlay(frame, verdict, grid=None):
    image = frame.copy()
    color = {'GREEN': (40, 180, 40), 'RED': (30, 30, 230), 'AMBER': (0, 170, 240)}[verdict['light_color']]
    if grid is not None:
        for point in grid.xy:
            cv2.circle(image, tuple(np.rint(point).astype(int)), 1, (210, 180, 100), -1)
    raw = {d.get('id'): d for d in verdict.get('detections', [])}
    for part in verdict.get('parts', []):
        for index, endpoint in enumerate(part['endpoints']):
            name = endpoint['hole']
            if name and grid is not None:
                pos = tuple(np.rint(grid.holes[name]).astype(int))
                cv2.circle(image, pos, 5, color, 2)
                cv2.putText(image, name, (pos[0] + 5, pos[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, .4, color, 1)
            elif part['id'] in raw:
                k = raw[part['id']]['keypoints'][index]
                if k.get('x') is not None and k.get('y') is not None:
                    pos = (round(k['x']), round(k['y']))
                    cv2.circle(image, pos, 6, (0, 170, 240), 2)
                    cv2.putText(image, '?', pos, cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 170, 240), 1)
    cv2.rectangle(image, (0, 0), (image.shape[1], 47), (35, 35, 35), -1)
    cv2.putText(image, f"{verdict['status']}: {verdict['message']}", (8, 29),
                cv2.FONT_HERSHEY_SIMPLEX, .55, color, 1)
    return image


def _click_anchors(image, names):
    points = {}
    title = 'Calibration: click named hole centers; Esc cancels'
    canvas = image.copy()
    def click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < len(names):
            points[names[len(points)]] = [x, y]
    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(title, click)
    try:
        while len(points) < len(names):
            canvas[:] = image
            cv2.putText(canvas, 'Click ' + names[len(points)], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 255), 2)
            for name, p in points.items():
                cv2.circle(canvas, tuple(p), 5, (0, 180, 0), 2)
                cv2.putText(canvas, name, tuple(p), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 180, 0), 1)
            cv2.imshow(title, canvas)
            if cv2.waitKey(20) & 255 == 27 or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                raise ValueError('Calibration cancelled')
    finally:
        cv2.destroyWindow(title)
    return points


def calibrate(args):
    source = cv2.imread(str(args.image))
    if source is None:
        raise ValueError('Cannot read the empty-board image')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'setup.json').exists():
        raise ValueError('Choose a new output directory; existing calibration will not be overwritten')
    registration = (BreadboardRegistration.from_calibration(args.registration) if args.registration else
                    BreadboardRegistration(output_size=tuple(args.size)))
    result = registration.register(source, enforce_quality=False)
    if result.rectified is None:
        raise ValueError('Need visible ArUco DICT_4X4_50 markers 0,1,2,3 clockwise around the board')
    registration.brightness_baseline = result.quality.brightness
    quality = registration.quality_report(result.rectified)
    if not quality.passed:
        raise ValueError('Empty-board quality failed: ' + ', '.join(quality.failures))
    cv2.imwrite(str(out / 'rectified_empty.jpg'), result.rectified)
    names = ['a1', f'a{args.columns}', f'j{args.columns}', 'j1']
    anchors_data = json.loads(Path(args.anchors).read_text()) if args.anchors else {'anchors': _click_anchors(result.rectified, names)}
    grid = HoleGrid.from_anchors(result.rectified.shape[1::-1], args.columns,
                                anchors_data['anchors'], rail_holes=anchors_data.get('rail_holes'),
                                rail_segments=anchors_data.get('rail_segments'))
    grid = grid.refined(result.rectified)
    observation = grid.observe(result.rectified)
    if not observation['passed']:
        raise ValueError('Refined grid did not align with visible holes')
    positions = {i: (p / registration.pixels_per_mm).tolist() for i, p in enumerate(registration.destination)}
    setup = {'format_version': 1, 'grid': {'holes': grid.holes, 'columns': grid.columns,
             'image_size': grid.image_size, 'rail_segments': grid.rail_segments},
             'registration': {'brightness_baseline': registration.brightness_baseline,
                'sharpness_threshold': registration.sharpness_threshold,
                'pixels_per_mm': registration.pixels_per_mm, 'output_size': registration.output_size,
                'marker_positions_mm': positions},
             'calibration_observation': {k: v for k, v in observation.items() if k != 'matches'}}
    setup['setup_id'] = content_fingerprint(setup)
    (out / 'setup.json').write_text(json.dumps(setup, indent=2, allow_nan=False))
    preview = draw_overlay(result.rectified, {'light_color': 'AMBER', 'status': 'CALIBRATION',
                           'message': 'Review hole numbering and alignment before use.'}, grid)
    for name in names:
        point = tuple(np.rint(grid.holes[name]).astype(int))
        cv2.putText(preview, name, point, cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 0, 220), 2)
    cv2.imwrite(str(out / 'calibration_overlay.jpg'), preview)
    print(json.dumps({'setup': str((out / 'setup.json').resolve()), 'observation': setup['calibration_observation']}, indent=2))


def inspect(args):
    inspector = LiveInspector(setup=args.setup, weights=args.weights, reference=args.reference,
                              device=args.device, image_size=args.image_size, refiner_weights=args.refiner)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'verdicts.jsonl').exists():
        raise ValueError('Choose a new output directory for this inspection')
    cap = None
    if args.camera is not None or args.video:
        cap = cv2.VideoCapture(args.camera if args.camera is not None else str(args.video))
        if not cap.isOpened():
            raise ValueError('Cannot open camera/video')
    try:
        with (output / 'verdicts.jsonl').open('w') as log:
            index = 0
            while True:
                if cap is None:
                    frame = cv2.imread(str(args.image))
                    ok = frame is not None
                else:
                    ok, frame = cap.read()
                if not ok:
                    if index == 0:
                        raise ValueError('Cannot read an input frame')
                    break
                verdict, overlay = inspector.inspect(frame)
                verdict['frame_index'] = index
                log.write(json.dumps(verdict, allow_nan=False) + '\n')
                log.flush()
                cv2.imwrite(str(output / 'latest.jpg'), overlay)
                print(f"{index}: {verdict['status']} — {verdict['message']}", flush=True)
                index += 1
                if args.show:
                    cv2.imshow('Sherlock Ohms — observed connectivity', overlay)
                    if cv2.waitKey(1) & 255 == 27:
                        break
                if cap is None or args.max_frames and index >= args.max_frames:
                    break
    finally:
        if cap is not None:
            cap.release()
        if args.show:
            cv2.destroyAllWindows()


def detect(args):
    """A useful image-only smoke test; deliberately has no correctness verdict."""
    frame = cv2.imread(str(args.image))
    if frame is None:
        raise ValueError('Cannot read image')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'endpoints.json').exists():
        raise ValueError('Choose a new output directory')
    detector = EndpointDetector(args.weights, device=args.device, image_size=args.image_size)
    detections = detector.detect(frame)
    if args.refiner:
        from .endpoint_refiner import EndpointRefiner
        detections = EndpointRefiner(args.refiner, device=args.device).refine(frame, detections)
    (output / 'endpoints.json').write_text(json.dumps({'status': 'UNCALIBRATED',
        'message': 'Endpoint predictions only; no hole IDs or correctness verdict.',
        'detections': detections}, indent=2, allow_nan=False))
    for d in detections:
        x1, y1, x2, y2 = map(round, d['bbox'])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (230, 180, 30), 1)
        cv2.putText(frame, f"{d['type']} {d['conf']:.2f}", (x1, max(40, y1)), cv2.FONT_HERSHEY_SIMPLEX, .45, (230, 180, 30), 1)
        for k in d['keypoints']:
            cv2.circle(frame, (round(k['x']), round(k['y'])), 4, (230, 180, 30), 2)
    cv2.putText(frame, 'ENDPOINT PREVIEW: no circuit verdict', (10, 24), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 170, 240), 2)
    cv2.imwrite(str(output / 'endpoints.jpg'), frame)
    print(json.dumps({'detections': len(detections), 'output': str(output.resolve())}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    cal = sub.add_parser('calibrate', help='Measure an empty board with four ArUco markers')
    cal.add_argument('--image', required=True)
    cal.add_argument('--columns', type=int, choices=(17, 30, 63), default=63)
    cal.add_argument('--size', type=int, nargs=2, default=(1400, 520), metavar=('WIDTH', 'HEIGHT'))
    cal.add_argument('--registration', help='Optional existing measured registration calibration.json')
    cal.add_argument('--anchors', help='JSON with anchors and optional measured rail_holes/rail_segments; otherwise click')
    cal.add_argument('--output', required=True)
    cal.set_defaults(func=calibrate)
    run = sub.add_parser('inspect', help='Inspect a raw marker-visible image, video or camera')
    run.add_argument('--setup', required=True)
    run.add_argument('--weights', required=True)
    run.add_argument('--refiner', help='Optional small endpoint-refiner checkpoint')
    run.add_argument('--reference', required=True)
    source = run.add_mutually_exclusive_group(required=True)
    source.add_argument('--image')
    source.add_argument('--video')
    source.add_argument('--camera', type=int)
    run.add_argument('--device', choices=('cpu', 'mps'), default='cpu')
    run.add_argument('--image-size', type=int, default=1280)
    run.add_argument('--output', required=True)
    run.add_argument('--show', action='store_true')
    run.add_argument('--max-frames', type=int, default=0)
    run.set_defaults(func=inspect)
    preview = sub.add_parser('detect', help='Preview endpoints on any photo without calibration or markers')
    preview.add_argument('--image', required=True)
    preview.add_argument('--weights', required=True)
    preview.add_argument('--refiner', help='Optional small endpoint-refiner checkpoint')
    preview.add_argument('--device', choices=('cpu', 'mps'), default='cpu')
    preview.add_argument('--image-size', type=int, default=1280)
    preview.add_argument('--output', required=True)
    preview.set_defaults(func=detect)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
