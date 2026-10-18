"""Generate printable markers and click a1,a63,j1,j63 on a rectified board.

Run python -m vision_ml.calibrate_cli --image rectified.png. Without --image,
only the printable marker sheet is generated. Print without resizing and record
actual marker-center positions in registration if using a different layout.
"""
import argparse
import json
import uuid
from pathlib import Path
import cv2
import numpy as np
from .registration import PIXELS_PER_MM, _gray, BreadboardRegistration, calibration_identity

PACKAGE_DIR = Path(__file__).resolve().parent
REFERENCE_NAMES = ('a1', 'a63', 'j1', 'j63')


def save_marker_sheet(output=PACKAGE_DIR / 'aruco_markers_print.png', marker_size=400):
    if marker_size < 32:
        raise ValueError('Marker size must be at least 32 pixels')
    margin, label_height = 80, 70
    cell = marker_size + 2*margin
    sheet = np.full((2*(cell+label_height), 2*cell, 3), 255, np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    for marker_id, (column, row) in enumerate(((0, 0), (1, 0), (1, 1), (0, 1))):
        x, y = column*cell+margin, row*(cell+label_height)+margin
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_size)
        sheet[y:y+marker_size, x:x+marker_size] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
        cv2.putText(sheet, f'ID {marker_id} - DICT_4X4_50', (x, y+marker_size+42),
                    cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 0, 0), 2)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), sheet):
        raise OSError(f'Cannot write {output}')
    return output


RAIL_PREFIXES = ('top_plus', 'top_minus', 'bottom_plus', 'bottom_minus')


def validate_geometry(geometry=None):
    """Rails are measured pixel centers, never extrapolated terminal columns.

    Geometry JSON: {"split_rails": false, "rails": {
      "top_plus": {"holes": {"top_plus_1": {"x": 40, "y": 30}, ...},
                   "segments": [["top_plus_1", ...]]}, ...}}
    Supply every actual rail hole and its electrically connected segment.
    Optional terminal_column_positions_mm (63 values) and
    terminal_row_positions_mm (10 values, a through j) describe terminal pitch.
    With split_rails=true the engine requires IDs 1..31 in the first segment,
    32..63 in the second; these indices need not align with terminal columns.
    Mixed rail topology or more than two segments requires an engine change.
    """
    geometry = json.loads(json.dumps(geometry if geometry is not None else {}))
    if not isinstance(geometry, dict):
        raise ValueError('Geometry must be a JSON object')
    if set(geometry) - {'split_rails', 'rails', 'terminal_column_positions_mm',
                        'terminal_row_positions_mm'}:
        raise ValueError('Unknown geometry field')
    split = geometry.setdefault('split_rails', False)
    if type(split) is not bool:
        raise ValueError('split_rails must be boolean')
    for key, default, count in (
        ('terminal_column_positions_mm', [i*2.54 for i in range(63)], 63),
        ('terminal_row_positions_mm', [i*2.54 for i in (0,1,2,3,4,7,8,9,10,11)], 10),
    ):
        values = np.asarray(geometry.setdefault(key, default), dtype=float)
        if values.shape != (count,) or not np.isfinite(values).all() or not (np.diff(values)>0).all():
            raise ValueError(f'{key} must contain {count} finite increasing values')
        geometry[key] = values.tolist()
    rails = geometry.setdefault('rails', {})
    if not isinstance(rails, dict) or set(rails) - set(RAIL_PREFIXES):
        raise ValueError('Unknown rail prefix')
    for prefix, rail in rails.items():
        if (not isinstance(rail, dict) or set(rail) != {'holes', 'segments'}
                or not isinstance(rail['holes'], dict) or not rail['holes']):
            raise ValueError('Each rail needs measured holes and explicit segments')
        expected = {}
        for name, point in rail['holes'].items():
            suffix = name.removeprefix(prefix+'_')
            if not suffix.isdigit() or name != f'{prefix}_{int(suffix)}' or not 1 <= int(suffix) <= 63:
                raise ValueError(f'Invalid rail hole ID: {name}')
            if not isinstance(point, dict) or set(point) != {'x', 'y'}:
                raise ValueError('Hole coordinates require x and y')
            xy = np.asarray([point['x'], point['y']], dtype=float)
            if not np.isfinite(xy).all() or not (0 <= xy[0] < 1400 and 0 <= xy[1] < 520):
                raise ValueError(f'Invalid coordinates: {name}')
            rail['holes'][name] = dict(zip(('x','y'), map(float, xy)))
            group = int(suffix)>31 if split else False
            expected.setdefault(group, set()).add(name)
        segments = rail['segments']
        if not isinstance(segments, list) or any(not isinstance(g, list) or not g for g in segments):
            raise ValueError('Segments must be nonempty lists of hole IDs')
        flat = [name for group in segments for name in group]
        if len(flat) != len(set(flat)) or set(flat) != set(rail['holes']):
            raise ValueError('Segments must partition every measured rail hole exactly once')
        if {frozenset(g) for g in segments} != {frozenset(g) for g in expected.values()}:
            raise ValueError('Rail segments cannot be represented by engine split_rails topology')
    return geometry


def create_hole_lut(reference_points, geometry=None):
    """Project terminal geometry from four anchors; append measured rail centers."""
    geometry = validate_geometry(geometry)
    try:
        points = np.asarray([reference_points[name] for name in REFERENCE_NAMES], np.float32)
    except (KeyError, TypeError) as exc:
        raise ValueError('Four reference holes are required') from exc
    if (points.shape != (4, 2) or not np.isfinite(points).all()
            or not ((points >= 0) & (points < [1400, 520])).all()):
        raise ValueError('Reference holes must be finite rectified pixel coordinates')
    clockwise = points[[0, 1, 3, 2]]
    if not cv2.isContourConvex(clockwise) or cv2.contourArea(clockwise, oriented=True) <= 1e-6:
        raise ValueError('Reference holes must form a clockwise nondegenerate quadrilateral')
    columns = geometry['terminal_column_positions_mm']
    rows = geometry['terminal_row_positions_mm']
    logical = np.float32([[columns[0], rows[0]], [columns[-1], rows[0]],
                          [columns[0], rows[-1]], [columns[-1], rows[-1]]])
    transform = cv2.getPerspectiveTransform(logical, points)
    lut = {}
    for row, logical_y in zip('abcdefghij', rows):
        coords = np.float32([[[column, logical_y] for column in columns]])
        coords = cv2.perspectiveTransform(coords, transform)[0]
        for column, (x, y) in enumerate(coords, 1):
            lut[f'{row}{column}'] = {'x': float(x), 'y': float(y)}
    for rail in geometry['rails'].values():
        lut.update(rail['holes'])
    return lut


def render_calibration_overlay(image, lut):
    """Dots for every hole; terminal landmarks and selected rail labels."""
    _gray(image)
    overlay = image.copy()
    if overlay.ndim == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
    rail_names = [name for name in lut if "_" in name]
    rail_labels = set()
    for prefix in RAIL_PREFIXES:
        names = [name for name in rail_names if name.startswith(prefix+"_")]
        if names:
            rail_labels.update(names[::5] + names[-1:])
    for name, point in lut.items():
        x, y = round(point['x']), round(point['y'])
        rail = '_' in name
        color = (0, 190, 255) if rail else (80, 240, 80)
        cv2.circle(overlay, (x, y), 2, color, -1)
        if name in rail_labels or (not rail and name[1:] in ('1', '16', '32', '48', '63')):
            cv2.putText(overlay, name, (max(0, min(x+3, 1270)), max(12, y-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, .28, color, 1, cv2.LINE_AA)
    return overlay


def save_calibration(image, reference_points, output_dir=PACKAGE_DIR, *,
                     geometry=None, setup_id=None, marker_positions_mm=None):
    gray = _gray(image)
    h, w = gray.shape
    if (w, h) != (1400, 520):
        raise ValueError('Calibration expects a 1400x520 rectified image')
    geometry = validate_geometry(geometry)
    lut = create_hole_lut(reference_points, geometry)
    calibration = {'version': 1, 'rectified_size': [w, h], 'pixels_per_mm': PIXELS_PER_MM,
                   'brightness_baseline': float(gray.mean()),
                   'sharpness_threshold': 100.0,
                   'reference_holes': {name: list(map(float, reference_points[name]))
                                       for name in REFERENCE_NAMES},
                   'hole_pitch_mm': 2.54, 'trench_center_spacing_mm': 7.62,
                   'marker_positions_mm': {0: [0, 0], 1: [(w-1)/PIXELS_PER_MM, 0],
                                           2: [(w-1)/PIXELS_PER_MM, (h-1)/PIXELS_PER_MM],
                                           3: [0, (h-1)/PIXELS_PER_MM]},
                   'marker_order': {'0': 'top_left', '1': 'top_right',
                                    '2': 'bottom_right', '3': 'bottom_left'}}
    calibration.update(version=2, geometry=geometry,
                       setup_id=str(uuid.uuid4()) if setup_id is None else setup_id)
    if not isinstance(calibration['setup_id'], str) or not calibration['setup_id'].strip():
        raise ValueError('setup_id must be a nonempty string')
    if marker_positions_mm is not None:
        calibration['marker_positions_mm'] = marker_positions_mm
    BreadboardRegistration(
        brightness_baseline=calibration['brightness_baseline'],
        marker_positions_mm={int(k): v for k, v in calibration['marker_positions_mm'].items()})
    # Canonicalize JSON keys before calculating identity.
    calibration = json.loads(json.dumps(calibration))
    calibration['calibration_id'] = calibration_identity(calibration, lut)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, data in [('calibration.json', calibration), ('hole_lut.json', lut)]:
        (output_dir / filename).write_text(json.dumps(data, indent=2)+'\n')
    if not cv2.imwrite(str(output_dir / 'calibration_overlay.png'),
                       render_calibration_overlay(image, lut)):
        raise OSError('Cannot write calibration overlay')
    return calibration, lut


def click_reference_holes(image):
    points = []
    window = 'Calibration: click a1, a63, j1, j63; R reset; Enter save; Esc cancel'

    def click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))

    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, click)
    try:
        while True:
            preview = image.copy()
            for name, point in zip(REFERENCE_NAMES, points):
                cv2.circle(preview, point, 5, (0, 0, 255), -1)
                cv2.putText(preview, name, (point[0]+8, point[1]-8),
                            cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 255), 2)
            cv2.imshow(window, preview)
            key = cv2.waitKey(20) & 0xff
            if key == 27 or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                raise RuntimeError('Calibration cancelled; no calibration saved')
            if key in (ord('r'), ord('R')):
                points.clear()
            if key in (10, 13) and len(points) == 4:
                return dict(zip(REFERENCE_NAMES, points))
    finally:
        cv2.destroyWindow(window)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path)
    parser.add_argument('--output-dir', type=Path, default=PACKAGE_DIR)
    parser.add_argument('--points', type=Path, help='JSON a1,a63,j1,j63 -> [x,y]; otherwise click')
    parser.add_argument('--geometry', type=Path, help=validate_geometry.__doc__)
    parser.add_argument('--setup-id', help='Physical board/camera/lighting setup identifier')
    parser.add_argument('--marker-positions', type=Path, help='JSON marker ID -> measured [x_mm,y_mm]')
    args = parser.parse_args()
    print(save_marker_sheet(args.output_dir / 'aruco_markers_print.png'))
    if args.image:
        image = cv2.imread(str(args.image))
        if image is None:
            raise ValueError(f'Cannot read {args.image}')
        if image.shape[:2] != (520, 1400):
            raise ValueError('Supply a rectified 1400x520 image')
        points = json.loads(args.points.read_text()) if args.points else click_reference_holes(image)
        calibration, lut = save_calibration(image, points, args.output_dir,
                         geometry=json.loads(args.geometry.read_text()) if args.geometry else None,
                         setup_id=args.setup_id,
                         marker_positions_mm=json.loads(args.marker_positions.read_text())
                         if args.marker_positions else None)
        print('Saved calibration.json, hole_lut.json, calibration_overlay.png')
        print(f"Setup: {calibration['setup_id']}; calibration: {calibration['calibration_id']}")
        if not calibration['geometry']['rails']:
            print('Terminal-only calibration: no measured rail coordinates supplied.')


if __name__ == '__main__':
    main()
