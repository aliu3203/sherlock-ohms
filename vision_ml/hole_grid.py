"""Measured hole-grid fitting and conservative endpoint association.

No neural model is needed to detect the dark holes. Four named terminal
anchors define orientation/numbering; real rail centers are supplied separately.
All coordinates are in the current (normally ArUco-rectified) image frame.
"""
from dataclasses import dataclass, replace
from pathlib import Path
import json
import re

import cv2
import numpy as np


@dataclass(frozen=True)
class HoleMatch:
    hole: str | None
    nearest_hole: str
    distance_px: float
    pitch_px: float
    reason: str | None


class HoleGrid:
    def __init__(self, holes, *, image_size, columns, rail_segments=None):
        self.image_size = tuple(map(int, image_size))
        self.columns = int(columns)
        self.holes = {str(k): tuple(map(float, v)) for k, v in holes.items()}
        self.rail_segments = rail_segments or {}
        if len(self.image_size) != 2 or len(self.holes) < 4 or self.columns < 2 or min(self.image_size) < 2:
            raise ValueError('A measured grid and image size are required')
        self.names = list(self.holes)
        self.xy = np.array(list(self.holes.values()), dtype=float)
        if self.xy.shape != (len(self.holes), 2) or not np.isfinite(self.xy).all():
            raise ValueError('Invalid hole coordinates')
        if (self.xy < 0).any() or (self.xy >= self.image_size).any():
            raise ValueError('Hole coordinates outside the calibration image')
        for name in self.names:
            if re.fullmatch(r'[a-j][1-9]\d*', name):
                if int(name[1:]) > self.columns:
                    raise ValueError('Terminal hole exceeds board width')
            elif name not in {h for group in self.rail_segments.values() for h in group}:
                raise ValueError(f'Rail hole needs an explicit electrical segment: {name}')
        rail_members = [h for group in self.rail_segments.values() for h in group]
        if (len(rail_members) != len(set(rail_members)) or any(h not in self.holes for h in rail_members)
                or any(re.fullmatch(r'[a-j][1-9]\d*', h) for h in rail_members)):
            raise ValueError('Rail segments must be disjoint known holes')
        d = np.linalg.norm(self.xy[:, None] - self.xy[None], axis=2)
        np.fill_diagonal(d, np.inf)
        self.pitches = d.min(axis=1)
        if self.pitches.min() < 2:
            raise ValueError('Duplicate holes or insufficient image resolution')
        self.pitch = float(np.median(self.pitches))

    @classmethod
    def from_anchors(cls, image_size, columns, anchors, *, rail_holes=None, rail_segments=None):
        names = ['a1', f'a{columns}', f'j{columns}', 'j1']
        if set(anchors) != set(names):
            raise ValueError(f'Provide four named hole centers: {names}')
        source = np.float32([[0, 0], [columns - 1, 0], [columns - 1, 11], [0, 11]])
        target = np.float32([anchors[n] for n in names])
        if not np.isfinite(target).all() or not cv2.isContourConvex(target) or abs(cv2.contourArea(target)) < 10:
            raise ValueError('Anchors must form a nondegenerate quadrilateral')
        transform = cv2.getPerspectiveTransform(source, target)
        logical = np.float32([[x, y] for y in (0, 1, 2, 3, 4, 7, 8, 9, 10, 11) for x in range(columns)])
        coords = cv2.perspectiveTransform(logical[None], transform)[0]
        holes = {f'{r}{c}': p.tolist() for r, row in zip('abcdefghij', coords.reshape(10, columns, 2))
                 for c, p in enumerate(row, 1)}
        holes.update(rail_holes or {})
        return cls(holes, image_size=image_size, columns=columns, rail_segments=rail_segments)

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))

    def save(self, path):
        Path(path).write_text(json.dumps({'holes': self.holes, 'image_size': self.image_size,
                                        'columns': self.columns, 'rail_segments': self.rail_segments}, indent=2))

    def node(self, hole):
        if hole not in self.holes:
            raise ValueError(f'Unknown hole {hole}')
        if re.fullmatch(r'[a-j][1-9]\d*', hole):
            return f'{"left" if hole[0] in "abcde" else "right"}:{hole[1:]}'
        return next(f'rail:{segment}' for segment, members in self.rail_segments.items() if hole in members)

    def associate(self, x, y, *, confidence=1., max_fraction=.25, margin_fraction=.25):
        if not 0 < max_fraction < .5 or not 0 < margin_fraction <= 1:
            raise ValueError('Association thresholds must preserve separation between holes')
        if not np.isfinite([x, y, confidence]).all() or not 0 <= confidence <= 1:
            raise ValueError('Invalid endpoint')
        distances = np.linalg.norm(self.xy - [x, y], axis=1)
        first, second = np.argsort(distances)[:2]
        pitch = float(self.pitches[first])
        reason = ('low_confidence' if confidence < .6 else
                  'too_far_from_hole' if distances[first] > max_fraction * pitch else
                  'ambiguous_neighbor' if distances[second] - distances[first] < margin_fraction * pitch else None)
        return HoleMatch(None if reason else self.names[first], self.names[first],
                         float(distances[first]), pitch, reason)

    def observe(self, image):
        """Find visible dark holes near expected positions without moving the map."""
        if image.shape[1::-1] != self.image_size:
            raise ValueError('Image and calibration sizes differ')
        candidates = detect_holes(image, self.pitch)
        if not len(candidates):
            return {'passed': False, 'matched': 0, 'coverage': 0., 'median_error_px': None, 'matches': []}
        distances = np.linalg.norm(self.xy[:, None] - candidates[None], axis=2)
        nearest = distances.argmin(axis=1)
        matches = []
        used = set()
        for i in np.argsort(distances.min(axis=1)):
            j = int(nearest[i]); dist = float(distances[i, j])
            if j in used or dist > .30 * self.pitches[i]:
                continue
            used.add(j); matches.append((int(i), j, dist))
        coverage = len(matches) / len(self.xy)
        median = float(np.median([m[2] for m in matches])) if matches else None
        # Both board halves must contribute; a locally convincing pattern is insufficient.
        terminals = [self.names[i] for i, _, _ in matches if re.fullmatch(r'[a-j]\d+', self.names[i])]
        halves = {name[0] in 'abcde' for name in terminals}
        columns = [int(name[1:]) for name in terminals]
        span = (max(columns) - min(columns)) / (self.columns - 1) if columns else 0.
        # A periodic grid can appear aligned after a whole-pitch shift. Require
        # observations at its physical boundaries, not just its repeating middle.
        edges = {'first_column': sum(c == 1 for c in columns),
                 'last_column': sum(c == self.columns for c in columns),
                 'row_a': sum(n[0] == 'a' for n in terminals),
                 'row_j': sum(n[0] == 'j' for n in terminals)}
        edges_ok = (min(edges['first_column'], edges['last_column']) >= 3 and
                    min(edges['row_a'], edges['row_j']) >= max(3, self.columns * .15))
        return {'passed': bool(coverage >= .20 and len(halves) == 2 and span >= .7 and
                               edges_ok and median < .16 * self.pitch),
                'matched': len(matches), 'coverage': coverage, 'median_error_px': median,
                'column_span_fraction': span, 'boundary_holes': edges,
                'matches': [(self.names[i], candidates[j].tolist()) for i, j, _ in matches]}

    def associate_keypoint(self, keypoint):
        """A patch refiner may improve precision, but may not select a new hole.

        If patch evidence is uninformative, retain only an original endpoint
        that already passes the stricter baseline rule. Conflicting confident
        proposals remain uncertain; the target circuit is never consulted.
        """
        point = self.associate(keypoint['x'], keypoint['y'], confidence=keypoint['conf'])
        if 'coarse_x' not in keypoint or 'coarse_y' not in keypoint:
            return point
        original = self.associate(keypoint['coarse_x'], keypoint['coarse_y'],
                                  confidence=keypoint.get('coarse_conf', 0))
        if keypoint['conf'] < .6:
            return original if original.hole else point
        supporting = self.associate(keypoint['coarse_x'], keypoint['coarse_y'],
                                    confidence=keypoint.get('coarse_conf', 0), max_fraction=.49)
        if point.hole and (supporting.hole is None or supporting.hole != point.hole):
            return replace(point, hole=None, reason='refinement_disagrees_with_coarse_hole')
        return point

    def refined(self, image):
        """Refine an empty-board calibration, rejecting large or unsupported warps."""
        obs = self.observe(image)
        if not obs['passed']:
            raise ValueError('Insufficient visible holes to verify calibration')
        src = np.float32([self.holes[n] for n, _ in obs['matches']])
        dst = np.float32([p for _, p in obs['matches']])
        h, inliers = cv2.findHomography(src, dst, cv2.RANSAC, self.pitch * .12)
        if h is None or inliers.mean() < .7:
            raise ValueError('Hole grid is inconsistent')
        updated = cv2.perspectiveTransform(self.xy.astype(np.float32)[None], h)[0]
        if np.max(np.linalg.norm(updated - self.xy, axis=1)) > .3 * self.pitch:
            raise ValueError('Refinement moved too far; check the reference holes')
        return HoleGrid(dict(zip(self.names, updated)), image_size=self.image_size,
                        columns=self.columns, rail_segments=self.rail_segments)


def detect_holes(image, pitch):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    block = max(7, int(pitch * 1.8) | 1)
    mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, block, 12)
    count, _, stats, centers = cv2.connectedComponentsWithStats(mask)
    result = []
    for i in range(1, count):
        _, _, w, h, area = stats[i]
        if (.012 * pitch ** 2 <= area <= .35 * pitch ** 2 and
                .12 * pitch <= min(w, h) <= max(w, h) <= .8 * pitch and
                max(w, h) / min(w, h) <= 2 and area / (w * h) >= .35):
            result.append(centers[i])
    return np.array(result, dtype=float).reshape(-1, 2)
