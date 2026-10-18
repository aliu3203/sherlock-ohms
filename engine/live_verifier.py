"""Conservative, hole-aware connectivity comparison for wire/resistor demos.

Public inputs are dictionaries, so Base44 or another UI can consume the result.
The verdict describes observed topology, not measured electrical continuity.
"""
from collections import Counter
from dataclasses import asdict
import math

import networkx as nx
import numpy as np


def _json_safe(value):
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(v) for v in value]
    return value


def _parts(parts, grid):
    result, ids, occupied = [], set(), set()
    for part in parts:
        pid, kind, pins = str(part['id']), part['type'], part['pins']
        if kind not in ('wire', 'resistor') or len(pins) != 2 or pid in ids:
            raise ValueError('Unique IDs and exactly two pins on wires/resistors are required')
        holes = list(pins.values())
        for hole in holes:
            grid.node(hole)
            if hole in occupied:
                raise ValueError(f'Multiple pins occupy {hole}')
            occupied.add(hole)
        ids.add(pid)
        result.append({'id': pid, 'type': kind, 'pins': dict(zip(('0', '1'), holes))})
    return result


def circuit_graph(parts, grid, terminals):
    """Collapse wire-connected strips, preserving physical external terminals.

    Resistor pins are unordered. Wire count and routed intermediate holes do not
    affect equivalence. Floating wire networks remain visible as extra nets.
    """
    conductors = nx.Graph()
    for hole in grid.names:
        conductors.add_node(grid.node(hole))
    for part in parts:
        if part['type'] == 'wire':
            conductors.add_edge(*(grid.node(h) for h in part['pins'].values()))
    roots = {node: min(group) for group in nx.connected_components(conductors) for node in group}
    graph = nx.Graph()
    for name, hole in terminals.items():
        net = ('net', roots[grid.node(hole)])
        graph.add_node(net, label=('net',))
        graph.add_node(('terminal', name), label=('terminal', name))
        graph.add_edge(net, ('terminal', name))
    for part in parts:
        nets = [roots[grid.node(h)] for h in part['pins'].values()]
        for net in nets:
            graph.add_node(('net', net), label=('net',))
        if part['type'] == 'wire':
            continue
        pid = part['id']
        graph.add_node(('part', pid), label=('part', part['type']))
        for pin, net in enumerate(nets):
            graph.add_node(('pin', pid, pin), label=('pin',))
            graph.add_edge(('part', pid), ('pin', pid, pin))
            graph.add_edge(('pin', pid, pin), ('net', net))
    return graph, roots


class LiveVerifier:
    def __init__(self, grid, reference, *, stable_frames=3):
        if not isinstance(stable_frames, int) or stable_frames < 2:
            raise ValueError('At least two consecutive frames are required')
        self.grid, self.stable_frames = grid, stable_frames
        self.reference = _parts(reference['parts'], grid)
        if not self.reference:
            raise ValueError('An empty reference cannot establish correct wiring')
        if reference.get('rails'):
            raise ValueError('Use terminals: {VCC: measured_hole, GND: measured_hole}')
        self.terminals = dict(reference.get('terminals', {}))
        for hole in self.terminals.values():
            grid.node(hole)
        self.wire_only = all(p['type'] == 'wire' for p in self.reference)
        if self.wire_only and not self.terminals:
            # A wire-only graph has no component identities. Anchor all reference
            # endpoint strips so an arbitrary wire elsewhere cannot pass.
            self.terminals = {grid.node(h): h for p in self.reference for h in p['pins'].values()}
        self.expected, roots = circuit_graph(self.reference, grid, self.terminals)
        if self._shorts(self.reference, roots):
            raise ValueError('Reference contains a shorted resistor or VCC/GND short')
        self._last = None
        self._count = 0

    def _shorts(self, parts, roots):
        findings = []
        if {'VCC', 'GND'} <= self.terminals.keys():
            if roots[self.grid.node(self.terminals['VCC'])] == roots[self.grid.node(self.terminals['GND'])]:
                findings.append({'code': 'POWER_SHORT', 'message': 'Observed wires connect VCC to GND.'})
        for part in parts:
            if part['type'] == 'resistor' and len({roots[self.grid.node(h)] for h in part['pins'].values()}) == 1:
                findings.append({'code': 'SHORTED_RESISTOR', 'part_id': part['id'],
                                 'message': 'Both resistor legs share an electrical net.'})
        return findings

    def uncertain(self, reason, *, detections=None, parts=None, findings=None):
        self._last, self._count = None, 0
        return {'status': 'UNCERTAIN', 'light_color': 'AMBER', 'passed': False,
                'stable_frames': 0, 'required_stable_frames': self.stable_frames,
                'message': reason, 'detections': _json_safe(detections or []), 'parts': _json_safe(parts or []),
                'findings': findings or [], 'scope': 'visible_wire_resistor_topology'}

    def inspect(self, detections, *, quality_ok=True, grid_ok=True):
        if not quality_ok or not grid_ok:
            return self.uncertain('Board tracking, image quality, or hole alignment is not reliable.')
        parts, issues, occupied, ids = [], [], set(), set()
        for det in detections:
            try:
                kind, pid = det['type'], str(det['id'])
                if kind not in ('wire', 'resistor') or pid in ids or len(det['keypoints']) != 2:
                    raise ValueError('Unsupported or malformed detection')
                ids.add(pid)
                conf = float(det['conf'])
                if not np.isfinite(conf) or not 0 <= conf <= 1:
                    raise ValueError('Invalid component confidence')
                if det.get('bbox') is not None and (len(det['bbox']) != 4 or not np.isfinite(det['bbox']).all()):
                    raise ValueError('Invalid component bounding box')
                matches = [self.grid.associate_keypoint(k) for k in det['keypoints']]
                part = {'id': pid, 'type': kind, 'pins': {str(i): m.hole for i, m in enumerate(matches)},
                        'endpoints': [asdict(m) for m in matches], 'bbox': det.get('bbox'), 'conf': conf}
                parts.append(part)
                if conf < .6 or any(m.hole is None for m in matches):
                    issues.append({'code': 'UNCERTAIN_ENDPOINT', 'part_id': pid,
                                   'message': 'Component or hole assignment is uncertain.'})
                else:
                    for m in matches:
                        if m.hole in occupied:
                            issues.append({'code': 'OVERLAPPING_PINS', 'part_id': pid,
                                           'message': f'Multiple predicted pins occupy {m.hole}.'})
                        occupied.add(m.hole)
            except (ValueError, KeyError, TypeError) as exc:
                issues.append({'code': 'INVALID_DETECTION', 'message': str(exc)})
        if issues:
            return self.uncertain('Some endpoints need a clearer view.', detections=detections, parts=parts, findings=issues)
        if not parts:
            return self.uncertain('No supported components detected; this is not a pass.')
        graph, roots = circuit_graph(parts, self.grid, self.terminals)
        findings = self._shorts(parts, roots)
        ref_counts = Counter(p['type'] for p in self.reference if p['type'] != 'wire')
        counts = Counter(p['type'] for p in parts if p['type'] != 'wire')
        equivalent = nx.is_isomorphic(self.expected, graph, node_match=lambda a, b: a['label'] == b['label'])
        if findings:
            candidate, message = 'INCORRECT', 'A short is visible in the inferred wiring.'
        elif equivalent:
            candidate, message = 'CORRECT', 'Observed connectivity matches the reference.'
        elif any(counts[k] < n for k, n in ref_counts.items()):
            candidate, message = 'INCOMPLETE', 'Expected components are missing or undetected.'
        else:
            candidate, message = 'INCORRECT', 'Observed connectivity does not match the reference.'
            findings.append({'code': 'CONNECTIVITY_MISMATCH', 'message': message})
        # IDs may change each frame. Require the same physical endpoints and
        # outcome, so jitter across neighboring holes resets the settling period.
        signature = (candidate, tuple(sorted((p['type'], tuple(sorted(p['pins'].values()))) for p in parts)))
        self._count = self._count + 1 if signature == self._last else 1
        self._last = signature
        status = candidate if self._count >= self.stable_frames else 'UNCERTAIN'
        return {'status': status, 'candidate_status': candidate,
                'light_color': {'CORRECT': 'GREEN', 'INCORRECT': 'RED'}.get(status, 'AMBER'),
                'passed': status == 'CORRECT', 'stable_frames': self._count,
                'required_stable_frames': self.stable_frames,
                'message': message if status != 'UNCERTAIN' else 'Waiting for consistent consecutive frames.',
                'detections': detections, 'parts': parts, 'findings': findings,
                'scope': 'visible_wire_resistor_topology'}
