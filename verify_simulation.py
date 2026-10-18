"""Verify the real registration/DSU/VF2 pipeline without camera or model weights.

Run: python verify_simulation.py
Uses the active Python environment (activate .venv first). Only detections are
mocked; ArUco, quality gates, snapping, netlist, VF2, rendering and GPIO fallback
are real. Outputs a machine-readable report and PNGs under --output-dir.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from networkx.algorithms.isomorphism import GraphMatcher

from engine.board import BoardSpec, HoleLUT
from engine.schemas import ReferenceCircuit
from pipeline import SherlockOhmsPipeline
from vision_ml.pi_hardware import TrafficLight
from vision_ml.registration import generate_synthetic_aruco_board_image


def golden_reference():
    return ReferenceCircuit(name='Simulation LED circuit',
                            rails={'VCC': 'top_plus', 'GND': 'top_minus'}, parts=[
        {'id': 'W_VCC', 'type': 'wire', 'pins': {'0': 'top_plus_5', '1': 'a5'}},
        {'id': 'R1', 'type': 'resistor', 'pins': {'0': 'a5', '1': 'a10'}},
        {'id': 'LED1', 'type': 'led', 'pins': {'anode': 'd10', 'cathode': 'd15'}},
        {'id': 'W_GND', 'type': 'wire', 'pins': {'0': 'a15', '1': 'top_minus_15'}},
    ])


def detection(lut, part_id, kind, *holes):
    xy = np.asarray([lut.holes[h] for h in holes])
    return {'id': part_id, 'type': kind, 'conf': .99,
            'bbox': (*map(float, xy.min(axis=0)-5), *map(float, xy.max(axis=0)+5)),
            'keypoints': [{'x': float(x), 'y': float(y), 'conf': .99, 'vis': 2} for x, y in xy]}


def golden_detections(lut):
    return [detection(lut, p['id'], p['type'], *p['pins'].values()) for p in golden_reference().parts]


class RecordingTrafficLight(TrafficLight):
    def __init__(self):
        super().__init__(force_fallback=True)
        self.transitions = []

    def set(self, color):
        super().set(color)
        self.transitions.append(self.state.upper())


class MockDetections:
    def __init__(self, values):
        self.values = values
        self.calls = 0

    def detect(self, image):
        assert image.shape == (520, 1400, 3)
        self.calls += 1
        return deepcopy(self.values)


def save_png(path, image):
    assert image is not None and image.shape == (520, 1400, 3)
    assert cv2.imwrite(str(path), image), f'Failed to write {path}'
    decoded = cv2.imread(str(path))
    assert decoded is not None and decoded.shape == image.shape, f'Cannot decode {path}'


def run_verification(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = output_dir / 'simulation_artifacts'
    artifacts.mkdir(exist_ok=True)
    lut = HoleLUT.generate_synthetic(BoardSpec.full_830())
    golden = golden_detections(lut)
    detector = MockDetections(golden)
    light = RecordingTrafficLight()
    pipeline = SherlockOhmsPipeline(golden_reference(), lut=lut, detector=detector, traffic_light=light)
    raw = generate_synthetic_aruco_board_image()
    cases = []

    def inspect_case(name, values, passed, expected_code=None, mode='connectivity'):
        detector.values = values
        start = len(light.transitions)
        with patch('engine.verifier.GraphMatcher', wraps=GraphMatcher) as vf2:
            result = pipeline.inspect_frame(raw, mode=mode)
            assert vf2.call_count > 0, 'NetworkX VF2 was not exercised'
        assert result.quality.passed and not result.aborted
        assert result.passed is passed, (name, result.findings)
        assert light.transitions[start:] == ['YELLOW', 'GREEN' if passed else 'RED']
        codes = sorted({f.code for f in result.findings})
        if expected_code:
            assert expected_code in codes, (name, codes)
        save_png(artifacts / f'{name}.png', result.annotated_image)
        save_png(output_dir / 'debug_last_inspection.png', result.annotated_image)
        cases.append({'case': name, 'check': 'OK', 'circuit_passed': passed,
                      'light': result.light_color, 'findings': codes,
                      'gpio_transitions': light.transitions[start:], 'vf2_calls': vf2.call_count})
        return result

    try:
        result = inspect_case('exact_match', golden, True)
        assert all(v.passed for v in result.verdicts.values())
        # These nets only exist when wire unions/rail bindings were applied.
        nets = [set(members) for members in result.verdict.actual_netlist.values()]
        assert any({'VCC', 'R1.0'} <= net for net in nets)
        assert any({'R1.1', 'LED1.anode'} <= net for net in nets)
        assert any({'GND', 'LED1.cathode'} <= net for net in nets)
        equivalent = [
            detection(lut, 'W_VCC', 'wire', 'top_plus_20', 'a20'),
            detection(lut, 'R1', 'resistor', 'a20', 'a25'),
            detection(lut, 'W_BRIDGE', 'wire', 'e25', 'f25'),
            detection(lut, 'LED1', 'led', 'i25', 'i30'),
            detection(lut, 'W_GND', 'wire', 'j30', 'top_minus_30'),
        ]
        inspect_case('equivalent_connectivity', equivalent, True)
        inspect_case('equivalent_placement', equivalent, False, 'MISPLACED_PIN', mode='placement')
        inspect_case('power_short', golden + [detection(lut, 'W_SHORT', 'wire', 'top_plus_40', 'top_minus_40')],
                     False, 'POWER_SHORT')
        open_circuit = deepcopy(golden)
        open_circuit[2] = detection(lut, 'LED1', 'led', 'd10', 'd16')
        inspect_case('open_connection', open_circuit, False, 'OPEN_CONNECTION')
        for defect in ('brightness', 'sharpness', 'glare', 'markers'):
            image = raw.copy()
            if defect == 'brightness':
                image = (image.astype(float)*.5).astype(np.uint8)
            elif defect == 'sharpness':
                image = cv2.GaussianBlur(image, (13, 13), 2.5)
            elif defect == 'glare':
                image[220:320, 400:900] = 255
            else:
                image[20:110, 20:110] = 180
            start, calls = len(light.transitions), detector.calls
            with patch('engine.verifier.GraphMatcher', wraps=GraphMatcher) as vf2:
                result = pipeline.inspect_frame(image)
                assert vf2.call_count == 0, 'Rejected image reached verification'
            assert result.aborted and not result.passed and result.verdict is None
            assert defect in result.quality.failures
            assert detector.calls == calls, 'Rejected image reached inference'
            assert light.transitions[start:] == ['YELLOW', 'RED']
            if result.annotated_image is not None:
                save_png(artifacts / f'quality_{defect}.png', result.annotated_image)
            cases.append({'case': f'quality_{defect}', 'check': 'OK', 'aborted': True,
                          'quality_failures': list(result.quality.failures),
                          'gpio_transitions': light.transitions[start:]})
    finally:
        pipeline.close()
    report = {'checks_passed': len(cases), 'checks_failed': 0, 'cases': cases,
              'debug_image': str((output_dir / 'debug_last_inspection.png').resolve()),
              'scope': 'Synthetic camera scene and mock detections; real OpenCV, engine, VF2, and GPIO fallback.'}
    (output_dir / 'simulation_report.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent)
    report = run_verification(parser.parse_args().output_dir)
    print('\nSHERLOCK OHMS — SIMULATION AUDIT')
    for case in report['cases']:
        detail = ', '.join(case.get('findings', case.get('quality_failures', []))) or 'matched reference'
        print(f"  OK  {case['case']}: {' -> '.join(case['gpio_transitions'])}; {detail}")
    print(f"\n{report['checks_passed']}/{report['checks_passed']} checks passed; 0 failed")
    print(f"Debug image: {report['debug_image']}")
    print(f"Scope: {report['scope']}")


if __name__ == '__main__':
    main()
