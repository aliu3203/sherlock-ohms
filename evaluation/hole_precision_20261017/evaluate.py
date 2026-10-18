"""Held-out endpoint scoring. Hole agreement below is an annotation proxy,
not independently hand-labeled hole ground truth or electrical testing.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
from vision_ml.endpoint_detector import EndpointDetector
from vision_ml.hole_grid import HoleGrid

SAMPLES = {
    210: [[33,215],[605,215],[606,429],[31,429]],
    230: [[34,215],[605,214],[606,429],[32,429]],
    240: [[34,212],[605,211],[606,427],[32,427]],
    250: [[34,213],[605,213],[606,426],[32,427]],
    260: [[34,213],[605,213],[606,426],[32,427]],
    270: [[34,214],[602,211],[603,425],[32,426]],
    280: [[34,213],[602,211],[602,425],[32,425]],
    330: [[34,208],[606,210],[605,426],[30,426]],
    530: [[34,211],[605,211],[606,426],[32,427]],
}


def overlap(a, b):
    intersect = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])).prod()
    union = (a[2:] - a[:2]).prod() + (b[2:] - b[:2]).prod() - intersect
    return float(intersect / union) if union else 0.


def matching(truth, proposed):
    graph = nx.Graph()
    for i, t in enumerate(truth):
        for j, p in enumerate(proposed):
            score = overlap(np.array(t['bbox']), np.array(p['bbox']))
            if t['type'] == p['type'] and score >= .5:
                graph.add_edge(('t', i), ('p', j), weight=score)
    return [(a[1], b[1]) if a[0] == 't' else (b[1], a[1])
            for a, b in nx.max_weight_matching(graph, maxcardinality=True)]


def load_truth(path, size):
    result = []
    for line in (ROOT / 'dataset/labels/test' / (path.stem + '.txt')).read_text().splitlines():
        values = np.array(list(map(float, line.split())))
        mid, extent = values[1:3] * size, values[3:5] * size
        points = values[5:].reshape(2, 3)
        result.append({'type': ('wire', 'resistor')[int(values[0])],
                       'bbox': np.r_[mid - extent / 2, mid + extent / 2].tolist(),
                       'points': points[:, :2] * size, 'visibility': points[:, 2]})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--refiner', help='Optional experimental local endpoint-refiner weights')
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    detector = EndpointDetector(args.weights, device=args.device, image_size=1280)
    refiner = None
    if args.refiner:
        from vision_ml.endpoint_refiner import EndpointRefiner
        refiner = EndpointRefiner(args.refiner,device=args.device)
    totals, records, hole_audit, times = {}, [], [], []
    for path in sorted((ROOT / 'dataset/images/test').glob('*.jpg')):
        image = cv2.imread(str(path))
        start = time.perf_counter()
        all_predictions = detector.detect(image)
        if refiner is not None:
            all_predictions = refiner.refine(image,all_predictions)
        times.append(1000 * (time.perf_counter() - start))
        pred = [p for p in all_predictions if p['conf'] >= .6 and all(k.get('coarse_conf',k['conf']) >= .6 for k in p['keypoints'])]
        truth = load_truth(path, np.array(image.shape[1::-1]))
        pairs = matching(truth, pred)
        for kind in ('wire', 'resistor'):
            source = path.name.split('_')[0]
            key = source + ':' + kind
            stats = totals.setdefault(key, {'truth': 0, 'predicted': 0, 'matched': 0, 'errors_px': []})
            stats['truth'] += sum(t['type'] == kind for t in truth)
            stats['predicted'] += sum(p['type'] == kind for p in pred)
            for i, j in pairs:
                if truth[i]['type'] != kind:
                    continue
                stats['matched'] += 1
                if min(truth[i]['visibility']) <= 0:
                    continue
                pxy = np.array([[k['x'], k['y']] for k in pred[j]['keypoints']])
                direct = np.linalg.norm(truth[i]['points'] - pxy, axis=1)
                reverse = np.linalg.norm(truth[i]['points'] - pxy[::-1], axis=1)
                stats['errors_px'].extend((direct if direct.sum() <= reverse.sum() else reverse).tolist())
        records.append({'image': str(path), 'detections': all_predictions})
        if source == 'ohm':
            number = int(path.name.split('_')[2])
            if number in SAMPLES:
                audit = {'image': path.name, 'manual_anchors': SAMPLES[number], 'truth_objects': len(truth)}
                grid = HoleGrid.from_anchors((640,640), 30, dict(zip(('a1','a30','j30','j1'), SAMPLES[number])))
                try:
                    grid = grid.refined(image)
                except ValueError as exc:
                    audit['grid_error'] = str(exc)
                    hole_audit.append(audit)
                    continue
                audit['grid_observation'] = {k:v for k,v in grid.observe(image).items() if k != 'matches'}
                audit['eligible_truth'] = audit['exact_pair_agreements'] = 0
                audit['matched_predictions_with_two_accepted_holes'] = audit['wrong_accepted_pairs'] = 0
                by_truth = dict(pairs)
                for i, truth_object in enumerate(truth):
                    reference = [grid.associate(*point) for point in truth_object['points']]
                    if any(p.hole is None for p in reference):
                        continue
                    audit['eligible_truth'] += 1
                    if i not in by_truth:
                        continue
                    proposed = [grid.associate_keypoint(k) for k in pred[by_truth[i]]['keypoints']]
                    if any(p.hole is None for p in proposed):
                        continue
                    audit['matched_predictions_with_two_accepted_holes'] += 1
                    same = sorted(p.hole for p in reference) == sorted(p.hole for p in proposed)
                    audit['exact_pair_agreements'] += int(same)
                    audit['wrong_accepted_pairs'] += int(not same)
                for xy in grid.xy:
                    cv2.circle(image, tuple(map(round, xy)), 2, (255,170,0), 1)
                for t in truth:
                    for xy in t['points']:
                        cv2.circle(image, tuple(map(round,xy)), 6, (255,255,0), 1)
                for p in pred:
                    for k in p['keypoints']:
                        cv2.circle(image, (round(k['x']),round(k['y'])), 3, (220,0,220), -1)
                cv2.putText(image, 'GT cyan | predictions magenta | grid blue', (8,30), cv2.FONT_HERSHEY_SIMPLEX, .55, (255,255,255), 1)
                cv2.imwrite(str(output / path.name), image)
                hole_audit.append(audit)
    for stats in totals.values():
        errors = stats.pop('errors_px')
        stats['precision'] = stats['matched'] / stats['predicted'] if stats['predicted'] else None
        stats['recall'] = stats['matched'] / stats['truth'] if stats['truth'] else None
        stats['missed'] = stats['truth'] - stats['matched']
        stats['false_positive'] = stats['predicted'] - stats['matched']
        stats['endpoint_median_px'] = float(np.median(errors)) if errors else None
        stats['endpoint_p90_px'] = float(np.percentile(errors, 90)) if errors else None
        stats['endpoint_within_5px_fraction'] = float(np.mean(np.array(errors) <= 5)) if errors else None
    report = {'weights': str(Path(args.weights).resolve()), 'sha256': hashlib.sha256(Path(args.weights).read_bytes()).hexdigest(),
              'refiner': args.refiner,
              'images': len(records), 'confidence': .6, 'matching_iou': .5, 'classes_by_source': totals,
              'confidence_policy': 'Box and original keypoint >=.6 for box/error metrics; hole proxy uses full coarse/refined agreement and fallback rules.',
              'device': args.device, 'image_size': 'native long edge capped at 1280',
              'inference_ms_median_excluding_first': float(np.median(times[1:])),
              'inference_ms_p90_excluding_first': float(np.percentile(times[1:],90)),
              'hole_proxy': hole_audit,
              'limits': ['Endpoint errors are conditional on one-to-one IoU-matched detections; misses reported separately.',
                         'Nine hole-audit photos selected by capture number before final model evaluation; correlated board scenes.',
                         'Hole proxy maps provided keypoint labels onto a visually anchored grid; excludes rail/out-of-grid/ambiguous GT.',
                         'This is not independent exact-hole ground truth, live accuracy, or proof of electrical contact.',
                         'Base thresholds fixed before test evaluation. Refiner architecture was added after inspecting base-model test errors; treat refined results as a development audit.']}
    (output / 'metrics.json').write_text(json.dumps(report, indent=2))
    (output / 'predictions.json').write_text(json.dumps(records, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k != 'hole_proxy'}, indent=2))


if __name__ == '__main__':
    main()
