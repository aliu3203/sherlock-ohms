"""Score application predictions without confusing misses with endpoint accuracy."""
from pathlib import Path
import json

import cv2
import numpy as np
import networkx as nx

ROOT = Path(__file__).resolve().parent


def iou(a, b):
    low = np.maximum(a[:2], b[:2])
    high = np.minimum(a[2:], b[2:])
    intersection = np.maximum(high - low, 0).prod()
    union = np.maximum(a[2:] - a[:2], 0).prod() + np.maximum(b[2:] - b[:2], 0).prod() - intersection
    return float(intersection / union) if union > 0 else 0.0


def main():
    records = json.loads((ROOT / 'application_predictions/predictions.json').read_text())
    totals = {c: {'true_positive': 0, 'false_positive': 0, 'missed_objects': 0,
                  'endpoint_errors_px': [], 'matched_pairs_scored': 0} for c in ('wire', 'resistor')}
    out = ROOT / 'application_evaluation'
    out.mkdir(exist_ok=True)
    for record in records:
        p = Path(record['image'])
        image = cv2.imread(str(p))
        height, width = image.shape[:2]
        scale = np.array([width, height])
        truth = []
        for line in (ROOT / 'dataset/labels/test' / (p.stem + '.txt')).read_text().splitlines():
            v = np.array(list(map(float, line.split())))
            center, extent = v[1:3] * scale, v[3:5] * scale
            points = v[5:].reshape(2, 3).copy()
            points[:, :2] *= scale
            truth.append({'type': ('wire', 'resistor')[int(v[0])],
                          'box': np.r_[center - extent / 2, center + extent / 2], 'points': points})
        for cls, stats in totals.items():
            gt = [t for t in truth if t['type'] == cls]
            pred = [d for d in record['detections'] if d['type'] == cls]
            overlaps = np.array([[iou(t['box'], np.array(d['bbox'])) for d in pred] for t in gt])
            matched = []
            if gt and pred:
                graph = nx.Graph()
                for i, j in zip(*np.where(overlaps >= .5)):
                    graph.add_edge(('gt', int(i)), ('pred', int(j)), weight=float(overlaps[i, j]))
                pairs = nx.max_weight_matching(graph, maxcardinality=True)
                matched = [(a[1], b[1]) if a[0] == 'gt' else (b[1], a[1]) for a, b in pairs]
            stats['true_positive'] += len(matched)
            stats['false_positive'] += len(pred) - len(matched)
            stats['missed_objects'] += len(gt) - len(matched)
            for i, j in matched:
                points = gt[i]['points']
                if np.any(points[:, 2] == 0):
                    continue
                proposed = np.array([[k['x'], k['y']] for k in pred[j]['keypoints']])
                direct = np.linalg.norm(points[:, :2] - proposed, axis=1)
                reverse = np.linalg.norm(points[:, :2] - proposed[::-1], axis=1)
                error = direct if direct.sum() <= reverse.sum() else reverse
                stats['endpoint_errors_px'].extend(error.tolist())
                stats['matched_pairs_scored'] += 1
        for t in truth:
            x1, y1, x2, y2 = map(round, t['box'])
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 0), 2)
            for x, y, vis in t['points']:
                if vis:
                    cv2.circle(image, (round(x), round(y)), 7, (255, 255, 0), 2)
        for d in record['detections']:
            x1, y1, x2, y2 = map(round, d['bbox'])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{d['type']} {d['conf']:.2f}", (x1, max(20, y1 - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 180, 0), 2)
            for k in d['keypoints']:
                cv2.circle(image, (round(k['x']), round(k['y'])), 4, (0, 0, 255), -1)
        cv2.rectangle(image, (0, 0), (width, 32), (255, 255, 255), -1)
        cv2.putText(image, 'Ground truth: cyan | Predictions: green boxes, red endpoints', (8, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 0), 1)
        cv2.imwrite(str(out / p.name), image)
    for stats in totals.values():
        tp, fp, missed = (stats[k] for k in ('true_positive', 'false_positive', 'missed_objects'))
        stats['precision'] = tp / (tp + fp) if tp + fp else None
        stats['recall'] = tp / (tp + missed) if tp + missed else None
        errors = stats.pop('endpoint_errors_px')
        stats['endpoint_mean_px'] = float(np.mean(errors)) if errors else None
        stats['endpoint_median_px'] = float(np.median(errors)) if errors else None
        stats['endpoint_p90_px'] = float(np.percentile(errors, 90)) if errors else None
        stats['endpoints_within_10px_fraction'] = float(np.mean(np.array(errors) <= 10)) if errors else None
    report = {'pipeline': 'Existing BreadboardPoseDetector, mock=False', 'image_count': len(records),
              'box_confidence': .4, 'keypoint_confidence': .5, 'matching_iou': .5,
              'matching': 'One-to-one same-class boxes; endpoint order ignored; both GT points must be labeled',
              'limitation': 'Endpoint errors are conditional on matched detections; misses are reported separately. Test frames are correlated.',
              'classes': totals}
    (out / 'metrics.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
