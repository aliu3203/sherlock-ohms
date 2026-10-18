"""One visually audited still photo, not a live accuracy benchmark.

Image 244: a horizontal wire visibly enters physical printed holes d3 and d24.
Four printed corner holes anchor a measured grid (a is the bottom image row).
The source keypoint annotations also agree with these two visible insertions.
"""
import argparse
import json
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
from engine.live_verifier import LiveVerifier
from vision_ml.endpoint_detector import EndpointDetector
from vision_ml.hole_grid import HoleGrid
from vision_ml.live_inspector import draw_overlay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--refiner')
    parser.add_argument('--output', default=str(ROOT/'known_wire_check'))
    args = parser.parse_args()
    image_path = next((ROOT / 'dataset/images/test').glob('ohm_img_244_*'))
    image = cv2.imread(str(image_path))
    anchors = {'a1': [32,427], 'a30': [606,426], 'j30': [605,212], 'j1': [34,213]}
    grid = HoleGrid.from_anchors((640,640), 30, anchors).refined(image)
    detector = EndpointDetector(args.weights, device=args.device, image_size=1280)
    detections = detector.detect(image)
    if args.refiner:
        from vision_ml.endpoint_refiner import EndpointRefiner
        detections = EndpointRefiner(args.refiner,device=args.device).refine(image,detections)
    scenarios = {'matching': ['d3','d24'], 'equivalent_holes': ['a3','e24'],
                 'adjacent_column_mistake': ['d3','d23'], 'wrong_board_half': ['d3','g24']}
    report = {'image': str(image_path), 'weights': str(Path(args.weights).resolve()),
              'device': args.device, 'visually_audited_wire_holes': ['d3','d24'], 'anchors': anchors,
              'refiner': args.refiner,
              'note': 'One still photo, no ArUco tracking test. Candidate outcomes only; never replayed to imply temporal confirmation.',
              'scenarios': {}}
    out = Path(args.output)
    out.mkdir(exist_ok=True)
    grid.save(out / 'hole_grid.json')
    for name, holes in scenarios.items():
        reference = {'parts': [{'id':'W1','type':'wire','pins': dict(zip(('0','1'), holes))}]}
        verdict = LiveVerifier(grid, reference).inspect(detections)
        report['scenarios'][name] = verdict
        cv2.imwrite(str(out / (name + '.jpg')), draw_overlay(image, verdict, grid))
    (out / 'results.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: {'status':v['status'],'candidate_status':v.get('candidate_status'),
                         'pins':[p['pins'] for p in v['parts']]} for k,v in report['scenarios'].items()}, indent=2))


if __name__ == '__main__':
    main()
