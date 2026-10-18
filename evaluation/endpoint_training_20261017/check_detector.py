"""Smoke-test saved weights through the application's real detector."""
from pathlib import Path
import argparse,json,sys
import cv2
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[1]))
from vision_ml.detector import BreadboardPoseDetector

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--image',type=Path)
    ap.add_argument('--weights',type=Path,default=ROOT/'runs/wire_resistor/weights/best.pt')
    args=ap.parse_args()
    detector=BreadboardPoseDetector(args.weights,device='cpu')
    assert not detector.mock
    paths=[args.image] if args.image else sorted((ROOT/'dataset/images/test').glob('*.jpg'))
    out=ROOT/'application_predictions'; out.mkdir(exist_ok=True)
    results=[]
    for p in paths:
        im=cv2.imread(str(p)); detections=detector.detect(im)
        for d in detections:
            x1,y1,x2,y2=map(int,d['bbox'])
            cv2.rectangle(im,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(im,f"{d['type']} {d['conf']:.2f}",(x1,max(12,y1)),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,0),1)
            for k in d['keypoints']: cv2.circle(im,(round(k['x']),round(k['y'])),5,(0,0,255),-1)
        cv2.imwrite(str(out/p.name),im)
        results.append({'image':str(p),'detections':detections})
    (out/'predictions.json').write_text(json.dumps(results,indent=2))
    print(json.dumps({'mock':detector.mock,'images':len(results),'detections':sum(len(r['detections']) for r in results),'output':str(out)},indent=2))
if __name__=='__main__': main()
