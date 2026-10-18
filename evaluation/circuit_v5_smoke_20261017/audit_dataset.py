"""Read-only dataset audit. No model loading, inference, training, or downloads."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
import cv2
import numpy as np

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resistor-endpoints', nargs=2, type=int, choices=range(4),
                        help='Explicit mapping chosen after numbered visual review')
    args = parser.parse_args()
    if args.resistor_endpoints and len(set(args.resistor_endpoints)) != 2:
        parser.error('Resistor endpoints must be distinct')
    args.output.mkdir(parents=True, exist_ok=True)
    overlays = args.output/'numbered_keypoints'
    crops = args.output/'resistor_crops'
    overlays.mkdir(exist_ok=True)
    crops.mkdir(exist_ok=True)
    files = sorted(p for p in args.dataset.rglob('*') if p.is_file())
    before = {str(p.relative_to(args.dataset)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    classes = {0:'breadboard', 1:'resistor', 2:'wire'}
    remap = {'breadboard':None, 'resistor':1, 'wire':0}
    counts, visibility, dimensions = Counter(), Counter(), Counter()
    annotations = []
    for image_path in sorted((args.dataset/'train/images').iterdir()):
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f'Unreadable image: {image_path}')
        h,w = image.shape[:2]
        dimensions[f'{w}x{h}'] += 1
        overlay = image.copy()
        objects = []
        label_path = args.dataset/'train/labels'/f'{image_path.stem}.txt'
        for object_id, line in enumerate(label_path.read_text().splitlines(),1):
            vals = list(map(float, line.split()))
            if len(vals) != 29:
                raise ValueError(f'Expected 5+8*3 fields: {label_path}')
            cls = classes[int(vals[0])]
            counts[cls] += 1
            cx,cy,bw,bh = np.array(vals[1:5])*[w,h,w,h]
            box = [cx-bw/2,cy-bh/2,cx+bw/2,cy+bh/2]
            points = np.array(vals[5:]).reshape(8,3)
            points[:,:2] *= [w,h]
            visibility[f"{cls}:{','.join(str(i) for i,p in enumerate(points) if p[2]>0)}"] += 1
            color = {'breadboard':(200,160,60),'resistor':(0,220,255),'wire':(255,80,200)}[cls]
            cv2.rectangle(overlay,tuple(np.round(box[:2]).astype(int)),tuple(np.round(box[2:]).astype(int)),color,1)
            cv2.putText(overlay,f'{cls} #{object_id}',(max(0,int(box[0])),max(15,int(box[1])-5)),cv2.FONT_HERSHEY_SIMPLEX,.4,color,1)
            for i,(x,y,v) in enumerate(points):
                if v > 0:
                    cv2.circle(overlay,(round(x),round(y)),3,color,-1)
                    cv2.putText(overlay,str(i),(round(x)+4,round(y)-4),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,0),3)
                    cv2.putText(overlay,str(i),(round(x)+4,round(y)-4),cv2.FONT_HERSHEY_SIMPLEX,.45,color,1)
            if cls == 'resistor':
                x0,y0 = np.maximum(0,np.floor(np.minimum(points[points[:,2]>0,:2].min(axis=0),box[:2])-20)).astype(int)
                x1,y1 = np.minimum([w,h],np.ceil(np.maximum(points[points[:,2]>0,:2].max(axis=0),box[2:])+20)).astype(int)
                crop = cv2.resize(image[y0:y1,x0:x1],None,fx=4,fy=4)
                for i,(x,y,v) in enumerate(points):
                    if v > 0:
                        xy=(round((x-x0)*4),round((y-y0)*4))
                        cv2.circle(crop,xy,5,color,2)
                        cv2.putText(crop,str(i),(xy[0]+6,xy[1]-6),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,0,0),4)
                        cv2.putText(crop,str(i),(xy[0]+6,xy[1]-6),cv2.FONT_HERSHEY_SIMPLEX,.8,color,2)
                cv2.imwrite(str(crops/f'{image_path.stem}_{object_id}.png'),crop)
            objects.append(dict(dataset_class=cls,target_class_id=remap[cls],bbox_xyxy=box,
                                numbered_keypoints=points.tolist(),
                                endpoint_indices=([0,1] if cls=='wire' else args.resistor_endpoints if cls=='resistor' else None)))
        cv2.putText(overlay,'GROUND TRUTH ONLY - no model predictions',(8,22),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,0),3)
        cv2.putText(overlay,'GROUND TRUTH ONLY - no model predictions',(8,22),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
        cv2.imwrite(str(overlays/f'{image_path.stem}.png'),overlay)
        annotations.append(dict(image=image_path.name,objects=objects))
    after = {str(p.relative_to(args.dataset)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    assert before == after, 'Dataset changed during audit'
    report=dict(status='BLOCKED_NO_COMPATIBLE_WEIGHTS',scope='smoke-test readiness audit; no inference',
        image_count=len(annotations),object_counts=dict(counts),image_dimensions=dict(dimensions),
        annotated_keypoint_indices=dict(visibility),class_remap_by_name=remap,
        original_dataset_unchanged=True,source_sha256=before,
        resistor_endpoint_mapping=(args.resistor_endpoints or 'UNVERIFIED'),
        metrics={c:dict(precision=None,recall=None,endpoint_error_px=None,missed_objects=None,
                       reason='No compatible trained weights; inference not run') for c in ('wire','resistor')})
    components = []
    for entry in annotations:
        objects = []
        for obj in entry['objects']:
            indices = obj['endpoint_indices']
            if obj['target_class_id'] is not None and indices is not None:
                objects.append(dict(type=obj['dataset_class'], class_id=obj['target_class_id'],
                                    bbox_xyxy=obj['bbox_xyxy'],
                                    endpoints=[obj['numbered_keypoints'][i] for i in indices]))
        components.append(dict(image=entry['image'], components=objects))
    (args.output/'component_ground_truth.json').write_text(json.dumps(components,indent=2)+'\n')
    (args.output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    (args.output/'annotations_numbered.json').write_text(json.dumps(annotations,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'},indent=2))

if __name__ == '__main__':
    main()
