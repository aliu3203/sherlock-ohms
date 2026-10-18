"""Prepare an isolated two-class endpoint experiment; never modify sources."""
from pathlib import Path
from collections import Counter
import hashlib, json, re, shutil
import cv2
import numpy as np
import yaml
ROOT = Path(__file__).resolve().parent
SOURCE = Path('/Users/danielxiao/Downloads/sherlock-ohms-datasets/circuit-endpoints-v14')
DEST = ROOT / 'dataset'

def group(name):
    s=name.split('.rf.')[0]
    if s.startswith('IMG_'):
        m=re.match(r'IMG_(\d+)',s)
        return 'IMG_'+str(int(m[1])//100)
    if s.startswith('scanned_result'): return 'scanned'
    if s.startswith('transformed'): return 'transformed'
    if s.startswith('frame'): return 'frames'
    if s.startswith('KakaoTalk'): return 'KakaoTalk'
    return s[:8]

def main():
    if DEST.exists(): raise RuntimeError('Refusing to overwrite prepared dataset')
    files=sorted(SOURCE.glob('*/images/*.jpg'))
    parents=list(range(len(files)))
    def find(i):
        while parents[i]!=i:
            parents[i]=parents[parents[i]]; i=parents[i]
        return i
    def union(i,j): parents[find(i)]=find(j)
    hashes=[]; groups={}
    for i,p in enumerate(files):
        g=group(p.name)
        if g in groups: union(i,groups[g])
        groups[g]=i
        gray=cv2.imread(str(p),0)
        dct=cv2.dct(cv2.resize(gray,(32,32)).astype(np.float32))[:8,:8].flatten()[1:]
        hashes.append(dct>np.median(dct))
    near=0
    for i in range(len(files)):
        for j in range(i):
            if np.count_nonzero(hashes[i]!=hashes[j])<=6:
                union(i,j); near+=1
    clusters={}
    for i,p in enumerate(files): clusters.setdefault(find(i),[]).append(p)
    splits={}; stats={}; manifest=[]
    # Hold out a complete photo session, and a complete frame sequence.
    for paths in clusters.values():
        gs={group(p.name) for p in paths}
        split='test' if 'frames' in gs else 'val' if '20241005' in gs else 'train'
        for p in paths: splits[p]=split
    for split in ['train','val','test']:
        (DEST/'images'/split).mkdir(parents=True)
        (DEST/'labels'/split).mkdir(parents=True)
        stats[split]={'images':0,'wire':0,'resistor':0}
    for p,split in splits.items():
        lines=[]
        for line in (p.parent.parent/'labels'/f'{p.stem}.txt').read_text().splitlines():
            v=list(map(float,line.split()))
            if int(v[0]) not in (2,3): continue
            assert len(v)==11 and all(np.isfinite(v))
            v[0]={3:0,2:1}[int(v[0])]
            assert all(0<=x<=1 for x in v[1:5])
            for k in (5,8):
                assert v[k+2] in (0,1,2)
                if v[k+2]==0: v[k:k+2]=[0,0]
                else: assert 0<=v[k]<=1 and 0<=v[k+1]<=1
            lines.append(' '.join(f'{x:.8g}' for x in v))
            stats[split][['wire','resistor'][int(v[0])]]+=1
        shutil.copy2(p,DEST/'images'/split/p.name)
        (DEST/'labels'/split/f'{p.stem}.txt').write_text('\n'.join(lines)+'\n' if lines else '')
        stats[split]['images']+=1
        manifest.append({'source':str(p),'split':split,'group':group(p.name),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    assert all(stats[s]['wire'] and stats[s]['resistor'] for s in stats)
    config={'path':str(DEST),'train':'images/train','val':'images/val','test':'images/test','kpt_shape':[2,3],'flip_idx':[0,1],'names':{0:'wire',1:'resistor'}}
    (ROOT/'data.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    audit={'source_url':'https://universe.roboflow.com/circuit-detection/circuit-end-point-detection/dataset/14','license':'CC BY 4.0','stats':stats,'near_duplicate_pairs_grouped':near,'groups':[[group(p.name) for p in ps][:1]+[len(ps)] for ps in clusters.values()],'mapping':{'wire':0,'register':1},'notes':['register visually verified as physical resistor in sample overlays','Two insertion keypoints retained; missing visibility remains 0','Capacitor and inductor excluded; no LED training','Session/filename grouping and perceptual hash reduce leakage but cannot prove circuit-disjointness'],'manifest':manifest}
    (ROOT/'audit.json').write_text(json.dumps(audit,indent=2))
    for f in SOURCE.glob('README*'): shutil.copy2(f,ROOT/f.name)
    print(json.dumps({k:v for k,v in audit.items() if k!='manifest'},indent=2))
if __name__=='__main__': main()
