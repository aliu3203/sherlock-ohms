from pathlib import Path
import json,re,hashlib,random,shutil
from collections import Counter
import cv2,numpy as np,yaml
ROOT=Path(__file__).resolve().parent
NEW=Path('/Users/danielxiao/Downloads/sherlock-ohms-datasets/ohm-vision-v2')
OLD=ROOT.parent/'endpoint_training_20261017/dataset'

def main():
 dest=ROOT/'dataset'
 if dest.exists(): raise RuntimeError('Refusing to overwrite dataset')
 files=sorted(NEW.glob('*/images/*.jpg')); n=len(files); par=list(range(n))
 def find(i):
  while par[i]!=i: par[i]=par[par[i]];i=par[i]
  return i
 def union(i,j): par[find(i)]=find(j)
 groups={}; hashes=[]
 for i,p in enumerate(files):
  group=int(re.search(r'img_(\d+)',p.name)[1])//10
  if group in groups: union(i,groups[group])
  groups[group]=i
  gray=cv2.imread(str(p),0);d=cv2.dct(cv2.resize(gray,(32,32)).astype(np.float32))[:8,:8].flatten()[1:];hashes.append(d>np.median(d))
 near=0
 for i in range(n):
  for j in range(i):
   if np.count_nonzero(hashes[i]!=hashes[j])<=5: union(i,j);near+=1
 clusters={}
 for i in range(n): clusters.setdefault(find(i),[]).append(i)
 grouped=list(clusters.values());random.Random(2026).shuffle(grouped)
 assigned={}; counts=Counter()
 for group in grouped:
  split='test' if counts['test']<65 else 'val' if counts['val']<65 else 'train'
  for i in group: assigned[i]=split
  counts[split]+=len(group)
 manifest=[]; stats={s:Counter() for s in ['train','val','test']}; seen={}
 for s in stats:
  for kind in ['images','labels']: (dest/kind/s).mkdir(parents=True)
 def add(p,lab,split,source,remap):
  sha=hashlib.sha256(p.read_bytes()).hexdigest()
  if sha in seen:
   if seen[sha]!=split: raise ValueError('Exact duplicate across splits')
   return
  seen[sha]=split
  rows=[]
  for l in lab.read_text().splitlines():
   a=list(map(float,l.split())); assert len(a)==11
   a[0]=remap[int(a[0])]
   if not np.isfinite(a).all() or not all(0<=x<=1 for x in a[1:5]):raise ValueError(l)
   for k in [5,8]:
    if a[k+2]==0:a[k:k+2]=[0,0]
    elif not (0<=a[k]<=1 and 0<=a[k+1]<=1):raise ValueError(l)
   rows.append(' '.join(f'{v:.9g}' for v in a));stats[split][['wire','resistor'][int(a[0])]]+=1
  name=source+'_'+p.name
  shutil.copy2(p,dest/'images'/split/name)
  (dest/'labels'/split/(Path(name).stem+'.txt')).write_text('\n'.join(rows)+ ('\n' if rows else ''))
  stats[split]['images']+=1
  manifest.append({'file':name,'split':split,'source':source,'original':str(p),'sha256':sha})
 for i,p in enumerate(files): add(p,p.parent.parent/'labels'/(p.stem+'.txt'),assigned[i],'ohm',{0:1,1:0})
 for split in stats:
  for p in sorted((OLD/'images'/split).glob('*.jpg')):add(p,OLD/'labels'/split/(p.stem+'.txt'),split,'circuit',{0:0,1:1})
 data={'path':str(dest),'train':'images/train','val':'images/val','test':'images/test','names':{0:'wire',1:'resistor'},'kpt_shape':[2,3],'flip_idx':[0,1]}
 (ROOT/'data.yaml').write_text(yaml.safe_dump(data,sort_keys=False))
 audit={'stats':stats,'ohm_grouped_counts':counts,'near_duplicate_pairs':near,'group_sizes':[len(x) for x in grouped],'sources':[{'url':'https://universe.roboflow.com/komkrit-workspace/ohm-vision-keypoint-aruco-2/dataset/2','license':'CC BY 4.0','images':541},{'url':'https://universe.roboflow.com/circuit-detection/circuit-end-point-detection/dataset/14','license':'CC BY 4.0','images':306}],'split_policy':'Ohm: groups of 10 consecutive capture IDs and pHash distance <=5 stay together. Circuit: prior session split preserved. Circuit identities are not provided, so scene independence is not guaranteed.','manifest':manifest}
 (ROOT/'dataset_audit.json').write_text(json.dumps(audit,indent=2))
 print(json.dumps({k:v for k,v in audit.items() if k!='manifest'},indent=2))
 for f in NEW.glob('README*'):shutil.copy2(f,ROOT/('ohm_'+f.name))
if __name__=='__main__':main()
