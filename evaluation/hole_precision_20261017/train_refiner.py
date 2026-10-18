"""Train a small local heatmap refiner using train/val keypoints only."""
from pathlib import Path
import json,sys,time,random
import cv2,numpy as np,torch
from torch.utils.data import Dataset,DataLoader
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[1]))
from vision_ml.endpoint_refiner import PatchNet,patch_input,XX,YY


class Crops(Dataset):
    def __init__(self,split):
        self.train=split=='train';self.records=[];self.images={}
        for path in sorted((ROOT/'dataset/images'/split).glob('*.jpg')):
            im=cv2.imread(str(path));self.images[str(path)]=im
            size=np.array(im.shape[1::-1])
            for line in (ROOT/'dataset/labels'/split/(path.stem+'.txt')).read_text().splitlines():
                v=np.array(list(map(float,line.split())));points=v[5:].reshape(2,3)
                if min(points[:,2])<=0:continue
                xy=points[:,:2]*size
                for j in (0,1):self.records.append((str(path),xy[j],xy[1-j]))
    def __len__(self):return len(self.records)
    def __getitem__(self,i):
        rng=np.random.default_rng(None if self.train else 10000+i)
        name,point,other=self.records[i]
        center=point+rng.uniform(-16,16,2)
        sample,origin=patch_input(self.images[name],center,other)
        if self.train:
            sample[:3]=np.clip(sample[:3]*rng.uniform(.8,1.2)+rng.uniform(-.05,.05),0,1)
        target=point-origin
        heat=np.exp(-((XX-target[0])**2+(YY-target[1])**2)/(2*1.5**2)).astype(np.float32)
        heat/=heat.sum()
        return torch.from_numpy(sample),torch.from_numpy(heat.flatten()),torch.tensor(target,dtype=torch.float32)


def main():
    torch.manual_seed(2026);np.random.seed(2026);random.seed(2026)
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    out=ROOT/'refiner';out.mkdir(exist_ok=False)
    model=PatchNet().to(device)
    train=Crops('train');val=Crops('val')
    loaders={s:DataLoader(d,batch_size=64,shuffle=s=='train',num_workers=0) for s,d in [('train',train),('val',val)]}
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,40,eta_min=.00005)
    best=float('inf');stale=0;history=[];start=time.time()
    print('PARAMETERS',sum(p.numel() for p in model.parameters()),'TRAIN',len(train),'VAL',len(val),flush=True)
    for epoch in range(1,41):
        row={'epoch':epoch}
        for phase in ('train','val'):
            model.train(phase=='train');losses=[];errors=[]
            for x,target,xy in loaders[phase]:
                x,target=x.to(device),target.to(device)
                with torch.set_grad_enabled(phase=='train'):
                    logits=model(x)
                    loss=-(target*logits.log_softmax(1)).sum(1).mean()
                    if phase=='train':
                        optimizer.zero_grad();loss.backward();optimizer.step()
                losses.append(float(loss.detach().cpu()))
                if phase=='val':
                    index=logits.detach().argmax(1).cpu()
                    predicted=torch.stack((index%64,index//64),1)
                    errors.extend(torch.linalg.vector_norm(predicted-xy,dim=1).tolist())
            row[phase+'_loss']=float(np.mean(losses))
            if errors:
                row['val_median_px']=float(np.median(errors));row['val_mean_px']=float(np.mean(errors))
                row['val_within_3px']=float(np.mean(np.array(errors)<=3))
        scheduler.step();history.append(row);print(json.dumps(row),flush=True)
        (out/'history.json').write_text(json.dumps(history,indent=2))
        if row['val_mean_px']<best:
            best=row['val_mean_px'];stale=0
            torch.save({'format':'endpoint-patch-v1','state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'epoch':epoch},out/'best.pt')
        else:stale+=1
        if stale>=10:break
    (out/'training_result.json').write_text(json.dumps({'parameters':sum(p.numel() for p in model.parameters()),'seconds':time.time()-start,
        'best_val_mean_px':best,'epochs':len(history),'note':'Validation uses fixed GT-jittered crops, not end-to-end predicted crops.'},indent=2))


if __name__=='__main__':main()
