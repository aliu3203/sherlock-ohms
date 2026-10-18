"""Optional small image-patch endpoint refiner; never changes a circuit verdict."""
from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class PatchNet(nn.Module):
    def __init__(self):
        super().__init__()
        def block(a,b):
            return nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.ReLU(),
                                 nn.Conv2d(b,b,3,padding=1),nn.ReLU())
        self.first = block(6,16)
        self.second = block(16,32)
        self.third = block(32,64)
        self.up_second = block(96,32)
        self.up_first = block(48,16)
        self.head = nn.Conv2d(16,1,1)

    def forward(self,x):
        a=self.first(x)
        b=self.second(F.max_pool2d(a,2))
        c=self.third(F.max_pool2d(b,2))
        b=self.up_second(torch.cat((F.interpolate(c,scale_factor=2,mode='bilinear',align_corners=False),b),1))
        a=self.up_first(torch.cat((F.interpolate(b,scale_factor=2,mode='bilinear',align_corners=False),a),1))
        return self.head(a).flatten(1)


YY,XX=np.mgrid[:64,:64].astype(np.float32)
PRIOR=np.exp(-((XX-31.5)**2+(YY-31.5)**2)/(2*12**2))


def patch_input(image,center,other):
    # Sampling has an explicit floating-point origin, avoiding integer crop bias.
    center=np.asarray(center,dtype=np.float32)
    origin=center-31.5
    patch=cv2.remap(image, XX+origin[0], YY+origin[1], cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,borderValue=(114,114,114))
    rgb=patch[:,:,::-1].astype(np.float32)/255
    direction=np.asarray(other,dtype=np.float32)-center
    direction/=max(float(np.linalg.norm(direction)),1.)
    result=np.concatenate((rgb,PRIOR[:,:,None],
                           np.full((64,64,1),direction[0],np.float32),
                           np.full((64,64,1),direction[1],np.float32)),axis=2)
    return result.transpose(2,0,1),origin


class EndpointRefiner:
    def __init__(self,weights,device='cpu'):
        checkpoint=torch.load(Path(weights),map_location='cpu',weights_only=True)
        if checkpoint.get('format')!='endpoint-patch-v1':
            raise ValueError('Unsupported endpoint refiner checkpoint')
        self.model=PatchNet()
        self.model.load_state_dict(checkpoint['state_dict'])
        self.device=device
        self.model.to(device).eval()

    def refine(self,image,detections):
        import copy
        output=copy.deepcopy(detections)
        patches,origins,locations=[],[],[]
        for i,d in enumerate(output):
            if len(d['keypoints'])!=2 or d['conf']<.6:
                continue
            for j,k in enumerate(d['keypoints']):
                if k['conf']<.6:
                    continue
                other=d['keypoints'][1-j]
                sample,origin=patch_input(image,[k['x'],k['y']],[other['x'],other['y']])
                patches.append(sample);origins.append(origin);locations.append((i,j))
        if not patches:
            return output
        with torch.inference_mode():
            probabilities=self.model(torch.from_numpy(np.stack(patches)).to(self.device)).softmax(1).cpu().numpy().reshape(-1,64,64)
        for heat,origin,(i,j) in zip(probabilities,origins,locations):
            y,x=np.unravel_index(np.argmax(heat),heat.shape)
            crop=heat[max(0,y-3):min(64,y+4),max(0,x-3):min(64,x+4)]
            mass=float(crop.sum())
            # Local weighted centroid within the selected mode, not a mean
            # between competing holes.
            yy,xx=np.mgrid[max(0,y-3):min(64,y+4),max(0,x-3):min(64,x+4)]
            xy=origin+np.array([(crop*xx).sum(),(crop*yy).sum()])/max(mass,1e-12)
            k=output[i]['keypoints'][j]
            shift=float(np.linalg.norm(xy-[k['x'],k['y']]))
            k['coarse_x'],k['coarse_y']=k['x'],k['y']
            k['coarse_conf']=k['conf']
            k['refinement_peak_mass'],k['refinement_shift_px']=mass,shift
            if mass>=.6 and shift<=20:
                k['x'],k['y']=map(float,xy)
                k['conf']=min(k['conf'],mass)
            else:
                # An unconcentrated patch prediction is not permission to use
                # whichever nearby hole gives a more convenient circuit answer.
                k['conf']=min(k['conf'],.59)
        return output
