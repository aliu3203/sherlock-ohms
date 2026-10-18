from pathlib import Path
import json,os
from ultralytics import YOLO
ROOT=Path(__file__).resolve().parent

def main():
 os.chdir(ROOT)
 model=YOLO(str(ROOT.parent/'endpoint_training_20261017/runs/wire_resistor/weights/best.pt'))
 model.train(data=str(ROOT/'data.yaml'),imgsz=640,epochs=60,batch=8,device='mps',workers=0,
  project=str(ROOT/'runs'),name='nano_endpoints',exist_ok=False,seed=2026,patience=15,
  optimizer='AdamW',lr0=.0008,lrf=.1,close_mosaic=10,mosaic=.25,degrees=5,translate=.05,
  scale=.15,hsv_v=.25,hsv_s=.3,hsv_h=.01,fliplr=0,flipud=0,cache=False,plots=True)
 best=Path(model.trainer.best)
 chosen=YOLO(str(best))
 result=chosen.val(data=str(ROOT/'data.yaml'),split='test',imgsz=640,device='mps',workers=0,
  project=str(ROOT/'runs'),name='test',plots=True)
 (ROOT/'training_result.json').write_text(json.dumps({'weights':str(best),'parameters':sum(p.numel() for p in chosen.model.parameters()),'test_metrics':result.results_dict},indent=2))
 print('TRAINING_COMPLETE',best)
if __name__=='__main__':main()
