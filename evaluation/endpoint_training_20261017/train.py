"""Local initial YOLO11n pose experiment, isolated from application defaults."""
from pathlib import Path
import json, os
from ultralytics import YOLO
ROOT=Path(__file__).resolve().parent

def main():
    os.chdir(ROOT)
    model=YOLO('yolo11n-pose.pt')
    model.train(data=str(ROOT/'data.yaml'), imgsz=640, epochs=40, batch=8,
                device='mps', workers=0, project=str(ROOT/'runs'), name='wire_resistor',
                exist_ok=False, seed=42, deterministic=True, patience=12,
                close_mosaic=10, degrees=5, translate=.05, scale=.1,
                hsv_v=.3, hsv_s=.4, hsv_h=.01, fliplr=0, flipud=0,
                cache=False, plots=True)
    best=Path(model.trainer.best)
    model=YOLO(str(best))
    metrics=model.val(data=str(ROOT/'data.yaml'),split='test',imgsz=640,
                      device='mps',batch=8,workers=0,project=str(ROOT/'runs'),
                      name='heldout_test',plots=True)
    summary={'weights':str(best),'test_metrics':metrics.results_dict,
             'limitations':['Only wire and resistor; no LEDs','Small dataset; session grouping is approximate','Pose AP is not hole accuracy or circuit correctness']}
    (ROOT/'results.json').write_text(json.dumps(summary,indent=2))
    model.predict(source=str(ROOT/'dataset/images/test'),imgsz=640,device='mps',
                  conf=.25,save=True,project=str(ROOT/'runs'),name='test_predictions')
    print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
