"""Run with python -m vision_ml.train_pose (GPU/Colab)."""
import argparse
from pathlib import Path


def train_and_export(data=None, epochs=100, batch=8, device='0',
                     project=None, name='breadboard', workers=2):
    from ultralytics import YOLO
    package = Path(__file__).resolve().parent
    data = Path(data or package / 'breadboard.yaml').resolve()
    model = YOLO(str(package / 'yolo11n-pose.pt'))
    model.train(data=str(data), imgsz=1280, epochs=epochs, batch=batch,
                close_mosaic=10, degrees=5, translate=.05, scale=.1,
                hsv_v=.3, hsv_s=.4, hsv_h=.01, device=device, workers=workers,
                project=str(project or package / 'runs'), name=name)
    best = Path(model.trainer.best)
    if not best.is_file():
        raise FileNotFoundError(f'No best checkpoint generated: {best}')
    model = YOLO(str(best))
    exported = model.export(format='ncnn', imgsz=(480, 1280))
    return best, exported


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--device', default='0')
    parser.add_argument('--project', type=Path)
    parser.add_argument('--name', default='breadboard')
    parser.add_argument('--workers', type=int, default=2)
    best, exported = train_and_export(**vars(parser.parse_args()))
    print(f'Best checkpoint: {best}\nNCNN model: {exported}')


if __name__ == '__main__':
    main()
