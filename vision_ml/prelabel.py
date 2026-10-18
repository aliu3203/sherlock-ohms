"""Reviewable labels: python -m vision_ml.prelabel --model best.pt --images DIR --output DIR."""
import argparse
from pathlib import Path
import cv2
from .detector import BreadboardPoseDetector, CLASS_NAMES


def pose_label(detection, width, height):
    x0, y0, x1, y1 = detection['bbox']
    values = [(x0+x1)/(2*width), (y0+y1)/(2*height), (x1-x0)/width, (y1-y0)/height]
    for kp in detection['keypoints']:
        values.extend((kp['x']/width, kp['y']/height, kp['vis']))
    cls = {name: idx for idx, name in CLASS_NAMES.items()}[detection['type']]
    return f'{cls} ' + ' '.join(f'{value:.8f}' for value in values)


def prelabel_images(model, images, output, *, overwrite=False, device='cpu',
                    box_conf=.4, iou=.5, kpt_conf=.5):
    images, output = Path(images), Path(output)
    if not images.is_dir():
        raise NotADirectoryError(images)
    sources = sorted(p for p in images.rglob('*') if p.suffix.lower() == '.png')
    if not sources:
        raise ValueError('No PNG images found')
    if model is None:
        raise ValueError('Prelabeling requires a trained model')
    detector = BreadboardPoseDetector(model, device=device, box_conf=box_conf,
                                      iou=iou, kpt_conf=kpt_conf)
    written = skipped = 0
    for source in sources:
        target = (output / source.relative_to(images)).with_suffix('.txt')
        if target.exists() and not overwrite:
            skipped += 1
            continue
        image = cv2.imread(str(source))
        if image is None:
            raise ValueError(f'Cannot read {source}')
        h, w = image.shape[:2]
        lines = [pose_label(d, w, h) for d in detector(image)]
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('w' if overwrite else 'x') as handle:
            handle.write('\n'.join(lines) + ('\n' if lines else ''))
        written += 1
    return {'written': written, 'skipped': skipped}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('model', 'images', 'output'):
        parser.add_argument('--'+option, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--box-conf', type=float, default=.4)
    parser.add_argument('--iou', type=float, default=.5)
    parser.add_argument('--kpt-conf', type=float, default=.5)
    print(prelabel_images(**vars(parser.parse_args())))
    print('Review all labels and missed components before training.')


if __name__ == '__main__':
    main()
