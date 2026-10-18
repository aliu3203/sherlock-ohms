"""Fixed-size pose inference with reversible pixel-center geometry."""
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np

CLASS_NAMES = {0: 'wire', 1: 'resistor', 2: 'led'}
INPUT_SIZE = (1280, 480)  # width, height; Ultralytics takes (height, width)


@dataclass(frozen=True)
class LetterboxTransform:
    original_size: tuple
    resized_size: tuple
    padding: tuple

    @property
    def scale(self):
        return np.asarray(self.resized_size) / np.asarray(self.original_size)

    def map_points(self, points):
        return np.asarray(points, dtype=np.float64) * self.scale + self.padding

    def unmap_points(self, points, *, clip=False):
        points = (np.asarray(points, dtype=np.float64) - self.padding) / self.scale
        return np.clip(points, 0, np.asarray(self.original_size) - 1) if clip else points


def letterbox(image, target_size=INPUT_SIZE):
    if (not isinstance(image, np.ndarray) or image.ndim != 3 or
            image.shape[2] != 3 or image.dtype != np.uint8 or image.size == 0):
        raise ValueError('Expected a nonempty uint8 HxWx3 BGR image')
    h, w = image.shape[:2]
    tw, th = target_size
    ratio = min(tw/w, th/h)
    nw, nh = round(w*ratio), round(h*ratio)
    left, top = (tw-nw)//2, (th-nh)//2
    resized = cv2.resize(image, (nw, nh))
    padded = cv2.copyMakeBorder(resized, top, th-nh-top, left, tw-nw-left,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return padded, LetterboxTransform((w, h), (nw, nh), (left, top))


class BreadboardPoseDetector:
    """No path means explicit empty mock results; invalid paths always raise.

    Each returned component has both endpoints above kpt_conf. vis=2 means a
    model-proposed visible endpoint, requiring human review for training labels.
    """
    def __init__(self, model_path=None, *, box_conf=.40, iou=.50, kpt_conf=.50,
                 device='cpu'):
        for value in (box_conf, iou, kpt_conf):
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('Confidence/IoU thresholds must be within [0, 1]')
        self.box_conf, self.iou, self.kpt_conf = box_conf, iou, kpt_conf
        self.device = device
        self.model = None
        if model_path is not None:
            path = Path(model_path)
            if not path.exists():
                raise FileNotFoundError(path)
            from ultralytics import YOLO
            self.model = YOLO(str(path), task='pose')

    @property
    def mock(self):
        return self.model is None

    def detect(self, image):
        padded, transform = letterbox(image)
        if self.mock:
            return []
        result = self.model.predict(padded, imgsz=(480, 1280), rect=False,
                                    conf=self.box_conf, iou=self.iou,
                                    device=self.device, verbose=False)[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []
        if result.keypoints is None:
            raise ValueError('Expected a pose model with two endpoint keypoints')
        boxes = result.boxes.data.cpu().numpy()
        keypoints = result.keypoints.data.cpu().numpy()
        if boxes.ndim != 2 or boxes.shape[1] != 6:
            raise ValueError('Expected untracked detection boxes [x1,y1,x2,y2,conf,class]')
        if keypoints.shape != (len(boxes), 2, 3):
            raise ValueError(f'Expected (N, 2, 3) keypoints, got {keypoints.shape}')
        names = getattr(result, 'names', CLASS_NAMES)
        detections = []
        for box, points in zip(boxes, keypoints):
            if not np.isfinite(box).all() or not np.isfinite(points).all():
                continue
            if box[4] < self.box_conf or np.any(points[:, 2] < self.kpt_conf):
                continue
            cls = int(box[5])
            if cls not in CLASS_NAMES or names[cls] != CLASS_NAMES[cls]:
                raise ValueError('Checkpoint class mapping does not match breadboard.yaml')
            bbox = transform.unmap_points(box[:4].reshape(2, 2), clip=True).reshape(-1)
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            xy = transform.unmap_points(points[:, :2], clip=True)
            detections.append({'id': f'det_{len(detections)+1}', 'type': CLASS_NAMES[cls],
                               'conf': float(box[4]), 'bbox': tuple(map(float, bbox)),
                               'keypoints': [{'x': float(p[0]), 'y': float(p[1]),
                                              'conf': float(c), 'vis': 2}
                                             for p, c in zip(xy, points[:, 2])]})
        return detections

    __call__ = detect
