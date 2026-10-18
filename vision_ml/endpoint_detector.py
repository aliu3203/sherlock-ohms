"""Small two-endpoint model, retaining uncertain detections for the verifier."""
from pathlib import Path
import numpy as np


class EndpointDetector:
    def __init__(self, weights, *, device='cpu', image_size=1280, proposal_confidence=.15):
        path = Path(weights)
        if not path.is_file():
            raise FileNotFoundError(path)
        if image_size < 320 or image_size % 32 or not 0 < proposal_confidence < 1:
            raise ValueError('Use a positive confidence and an image size divisible by 32')
        from ultralytics import YOLO
        self.model = YOLO(str(path), task='pose')
        self.names = dict(self.model.names)
        if self.names != {0: 'wire', 1: 'resistor'}:
            raise ValueError(f'Expected wire/resistor endpoint weights, found {self.names}')
        if tuple(self.model.model.model[-1].kpt_shape) != (2, 3):
            raise ValueError('Expected two endpoint keypoints with visibility confidence')
        self.device, self.image_size = device, image_size
        self.proposal_confidence = proposal_confidence
        self.weights = str(path.resolve())

    def detect(self, image):
        if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or
                image.ndim != 3 or image.shape[2] != 3 or not image.size):
            raise ValueError('Expected a nonempty uint8 BGR image')
        # Enlarging a low-resolution photo creates no new hole information.
        size = min(self.image_size, ((max(image.shape[:2]) + 31) // 32) * 32)
        result = self.model.predict(image, imgsz=size, rect=True,
                                    conf=self.proposal_confidence, iou=.5,
                                    device=self.device, verbose=False)[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []
        boxes = result.boxes.data.cpu().numpy()
        points = result.keypoints.data.cpu().numpy()
        if points.shape != (len(boxes), 2, 3):
            raise ValueError('Model returned an incompatible endpoint shape')
        # Ultralytics returns coordinates in the original image. Do not re-scale
        # or clip them to the nearest valid hole, which could hide bad predictions.
        return [{'id': f'det_{i}', 'type': self.names[int(b[5])],
                 'conf': float(b[4]), 'bbox': b[:4].tolist(),
                 'keypoints': [{'x': float(p[0]), 'y': float(p[1]), 'conf': float(p[2])}
                               for p in k]}
                for i, (b, k) in enumerate(zip(boxes, points))]
