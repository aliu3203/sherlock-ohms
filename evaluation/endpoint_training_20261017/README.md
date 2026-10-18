# Initial wire/resistor endpoint experiment

Source: [Circuit End Point Detection v14](https://universe.roboflow.com/circuit-detection/circuit-end-point-detection/dataset/14), by circuit detection, CC BY 4.0. Original download and extracted source remain in Downloads/sherlock-ohms-datasets/circuit-endpoints-v14. Attribution files are included here.

The 306 original, unaugmented images show physical breadboards. Sample numbered endpoint overlays were inspected in label_audit.jpg. Source `register` represents resistors in inspected examples. Map wire (source 3) to 0 and register (source 2) to resistor 1. Preserve both keypoints and visibility; exclude capacitor/inductor objects. There is no LED training or evaluation. Images containing only excluded classes are useful negative examples for these two classes.

Prepared split: 245 training images (482 wires, 478 resistors), 41 validation (17 wires, 23 resistors), 20 test (49 wires, 70 resistors). Group filename sequences/sessions and perceptual near duplicates before splitting: hold out the 20241005 session for validation and the frame sequence for test. This reduces leakage compared to a random image split but does not establish independent circuit identities across all sources. Test images are correlated frames, so metrics are preliminary.

Training: official Ultralytics YOLO11n-pose pretrained checkpoint, two classes/two keypoints, 640px, batch 8, 40-epoch maximum, patience 12, Apple MPS. Geometric flips disabled to avoid unverified endpoint identity semantics. This standalone runner uses the same model family and conservative augmentations as vision_ml/train_pose.py, but avoids its automatic NCNN export and fixed 1280px GPU-oriented training configuration. Existing application files/defaults remain unchanged.

Run from repository root:

```sh
.venv/bin/python -u evaluation/endpoint_training_20261017/train.py
.venv/bin/python evaluation/endpoint_training_20261017/check_detector.py
.venv/bin/python evaluation/endpoint_training_20261017/check_detector.py --image /absolute/path/to/photo.jpg
```

`train.py` writes best/last weights, training plots, held-out metrics, and test prediction overlays. Re-running training creates another run directory; select its weights explicitly when using check_detector.py. `check_detector.py` loads an explicit checkpoint through the existing BreadboardPoseDetector, asserts mock mode is off, and uses its production letterbox/confidence behavior. Those results may differ from square 640px validation because the application uses a 1280x480 canvas.

Endpoint coordinates alone do not identify breadboard hole names or verify circuit correctness. Board calibration/registration and reference circuit checking remain separate steps. No QR codes are needed for image-only model inference.
