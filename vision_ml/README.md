# Breadboard vision package

Run module commands from the repository root with the existing `.venv` Python.
No camera, GPIO, Torch, or Ultralytics import is required for mock inference or
registration; hardware and model dependencies are loaded only when requested.

## Registration and quality gate

```python
from vision_ml import BreadboardRegistration, QualityGateError

registration = BreadboardRegistration.from_calibration('vision_ml/calibration.json')
try:
    result = registration.register(camera_frame)  # uint8 BGR
    board = result.rectified  # 520 high x 1400 wide
except QualityGateError as error:
    print(error.result.quality.failures)
```

Before calibration, use `BreadboardRegistration(brightness_baseline=180,
sharpness_threshold=100)`. Tune the sharpness threshold for your camera.
Marker IDs are clockwise 0=top-left, 1=top-right, 2=bottom-right, 3=bottom-left.
The four marker centers map to rectified pixel centers `(0,0)`, `(1399,0)`,
`(1399,519)`, `(0,519)` at 8 px/mm. Thus their default physical separations are
174.875 x 64.875 mm. Arrange the physical markers accordingly, or supply measured
`marker_positions_mm={0: (x,y), ...}` in the same board coordinate frame and adjust
`output_size` if needed. A generic printable marker sheet is not a placement jig.

All four markers are mandatory. Sharpness, brightness, and glare are measured on
the rectified board region, excluding the surrounding camera scene. Sharpness is
Laplacian variance >= threshold; brightness must remain within ±15% of baseline;
>=250 grayscale pixels must occupy strictly less than 2% of the board. These gates
run before inference. `register(..., enforce_quality=False)` returns failure metrics
for diagnostics; a missing marker yields no warp. Duplicate IDs or invalid marker
geometry raise `ValueError`. `generate_synthetic_aruco_board_image()` supplies a
camera-free test scene with actual encoded markers.

## Printable markers and hole calibration

```sh
.venv/bin/python -m vision_ml.calibrate_cli
.venv/bin/python -m vision_ml.calibrate_cli --image vision_ml/rectified.png
```

The first command saves `vision_ml/aruco_markers_print.png`. The second displays
an image for clicks in order **a1, a63, j1, j63**. Press Enter after four clicks,
R to reset, or Esc to cancel. It saves `vision_ml/calibration.json` and
`vision_ml/hole_lut.json`. An OpenCV build with GUI support is needed for clicking.
`save_calibration(image, reference_points, output_dir)` supports headless use.
The LUT contains 630 hole IDs (`a1` through `j63`) mapped to `{x,y}` pixels. It
assumes 2.54 mm pitch and a 7.62 mm e-to-f center trench; verify this for your board.
Four clicks define a projective mapping, with the trench represented explicitly.

## Dataset, training, NCNN

Pose labels have 11 fields: class, normalized cx/cy/w/h, and two normalized x/y
endpoint positions each followed by visibility (0=missing, 1=occluded, 2=visible).
Keep endpoint order consistent, especially for LED polarity. The classes are
0=wire, 1=resistor, 2=led. `flip_idx: [0,1]` preserves endpoint identity.

Name images `BUILD__PHOTO.png` or `BUILD_PHOTO.png`. Default build IDs end at the
first underscore. For IDs with underscores, use `--build-pattern '^(?P<build>.+)__'`.
The split is 70/20/10 by **build count**, so photo counts can differ. At least three
builds are needed for all three splits. Image/label relative subfolders are preserved;
all labels must exist (empty files are valid negative examples).

```sh
.venv/bin/python -m vision_ml.split_dataset \
  --images-dir vision_ml/raw/images --labels-dir vision_ml/raw/labels \
  --output-dir vision_ml/dataset --seed 42
```

Set `path` in `vision_ml/breadboard.yaml` to the **absolute** dataset directory.
It must contain `images/{train,val,test}` and matching `labels/{train,val,test}`.
On a GPU machine or Colab GPU runtime:

```sh
.venv/bin/python -m vision_ml.train_pose --device 0
```

Default training uses YOLO11n-pose, imgsz=1280, epochs=100, batch=8, close_mosaic=10,
degrees=5, translate=.05, scale=.1, hsv_v=.3, hsv_s=.4, hsv_h=.01. CLI flags allow
batch/epochs/device/data overrides. Best validation weights are reloaded and
exported with `format='ncnn', imgsz=(480,1280)`. Downloads and export tooling require
network access on the training machine. Runs default to `vision_ml/runs`; actual
output paths are printed. Export/inference APIs follow the
[Ultralytics NCNN documentation](https://docs.ultralytics.com/integrations/ncnn/).

## Inference and pre-labeling

```python
from vision_ml import BreadboardPoseDetector

detector = BreadboardPoseDetector('vision_ml/runs/breadboard/weights/best_ncnn_model')
# A trained .pt path also works. No path creates an empty mock detector.
detections = detector(board)
```

Input is a rectified BGR image, normally 1400x520. It resizes to 1280x475 with
2 top/3 bottom padding pixels. Both exact rounded resize scales are inverted for
output coordinates. Boxes/keypoints are clipped to original pixel-center bounds.
Box confidence .40 and NMS IoU .50 are passed to Ultralytics; both endpoint
confidences must be >=.50. Each output has `id`, `type`, `conf`, `bbox` (four-float
tuple), and two `keypoints` with `x`, `y`, `conf`, `vis`. IDs restart at `det_1` per
image. Predicted endpoints have `vis=2`; this is a review proposal, not a measured
occlusion state. Invalid model paths raise instead of silently producing mocks.

After the first 30 manually annotated images and an initial training run:

```sh
.venv/bin/python -m vision_ml.prelabel \
  --model vision_ml/runs/breadboard/weights/best.pt \
  --images vision_ml/unlabeled --output vision_ml/proposed_labels
```

PNG discovery is recursive; relative folders are preserved. Existing label files
are skipped unless `--overwrite` is given. No detections produce an empty label.
Review all predictions and recover missed components before adding training data.

## Pi hardware and laptop fallback

```python
from vision_ml.pi_hardware import BreadboardCamera, TrafficLight

with BreadboardCamera(local_images='vision_ml/local_frames') as camera, TrafficLight() as lights:
    lights.set('yellow')
    frame = camera.capture_burst()  # sharpest of three frames
    # Register, check quality, then infer before choosing red/green.
```

Pi focus is manual; exposure/gain and white balance are locked after warmup. Queued
frames are drained before the first burst. GPIO LEDs are red=17, yellow=27,
green=22, with only one lit at a time. Missing/failed Pi hardware automatically uses
cyclic local PNG frames and colored `[GPIO] GREEN`-style terminal logs. Supply local
images for camera fallback. `force_fallback=True` bypasses hardware probing.

Vision tests only:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTEST_ADDOPTS='-p no:cacheprovider' .venv/bin/pytest tests/test_vision_ml.py
```

## Zero-training classical detector (Option B)

Use `ClassicalBreadboardDetector` with the same registered 1400x520 coordinate
system for the empty baseline, inspection frames, and hole LUT:

```python
from vision_ml import ClassicalBreadboardDetector
from pipeline import SherlockOhmsPipeline

# baseline may be a BGR numpy image or an image filename.
detector = ClassicalBreadboardDetector(
    baseline_img='vision_ml/empty_baseline.png',
    led_anode_direction='left',  # only if this is the known demo orientation
)
pipeline = SherlockOhmsPipeline(golden_layout, lut=calibrated_lut, detector=detector)
result = pipeline.inspect(raw_camera_frame)
```

To create the baseline, collect at least three **empty**, rectified frames with
focus/exposure locked, then use:

```python
detector = ClassicalBreadboardDetector(baseline_frames=empty_rectified_frames)
detector.save_baseline('vision_ml/empty_baseline.png')
# Or call detector.detect(rectified_frame) directly for RawDetection dictionaries.
```

Median capture rejects transient objects present in fewer than half the frames.
The detector never updates the baseline from inspected, populated boards. A missing
baseline raises an error instead of learning the current circuit as background.

Foreground segmentation uses absolute color differences, adaptive threshold seeds,
and connected contours. A small global exposure offset is compensated; large
lighting changes or widespread foreground raise `BaselineMismatchError`. Geometry,
body thickness and saturation distinguish wires, resistors and LED domes. PCA and
skeleton lead tips provide two endpoints in original rectified coordinates. Small
speckles are removed without opening away thin leads. Tunable constructor settings
include `min_difference`, `min_area`, `body_radius`, and `max_blob_area`.

Keep components visually separated with both insertion tips visible. Crossing or
touching components can merge into one contour, and occluded leads cannot be
recovered reliably. Unresolved/cropped endpoints have confidence below .50 so the
snapping engine marks them uncertain. Confidence scores are heuristics. LED color
cannot establish polarity: the default leaves its endpoint confidence below .50.
`led_anode_direction='left'/'right'/'top'/'bottom'` is an explicit known-orientation
assumption for all LEDs in the demo, not a polarity measurement. Inspect the overlay
and tune thresholds against your real camera before presenting physical-board results.

The segmentation uses standard [OpenCV thresholding](https://docs.opencv.org/4.10.0/d7/d4d/tutorial_py_thresholding.html)
and contour geometry; no checkpoint or training is required.
