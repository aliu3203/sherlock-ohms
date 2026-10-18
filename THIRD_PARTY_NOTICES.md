# Third-party model and dataset notices

## Ultralytics

The detector uses Ultralytics YOLO11n-pose. Initial training started from the
official `yolo11n-pose.pt` checkpoint, followed by local wire/resistor fine-tuning.
The installed Ultralytics distribution identifies its license as AGPL-3.0. A
verbatim copy is retained in [licenses/ultralytics-AGPL-3.0.txt](licenses/ultralytics-AGPL-3.0.txt).
See the [upstream project](https://github.com/ultralytics/ultralytics) for its source
and licensing information. The project source includes the model adaptations,
training scripts, and inference wrapper used for the included final detector.

The endpoint patch refiner is a separate small PyTorch network trained locally;
it is not an official Ultralytics model.

## Final training datasets

Both downloaded dataset notices identify their license as
[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
Dataset availability and license notices describe the source uploads; they do not
constitute an independent verification of all underlying image rights.

- **Ohm vision keypoint ArUco 2, version 2**, provided by **Komkrit Workspace**:
  [source](https://universe.roboflow.com/komkrit-workspace/ohm-vision-keypoint-aruco-2/dataset/2),
  541 photos. Class IDs were remapped to wire=0, resistor=1. Consecutive capture
  groups and close perceptual duplicates were regrouped across train/validation/test.
  The [original notice](evaluation/hole_precision_20261017/ohm_README.dataset.txt)
  and export metadata are retained.
- **Circuit end point detection, version 14**, provided by the **Circuit
  Detection workspace**:
  [source](https://universe.roboflow.com/circuit-detection/circuit-end-point-detection/dataset/14),
  306 photos. The source `register` class was mapped to resistor; capacitor and
  inductor labels were excluded. Labels were converted to two endpoint keypoints.
  The [original notice](evaluation/endpoint_training_20261017/README.dataset.txt)
  and export metadata are retained.

The final experiment contains selected derived diagnostic overlays with predicted
or source annotation points. These are development examples, not new independent
ground truth. Bulk downloaded datasets and older BME dataset image crops are not
included. Earlier BME audit reports remain as historical records and do not
describe the final trained detector.

Generated ArUco print sheets, synthetic test boards, and the frontend's breadboard
illustration are programmatic project assets. Third-party library licenses remain
with their respective distributions. This notice records dependency provenance;
it does not assign a new project-wide license to the team's original code.
