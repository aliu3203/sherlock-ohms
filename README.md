# Sherlock Ohms

A local breadboard inspection workbench that detects wire and resistor endpoints,
maps them to calibrated holes, and compares observed electrical connectivity with
a reference circuit. Electrically equivalent wiring can pass.

## Run the app

Python 3.13 is the tested environment. From the repository root:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
SHERLOCK_DATA_DIR="$PWD/.local-data" \
  .venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000/** for the trained-model workspace (it redirects to
`/live`). The original dashboard, including its browser demo, is available at
`/classic`, and interactive API documentation is at `/docs`.
There is no frontend build step.

- Upload a photo, or connect a browser camera and press **Take photo**, to detect
  wire/resistor endpoints without markers.
- For named hole IDs and wiring checks, attach ArUco DICT_4X4_50 markers 0, 1, 2,
  3 clockwise around the board and calibrate its four corner holes.
- Save a reference circuit, select it with your board calibration, and start a
  live check. Three consistent fresh frames are required for confirmed feedback.
- References, calibrations, and bounded inspection history are stored locally.

The app uses the two included final checkpoints:

```text
evaluation/hole_precision_20261017/runs/nano_endpoints/weights/best.pt
evaluation/hole_precision_20261017/refiner/best.pt
```

Together they contain about 2.78 million parameters and occupy 6.1 MB. CPU is the
default; set `SHERLOCK_ENDPOINT_DEVICE=mps` for the Apple GPU. No cloud inference
or paid service is required.

## Setup and project guides

- [Web app, API contract, and Base44/Paper handoff](app/README.md)
- [Model training, evaluation results, and physical-board setup](evaluation/hole_precision_20261017/README.md)
- [Printable 30 mm tracking markers](output/pdf/sherlock-ohms-markers-30mm.pdf)
- [Vision package and command-line tools](vision_ml/README.md)
- [Model card and checkpoint hashes](evaluation/hole_precision_20261017/model_card.json)
- [Third-party attribution](THIRD_PARTY_NOTICES.md)

## Validation

```sh
.venv/bin/python -m pip install -r requirements-dev.txt
SHERLOCK_DATA_DIR=/private/tmp/sherlock-tests .venv/bin/python -m pytest -q
node --test app/static/tests/*.test.mjs
```

The current suite has 143 passing Python tests and 19 frontend checks. Synthetic
geometry and injected endpoint tests verify software behavior; they do not
establish neural accuracy on a physical camera.

## Current limits

This is a testable prototype. Independent accuracy on the user's actual camera
and board still needs measurement. A small development hole audit accepted 16
correct component pairs out of 28 eligible pairs, with 12 unresolved. Its zero
incorrect accepted pairs is not proof of a zero false-green rate. Development
after base-test inspection means refined results require a fresh independent
test set before deployment accuracy claims.

Supported components are wires and resistors. Electrical contact, resistor
values, LEDs, ICs, and the new workspace's power-rail setup are outside its
verified scope. The reference editor specifies desired wiring; it does not
generate an optimal layout.

The Python API is for local development and has no authentication. Public
deployment requires authentication and user ownership checks. The local app has
not been published to Base44 or Paper.

## Reproducing training

The final detector was fine-tuned from YOLO11n-pose using two public endpoint
datasets: 847 images total, split into 550 training, 187 validation, and 110 test
images. A small patch network refines endpoint positions. Source datasets,
conversion/training scripts, original split manifests, and evaluation caveats are
documented in the experiment directory.

Downloaded images, virtual environments, local camera data/databases, caches,
intermediate checkpoints, and bulk old image overlays are excluded from Git.
The final two inference checkpoints are included deliberately. Historical audit
files retain original local source paths; adjust the dataset paths in the
preparation scripts when reproducing on another machine.
