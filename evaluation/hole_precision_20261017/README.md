# Hole-aware breadboard inspection experiment

This adds a local camera pipeline around a small wire/resistor endpoint model.
It tracks a calibrated board using four ArUco markers, verifies visible hole
alignment, associates each endpoint with a hole, and compares electrical topology
with a reference layout. Equivalent holes and alternate wire routes can pass.

The first version supports **wires and resistors only**. It does not identify
resistor values, LEDs, polarity, IC pins, hidden connections, or internal contact.
`CORRECT` means the **observed wire/resistor topology** matches the reference.
It is a prototype, not a continuity tester. Real-board accuracy is not established
by dataset AP or the synthetic integration tests.

## Data and model

- [Ohm vision keypoint ArUco 2, version 2](https://universe.roboflow.com/komkrit-workspace/ohm-vision-keypoint-aruco-2/dataset/2): 541 photos, CC BY 4.0. Attribution: Komkrit Workspace; original export notices remain in the downloaded source.
- [Circuit end point detection, version 14](https://universe.roboflow.com/circuit-detection/circuit-end-point-detection/dataset/14): 306 photos, CC BY 4.0. Attribution: Circuit Detection workspace. Its `register` class is mapped to resistor; capacitor/inductor labels are excluded.
- Combined: **847 images**; 550 training, 187 validation, 110 test.
- Ohm labels map resistor/wire to the shared wire=0, resistor=1 format. Both use two endpoint keypoints.
- Consecutive capture groups and close perceptual duplicates stay in one split. Earlier Circuit session splits are preserved. Scene independence is still not guaranteed because circuit IDs are not provided.
- `dataset_audit.json` records the split, source paths, image hashes and counts. `prepare.py` reproduces conversion; the generated dataset has separate image and label directories.
- YOLO11n pose, about **2.66 million parameters**, fine-tuned from the previous local wire/resistor checkpoint. Training: 640px, batch 8, MPS, at most 60 epochs, early stopping after 15 epochs without improvement. No cloud jobs.
- Final weights: `runs/nano_endpoints/weights/best.pt`. Training writes `training_result.json` after held-out evaluation; application scoring is separate in `model_evaluation/metrics.json`.
- Optional local endpoint refinement: `refiner/best.pt`, a 118,705-parameter
  patch heatmap network trained for 40 epochs from the same train/validation
  keypoints. It examines a 64px neighborhood around each coarse endpoint.
  Combined model size is 2,780,857 parameters. `refined_evaluation/metrics.json`
  records the full detector-plus-refiner audit. Unconcentrated or excessive
  refinements cannot change the hole selected by an unambiguous coarse estimate.
  Conflicting hole choices cause uncertainty. If patch evidence is inconclusive,
  an original endpoint can be used only if it already passes the strict baseline
  hole check. The final hole-association radius remains one-quarter of a pitch.

No dataset contains independently verified hole IDs or measured electrical
continuity. Hole numbering comes from board calibration. `evaluate.py` includes a
nine-photo **hole-assignment proxy** using visually selected anchors and source
keypoint labels. It reports excluded rail/ambiguous labels and missed objects;
this is not an independent exact-hole accuracy benchmark.

The refiner was added after inspecting base-model test errors. Its weights use
only train/validation images, but the refined test results are now a development
audit, not a fresh independent final test. A new held-out set from the actual
camera and board is needed for a defensible deployment accuracy claim.

## Completed results

Both training runs completed locally. The detector ran 60 epochs; the small
refiner ran 40. Their checkpoints total **6,121,109 bytes (about 6.1 MB)**.
The repository test suite passes **130 tests** (one pre-existing FastAPI/httpx
deprecation warning). Test fixtures cover logic and geometry, not neural accuracy.

On the 90 close-up Ohm test/development photos (141 labeled wires, 179 resistors):

| Measure | Wires | Resistors |
|---|---:|---:|
| Detection precision at confidence 0.6 | 100% | 99.4% |
| Detection recall | 96.5% | 98.9% |
| Median endpoint error, before refinement | 4.74 px | 7.58 px |
| Median endpoint error, with refinement | 1.22 px | 1.07 px |
| 90th-percentile endpoint error, with refinement | 11.0 px | 18.1 px |

Endpoint errors are conditional on correctly box-matched detections; missing
objects are counted separately. These numbers describe detection and point
localization, **not exact-hole correctness**. The large tail errors matter.
Performance on the other 20 Circuit photos is worse: median wire/resistor endpoint
errors remain 14.8/10.0 px. Do not assume generalization to arbitrary camera views.

The nine-photo hole audit contains 40 labeled components. One photo is rejected
because its boundary holes are obscured; rail/out-of-grid/ambiguous source labels
are also excluded. Of **28 eligible component pairs, 16 are assigned correctly,
12 remain unresolved, and 0 incorrect pairs are accepted**. This small,
correlated annotation-based development audit is not proof of a zero false-green
rate. Before the agreement guard, unconstrained patch refinement accepted three
wrong pairs; that diagnostic is retained in
`refined_evaluation/before_agreement_guard_metrics.json`.

`known_wire_refined/results.json` records a separate visually audited photo:
the wire enters physical holes **d3 and d24**. On MPS, the real trained models
identify those holes. Matching and same-strip-equivalent references produce a
`CORRECT` candidate; a one-column change or wrong-half reference produces an
`INCORRECT` candidate. All four still-image results remain `UNCERTAIN` overall
because they have not supplied three consecutive camera frames. No still image
was replayed to manufacture temporal confirmation.

Detector-plus-refiner processing took a median of about 70 ms on CPU over the
110 photos, excluding the first call. That excludes camera capture, registration,
and UI display; it is not a measured live camera frame rate. MPS inference and
the image-preview CLI were also exercised successfully.

**Deployment status:** a testable local prototype. The real camera, marker
attachment, board calibration, and independent correct/wrong-hole clips still
need validation before trusting red/green feedback in use. Electrical contact,
component values, and unsupported component types remain outside its scope.

## Test a photo immediately

Run from the repository root using the existing environment:

```sh
.venv/bin/python -m vision_ml.live_inspector detect \
  --weights evaluation/hole_precision_20261017/runs/nano_endpoints/weights/best.pt \
  --refiner evaluation/hole_precision_20261017/refiner/best.pt \
  --image /absolute/path/to/your_photo.jpg \
  --device mps \
  --output evaluation/my_first_endpoint_preview
```

Open `endpoints.jpg` beside `endpoints.json`. Blue circles are predicted endpoints,
not correctness indicators. This command needs no markers and returns
`UNCALIBRATED`, with no red/green verdict. Every run needs a new output directory.
Use `--device cpu` if MPS is unavailable.

## Calibrate the real board once

1. Print `aruco_markers_print.png`. These are **ArUco DICT_4X4_50 markers**, not
   ordinary QR codes. Put IDs 0, 1, 2, 3 clockwise around the board, starting at the
   top left. Keep all four visible, firmly attached to the same assembly, and as
   close as practical to the plane of the holes. Camera motion is supported;
   moving a marker relative to the board invalidates calibration.
2. Take a sharp overhead photo of the empty board, including all four markers.
   Avoid clipped white highlights and use steady, diffuse lighting. Save the raw
   camera photo, not a cropped/previously rectified image.
3. Run:

```sh
.venv/bin/python -m vision_ml.live_inspector calibrate \
  --image /absolute/path/to/empty_board.jpg \
  --columns 63 \
  --size 1400 520 \
  --output evaluation/my_board_setup
```

Click the centers of `a1`, `a63`, `j63`, and `j1` when prompted. Use the board's
actual printed numbering, not an assumed camera orientation. For a 30-column
board use `--columns 30 --size 800 400`. Choose an output aspect ratio reasonably
close to the physical marker rectangle, or pass an existing measured
`--registration /path/to/calibration.json` for its geometry/quality configuration.

The grid assumes regular terminal spacing and a three-pitch gap between rows e
and f. Refinement checks real dark holes rather than silently accepting clicked
geometry. Inspect `calibration_overlay.jpg`: every plotted center and named
corner must match the board. Recalibrate if they do not. Setup data carries a
content fingerprint so accidental edits fail closed.

Keep some holes visible in the first/last columns and rows a/j. Boundary checks
reduce the chance of accepting a whole-hole shift in the repeating pattern;
covering the edges can therefore cause an amber result even with correct wiring.
Markers must remain fixed to the board: image checks do not prove attachment.

For headless calibration, supply `--anchors anchors.json`. Coordinates are in
the rectified image; `rectified_empty.jpg` is written before interactive clicks.
The JSON structure is:

```json
{
  "anchors": {"a1": [70,80], "a30": [650,80], "j30": [650,300], "j1": [70,300]},
  "rail_holes": {"positive_1": [70,30], "positive_2": [90,30]},
  "rail_segments": {"positive_left": ["positive_1", "positive_2"]}
}
```

Those numbers are only a format example. Measure your own coordinates. Rails
are optional and never guessed: list each actual rail hole and its electrically
connected segment, including splits. Terminal-only demos need no rail entries.

## Define the reference and start inspection

`wire_reference.json` is a tiny first test: connect strips 1 and 5 on the a-e
half. Any pair of holes in those strips, or a chain of wires joining them, is
electrically equivalent. A wire joining strips 2 and 6 must fail this reference.

For a resistor circuit, use the same `parts` schema with `type: "resistor"` and
two hole IDs in `pins`. IDs are arbitrary and need not match detector IDs.
`terminals` maps fixed external connection names to measured hole IDs, for example
`{"VCC":"positive_1", "GND":"negative_1"}`. These anchors preserve connections
to external equipment while allowing internal components to move. Without
external terminals, resistor networks are compared up to graph isomorphism.
Wire-only layouts automatically anchor their reference endpoint strips if no
explicit terminals are supplied. Unsupported component types are rejected.

```sh
.venv/bin/python -m vision_ml.live_inspector inspect \
  --setup evaluation/my_board_setup/setup.json \
  --weights evaluation/hole_precision_20261017/runs/nano_endpoints/weights/best.pt \
  --refiner evaluation/hole_precision_20261017/refiner/best.pt \
  --reference evaluation/hole_precision_20261017/wire_reference.json \
  --camera 0 --device mps --show \
  --output evaluation/my_first_live_test
```

Use `--video /path/to/clip.mp4` or `--image /path/to/frame.jpg` instead of
`--camera`. Escape closes the preview. `--max-frames 100` bounds a camera test.
Logs are `verdicts.jsonl`; `latest.jpg` is the latest rectified overlay.

- **Green / CORRECT:** three consecutive frames with the same confidently
  assigned holes and equivalent observed connectivity.
- **Red / INCORRECT:** a stable observed mismatch or short. A global mismatch
  does not prove which individual wire caused it.
- **Amber / UNCERTAIN:** missing markers, bad image quality, inconsistent hole
  alignment, ambiguous endpoints, weak proposals, or still settling.
- **Amber / INCOMPLETE:** expected components are missing or undetected.

A single still image remains amber and exposes `candidate_status` if it can be
evaluated. Replaying a still image does not provide new accuracy evidence.
Confidence thresholds are heuristics, not calibrated probabilities. A confident
wrong prediction or a completely missed extra wire can still cause a false pass;
temporal agreement cannot fix a consistently wrong model.

## Web workspace and Python integration

The local `/live` workspace now uses this detector and refiner through
`app/hole_api.py`. It supports photo uploads, camera snapshots, board calibration,
saved reference circuits, live sessions, and bounded SQLite history. See
[`app/README.md`](../../app/README.md) for startup and API details. The original
dashboard is at `/classic`; its older detector path is separate.

Use `vision_ml.live_inspector.LiveInspector(setup=..., weights=...,
reference=..., refiner_weights=..., device='mps')`, then call `inspect(frame)` with a BGR NumPy image.
It returns `(JSON-compatible verdict, annotated_image)`. Keep one instance per
camera session. The result includes actual hole IDs, ambiguous assignments,
tracking quality, frame time and an outcome. Respect `valid_for_ms`: clear a
displayed green result if fresh observations stop. Invalid input raises and
resets the settling state. The new web API stores its data separately from the
original dashboard's references and inspections.

## Validation and next real-board measurements

`tests/test_hole_inspection.py` exercises actual synthetic ArUco registration,
grid fitting, electrical equivalence, shorts, uncertainty resets and setup
tampering. The integration test injects known endpoints; it does not measure
neural inference accuracy. Run `.venv/bin/python -m pytest -q` for the repository.

After calibration, record held-out clips with explicit correct/wrong-hole
labels: same-strip equivalents, adjacent-column mistakes, gap crossings, rail
splits, hands covering endpoints, and changed lighting. Keep these clips out of
training. Report exact-hole precision/recall, false-green rate, uncertainty rate,
and latency separately. Manually review every proposed label before fine-tuning.
Do not train from the application's own verdicts.

New implementation is isolated in `vision_ml/hole_grid.py`,
`vision_ml/endpoint_detector.py`, `vision_ml/endpoint_refiner.py`, `vision_ml/live_inspector.py`,
`engine/live_verifier.py`, and `tests/test_hole_inspection.py`. It reuses existing
registration without changing the earlier detector or classical pipeline.
This keeps the work separate from the existing Codex/Claude-owned modules.
