# Sherlock Ohms web workspace

The new workspace is at **http://127.0.0.1:8000/live**. It uses the trained
wire/resistor endpoint detector and patch refiner. The previous dashboard remains
at `/` and has a link to Live wiring.

Start from the repository root:

```sh
SHERLOCK_DATA_DIR="$PWD/.local-data" \
  .venv/bin/python -B -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open `/docs` for the interactive API contract. No Node build or cloud account is
needed. Browser camera access works on localhost or HTTPS; the browser asks for
permission when you press Connect camera. Stop, navigate away, or disconnect to
end capture. Frames are posted sequentially, so a slow model does not create a
growing upload queue.

## What works

- **Photo preview:** upload a JPEG, PNG, or WebP, up to 8 MB / 16 megapixels. The
  real trained model annotates wire and resistor endpoints. Without calibration,
  the result is `UNCALIBRATED`, amber, with no circuit correctness claim.
  To take a photo directly, press **Connect camera**, allow camera access, aim at
  the board, then press **Take photo**. This captures one frame, turns the camera
  off, and inspects it through the same photo path. No markers are needed for
  endpoint preview; a saved board and reference enable the calibrated still check.
- **Board setup:** print the provided ArUco DICT_4X4_50 markers, attach IDs 0–3
  clockwise from top left, and upload a sharp overhead empty-board photo. Choose
  17, 30, or 63 columns. Click the four named corner holes on the rectified image.
  Saved calibration must pass the same measured-hole checks as the CLI. Ordinary
  QR codes cannot replace these markers. Keep the marker rectangle in landscape
  orientation, reasonably close to the board's aspect ratio; extreme stretching
  can cause calibration to be rejected. Terminal rows a–j are supported; this UI
  does not configure power rails.
- **References:** create wire/resistor circuits with two hole names per part and
  optional VCC/GND terminal holes. Equivalent connections within strips and
  alternate wire routes can pass. Unsupported components and duplicate occupied
  holes are rejected. This editor specifies desired wiring; it does not calculate
  an optimal layout.
- **Live wiring:** choose a saved board and circuit, connect the camera, then
  start checking. Each browser capture is a new frame. The server requires three
  consistent, distinct frames for confirmed red/green and resets stability after
  errors, lost tracking, repeated frames, or gaps. The browser expires feedback
  within two seconds of capture, including upload/inference time. A still photo
  stays amber and can report a candidate only.
- **History:** references, setup metadata, and the latest 100 inspection records
  and JPEGs persist in `.local-data/live/live.sqlite3`. A live session updates one
  row instead of creating a new history row for every frame. Only the newest 50
  records are listed. Setup documents are stored beside the DB. Browser storage
  remembers the selected reference, board, and backend address.

The weights are discovered under
`evaluation/hole_precision_20261017/runs/nano_endpoints/weights/best.pt` and
`evaluation/hole_precision_20261017/refiner/best.pt`. Missing or incompatible
weights produce an error; this API does not return mock detections. Models load
on first use and are shared under a lock; temporal verifier state stays isolated
per session. CPU is the default. Set `SHERLOCK_ENDPOINT_DEVICE=mps` to use Apple
GPU inference. `SHERLOCK_ENDPOINT_WEIGHTS` and `SHERLOCK_ENDPOINT_REFINER` can
override the two checkpoint paths.

## Frontend / backend contract

All routes below are under `/api/live`. Images are JSON `image_base64` strings
(responses use raw JPEG base64, without a data URL prefix).

| Method | Route | Purpose |
|---|---|---|
| GET | `/status` | Model availability, saved boards, references, supported scope |
| POST | `/preview` | Real endpoint detection from `{image_base64}` |
| GET, POST | `/references` | List or save `{name,parts,terminals}` |
| POST | `/calibration/preview` | Rectify `{image_base64,columns}`; return image and dimensions |
| GET, POST | `/setups` | List or save `{name,image_base64,columns,anchors}` |
| POST | `/sessions` | Create isolated verifier from `{setup_id,reference_id}` |
| POST | `/sessions/{id}/frame` | Inspect `{image_base64,mode,frame_id}` |
| DELETE | `/sessions/{id}` | End a session and clear its verifier |
| GET | `/history` | Saved inspection summaries |
| GET | `/inspections/{id}` | Saved result and annotated JPEG |

Example reference:

```json
{
  "name": "One wire between strips 1 and 5",
  "parts": [{"id":"W1","type":"wire","pins":{"0":"a1","1":"a5"}}],
  "terminals": {}
}
```

Calibration anchors are measured in the **returned rectified image**, while
`image_base64` in the save request must still be the **original uploaded photo**.
For a 30-column board, the four anchor names are `a1`, `a30`, `j30`, `j1`.

Treat `status`, `passed`, `candidate_status`, `stable_frames`,
`required_stable_frames`, `observed_at_unix_ms`, and `valid_for_ms` distinctly.
`candidate_status: CORRECT` on a still image is not a confirmed live pass. Do not
render green merely because a candidate matches, a model score is high, or the
last saved inspection was green. Scores are model outputs, not calibrated
probabilities of correct wiring. History describes conditions at capture time.

## Moving the interface to Base44 and Paper

This is a working local frontend and Python API. It has **not** created or
published a Base44 app or Paper design. The four screens and API contract can be
used to build that interface while keeping inference on the Mac or a separately
deployed Python service. Paper handles the visual design; Base44 can own project,
reference, and inspection metadata. Keep image inference and session stability in
this service so each request uses the trained model and measured grid.

The Connection settings control accepts a separate API base address. For another
frontend origin, configure an explicit allowlist before starting the backend:

```sh
SHERLOCK_ALLOWED_ORIGINS="https://your-frontend.example" \
SHERLOCK_DATA_DIR="$PWD/.local-data" \
  .venv/bin/python -B -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Use the actual frontend origin, without a path. A hosted frontend needs a backend
it can reach over an appropriate secure connection; pointing it at this Mac's
localhost is not a deployment. The current API is for local development and has
no authentication. Add authentication and per-user ownership before public
hosting; CORS alone does not provide access control. No Base44 credentials or
remote database changes are required for the local prototype.

## Verification and remaining validation

```sh
SHERLOCK_DATA_DIR=/private/tmp/sherlock-tests \
  .venv/bin/python -B -m pytest -q
node --test app/static/tests/*.test.mjs
```

Backend tests cover real ArUco calibration with synthetic board fixtures and
injected endpoint predictions for HTTP/session behavior. Frontend tests cover
freshness, still-image handling, reference validation, and existing dashboard
logic. These tests verify implementation behavior, not neural accuracy.

Browser checks exercised reference creation, real-weight photo inference, saved
history, and four-point calibration with an explicitly synthetic fixture.
Physical camera capture and correct/wrong wiring on the user's board remain to
be tested. The detector supports visible wires/resistors only; electrical
continuity, resistor values, LEDs, ICs, and power rails remain outside this UI's
verified scope. See the experiment model card for dataset and accuracy limits.
