"""Backend tests covering /api/inspect, references, calibration, and history."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.storage import Storage


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def storage(tmp_path: Path) -> Storage:
    return Storage(root=tmp_path / "data")


@pytest.fixture
def client(storage: Storage):
    app = create_app(storage=storage)
    with TestClient(app) as c:
        yield c


def _wait_for_terminal(client: TestClient, inspection_id: str, timeout: float = 15.0) -> dict:
    """Poll until the inspection reaches a terminal status."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = client.get(f"/api/inspections/{inspection_id}")
        assert resp.status_code == 200
        body = resp.json()
        if body["status"] in ("done", "error", "aborted"):
            return body
        time.sleep(0.1)
    raise AssertionError(f"Inspection {inspection_id} did not finish within {timeout}s")


# -----------------------------------------------------------------------------
# References
# -----------------------------------------------------------------------------

def test_seeded_references_are_available(client: TestClient):
    resp = client.get("/api/references")
    assert resp.status_code == 200
    refs = resp.json()["references"]
    ids = {r["id"] for r in refs}
    assert "led_blink" in ids
    assert "series_parallel_leds" in ids
    led_blink = next(r for r in refs if r["id"] == "led_blink")
    assert led_blink["is_seed"] is True
    assert led_blink["data"]["name"] == "LED Blink"
    assert any(p["type"] == "led" for p in led_blink["data"]["parts"])


def test_upload_reference_appears_in_list(client: TestClient):
    payload = {
        "name": "Minimal Resistor",
        "rails": {"VCC": "top_plus", "GND": "top_minus"},
        "parts": [
            {"id": "R1", "type": "resistor", "pins": {"0": "VCC", "1": "GND"}},
        ],
    }
    resp = client.post("/api/references", json=payload)
    assert resp.status_code == 201, resp.text
    created = resp.json()
    assert created["name"] == "Minimal Resistor"
    assert created["is_seed"] is False
    assert created["data"]["parts"][0]["id"] == "R1"

    listing = client.get("/api/references").json()["references"]
    assert any(r["id"] == created["id"] for r in listing)


def test_rejects_invalid_reference(client: TestClient):
    resp = client.post("/api/references", json={"name": "bad"})
    assert resp.status_code == 400


# -----------------------------------------------------------------------------
# Inspection lifecycle
# -----------------------------------------------------------------------------

def test_inspect_against_seed_runs_and_records_history(client: TestClient):
    resp = client.post(
        "/api/inspect",
        json={"reference_id": "led_blink", "mode": "connectivity"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    inspection_id = body["inspection_id"]
    assert body["reference_id"] == "led_blink"
    assert body["mode"] == "connectivity"
    assert body["status"] == "pending"

    final = _wait_for_terminal(client, inspection_id)
    # Mock detector returns no parts; LED Blink reference expects two parts, so
    # we expect FAIL (RED) with MISSING_PART findings — but the pipeline must
    # have run end-to-end to that point.
    assert final["status"] == "done"
    assert final["light_color"] in ("GREEN", "RED")
    assert "findings" in final
    assert isinstance(final["snapped_parts"], list)

    history = client.get("/api/inspections").json()["inspections"]
    assert any(h["inspection_id"] == inspection_id for h in history)


def test_inspection_image_endpoints(client: TestClient):
    resp = client.post(
        "/api/inspect", json={"reference_id": "led_blink", "mode": "connectivity"}
    )
    inspection_id = resp.json()["inspection_id"]
    _wait_for_terminal(client, inspection_id)

    for layer in ("raw", "rectified", "annotated"):
        r = client.get(f"/api/inspections/{inspection_id}/image", params={"layer": layer})
        assert r.status_code == 200, f"layer={layer}: {r.status_code} {r.text}"
        assert r.headers["content-type"] == "image/png"
        assert len(r.content) > 100  # non-trivial PNG

    bad = client.get(f"/api/inspections/{inspection_id}/image", params={"layer": "bogus"})
    assert bad.status_code == 400


def test_inspect_with_unknown_reference_404s(client: TestClient):
    resp = client.post("/api/inspect", json={"reference_id": "nope", "mode": "connectivity"})
    assert resp.status_code == 404


def test_get_inspection_unknown_404s(client: TestClient):
    resp = client.get("/api/inspections/does-not-exist")
    assert resp.status_code == 404


# -----------------------------------------------------------------------------
# References from inspection
# -----------------------------------------------------------------------------

def test_reference_from_completed_inspection(client: TestClient):
    resp = client.post(
        "/api/inspect", json={"reference_id": "led_blink", "mode": "connectivity"}
    )
    inspection_id = resp.json()["inspection_id"]
    _wait_for_terminal(client, inspection_id)

    r = client.post(
        f"/api/references/from-inspection/{inspection_id}",
        json={"name": "Captured LED"},
    )
    assert r.status_code == 201, r.text
    created = r.json()
    assert created["name"] == "Captured LED"
    assert created["is_seed"] is False
    # Listed too.
    ids = {ref["id"] for ref in client.get("/api/references").json()["references"]}
    assert created["id"] in ids


# -----------------------------------------------------------------------------
# Calibration & preview
# -----------------------------------------------------------------------------

def test_calibrate_endpoint_persists(client: TestClient):
    resp = client.post("/api/calibrate")
    assert resp.status_code == 200, resp.text
    data = resp.json()["data"]
    assert "marker_centers" in data
    assert "quality" in data

    stored = client.get("/api/calibration")
    assert stored.status_code == 200
    assert "data" in stored.json()


def test_preview_returns_png(client: TestClient):
    resp = client.get("/api/preview")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert len(resp.content) > 100


# -----------------------------------------------------------------------------
# SSE endpoint existence
# -----------------------------------------------------------------------------

def test_sse_endpoint_streams_events(client: TestClient):
    resp = client.post(
        "/api/inspect", json={"reference_id": "led_blink", "mode": "connectivity"}
    )
    inspection_id = resp.json()["inspection_id"]
    _wait_for_terminal(client, inspection_id)

    # After the inspection finished, the SSE endpoint still works: it replays
    # history and closes once it hits the terminal event.
    with client.stream("GET", f"/api/events/{inspection_id}") as stream:
        assert stream.status_code == 200
        assert "text/event-stream" in stream.headers.get("content-type", "")
        chunks = []
        for raw in stream.iter_lines():
            chunks.append(raw)
            if len(chunks) > 60:
                break
    text = "\n".join(chunks)
    # At least the terminal event should appear.
    assert "done" in text or "error" in text
