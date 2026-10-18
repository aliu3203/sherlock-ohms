"""Sherlock Ohms FastAPI backend.

Exposes inspection, reference management, calibration, and SSE progress
endpoints. The real :class:`SherlockOhmsPipeline` from ``pipeline.py`` is
constructed lazily per reference and shared across inspections. Hardware is
optional: without a camera we fall back to the synthetic ArUco scene, and the
traffic light uses the terminal when GPIO is unavailable.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Literal

import cv2
import numpy as np
from fastapi import Body, FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

# Ensure the project root is importable when the module is loaded via uvicorn.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from engine.schemas import ReferenceCircuit  # noqa: E402
from engine.snapping import snap_detections  # noqa: E402
from engine.verifier import verify_circuit  # noqa: E402
from vision_ml.pi_hardware import BreadboardCamera, TrafficLight  # noqa: E402
from vision_ml.registration import (  # noqa: E402
    BreadboardRegistration,
    QualityGateError,
    QualityReport,
    generate_synthetic_aruco_board_image,
)

from app.background_worker import BackgroundWorker, ProgressBroker  # noqa: E402
from app.storage import (  # noqa: E402
    INSPECTION_ABORTED,
    INSPECTION_DONE,
    INSPECTION_ERROR,
    INSPECTION_RUNNING,
    VALID_LAYERS,
    InspectionRecord,
    Storage,
)

LOGGER = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Request/response models
# -----------------------------------------------------------------------------

class InspectRequest(BaseModel):
    reference_id: str
    mode: Literal["placement", "connectivity"] = "connectivity"


class InspectResponse(BaseModel):
    inspection_id: str
    status: str
    reference_id: str
    mode: str


class ReferenceCreateFromInspection(BaseModel):
    name: str = Field(min_length=1)


# -----------------------------------------------------------------------------
# App factory
# -----------------------------------------------------------------------------

PipelineFactory = Callable[["ReferenceCircuit"], Any]
InspectionRunner = Callable[[FastAPI, str, str, str], None]


def create_app(
    *,
    storage: Storage | None = None,
    broker: ProgressBroker | None = None,
    worker: BackgroundWorker | None = None,
    camera: BreadboardCamera | None = None,
    traffic_light: TrafficLight | None = None,
    pipeline_factory: PipelineFactory | None = None,
    inspection_runner: InspectionRunner | None = None,
    model_path: str | os.PathLike | None = None,
) -> FastAPI:
    """Build the FastAPI application. Hooks let tests inject fakes."""
    broker = broker or ProgressBroker()
    worker = worker or BackgroundWorker(broker=broker)
    storage = storage or Storage()
    traffic_light = traffic_light or TrafficLight(force_fallback=True)

    env_model = os.environ.get("SHERLOCK_MODEL_PATH")
    if model_path is None and env_model:
        model_path = env_model

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            worker.shutdown()
            try:
                traffic_light.close()
            except Exception:
                pass

    app = FastAPI(title="Sherlock Ohms", version="0.1.0", lifespan=lifespan)
    app.state.storage = storage
    app.state.broker = broker
    app.state.worker = worker
    app.state.camera = camera
    app.state.traffic_light = traffic_light
    app.state.pipeline_factory = pipeline_factory or _default_pipeline_factory(
        camera=camera,
        traffic_light=traffic_light,
        model_path=model_path,
    )
    app.state.pipeline_cache = {}
    app.state.inspection_runner = inspection_runner or _default_inspection_runner

    _register_routes(app)
    from app.hole_api import register_live_api
    register_live_api(app)
    allowed_origins = [origin.strip() for origin in os.environ.get(
        "SHERLOCK_ALLOWED_ORIGINS", ""
    ).split(",") if origin.strip() and origin.strip() != "*"]
    if allowed_origins:
        from fastapi.middleware.cors import CORSMiddleware
        app.add_middleware(CORSMiddleware, allow_origins=allowed_origins,
                           allow_methods=["GET", "POST", "DELETE"],
                           allow_headers=["Content-Type"])
    return app


# -----------------------------------------------------------------------------
# Pipeline factory + runner
# -----------------------------------------------------------------------------

def _default_pipeline_factory(
    *,
    camera: BreadboardCamera | None,
    traffic_light: TrafficLight,
    model_path: str | os.PathLike | None,
) -> PipelineFactory:
    """Return a callable that builds a SherlockOhmsPipeline for a reference."""

    def build(reference: ReferenceCircuit):
        # Import lazily so a missing model file doesn't break app import.
        from pipeline import SherlockOhmsPipeline

        return SherlockOhmsPipeline(
            reference=reference,
            model_path=model_path,
            camera=camera,
            traffic_light=traffic_light,
        )

    return build


def _get_or_create_pipeline(app: FastAPI, reference_id: str, reference: ReferenceCircuit):
    cache = app.state.pipeline_cache
    if reference_id not in cache:
        cache[reference_id] = app.state.pipeline_factory(reference)
    return cache[reference_id]


def _quality_to_dict(quality: QualityReport | None) -> dict[str, Any] | None:
    if quality is None:
        return None
    return {
        "passed": quality.passed,
        "markers_ok": quality.markers_ok,
        "sharpness": _finite_or_none(quality.sharpness),
        "brightness": _finite_or_none(quality.brightness),
        "glare_fraction": _finite_or_none(quality.glare_fraction),
        "failures": list(quality.failures),
    }


def _finite_or_none(value: float) -> float | None:
    try:
        return float(value) if np.isfinite(value) else None
    except Exception:
        return None


def _default_inspection_runner(app: FastAPI, inspection_id: str, reference_id: str, mode: str) -> None:
    storage: Storage = app.state.storage
    broker: ProgressBroker = app.state.broker
    traffic_light: TrafficLight = app.state.traffic_light

    try:
        storage.update_inspection_status(inspection_id, INSPECTION_RUNNING)
        reference_record = storage.get_reference(reference_id)
        if reference_record is None:
            raise LookupError(f"Reference {reference_id!r} not found")
        reference = ReferenceCircuit(**reference_record["data"])
        pipeline = _get_or_create_pipeline(app, reference_id, reference)
        try:
            traffic_light.set("yellow")
        except Exception as exc:
            LOGGER.debug("TrafficLight set yellow failed: %s", exc)

        broker.publish(inspection_id, "capturing")
        raw = (
            pipeline.camera.capture_burst()
            if pipeline.camera is not None
            else generate_synthetic_aruco_board_image()
        )
        if raw.ndim == 2:
            raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
        storage.save_image(inspection_id, "raw", raw)

        broker.publish(inspection_id, "registering")
        try:
            registration = pipeline.registration.register(raw)
        except QualityGateError as exc:
            quality = exc.result.quality
            if exc.result.rectified is not None:
                storage.save_image(inspection_id, "rectified", exc.result.rectified)
            data = {
                "mode": mode,
                "quality": _quality_to_dict(quality),
                "findings": [
                    {
                        "code": "UNCERTAIN",
                        "severity": "critical",
                        "message": "Quality gate failed: " + ", ".join(quality.failures),
                        "part_id": None,
                        "holes": [],
                        "expected_holes": [],
                        "bbox": None,
                    }
                ],
                "snapped_parts": [],
                "actual_netlist": {},
            }
            storage.finalize_inspection(
                inspection_id,
                status=INSPECTION_ABORTED,
                passed=False,
                light_color="RED",
                data=data,
                error=str(exc),
            )
            try:
                traffic_light.set("red")
            except Exception:
                pass
            broker.publish(
                inspection_id,
                "done",
                {"passed": False, "light_color": "RED", "status": INSPECTION_ABORTED},
            )
            return
        rectified = registration.rectified
        storage.save_image(inspection_id, "rectified", rectified)

        broker.publish(inspection_id, "detecting")
        detections_raw = pipeline.detector.detect(rectified)
        from pipeline import SherlockOhmsPipeline  # late import to avoid cycle

        validated = SherlockOhmsPipeline._raw_detections(detections_raw)

        broker.publish(inspection_id, "checking")
        snapped, extra_findings = snap_detections(
            validated, pipeline.lut, max_snap_px=pipeline.max_snap_px
        )
        reference_for_mode = (
            pipeline.placement_reference
            if mode == "placement"
            else pipeline.connectivity_reference
        )
        if reference_for_mode is None:
            raise ValueError(
                f"Reference {reference_id!r} does not define a {mode} view"
            )
        verdict = verify_circuit(
            snapped,
            reference=reference_for_mode,
            board=pipeline.board,
            rail_bindings=pipeline.rail_bindings,
            mode=mode,
            extra_findings=extra_findings,
        )
        from pipeline import PipelineVerdict  # late import

        pv = PipelineVerdict(**verdict.model_dump(), raw_detections=validated)
        annotated = pipeline.render_annotated_image(rectified, pv, pipeline.lut)
        storage.save_image(inspection_id, "annotated", annotated)

        data = verdict.model_dump()
        data["mode"] = mode
        data["quality"] = _quality_to_dict(registration.quality)
        storage.finalize_inspection(
            inspection_id,
            status=INSPECTION_DONE,
            passed=verdict.passed,
            light_color=verdict.light_color,
            data=data,
        )
        try:
            traffic_light.set(verdict.light_color.lower())
        except Exception:
            pass
        broker.publish(
            inspection_id,
            "done",
            {
                "passed": verdict.passed,
                "light_color": verdict.light_color,
                "status": INSPECTION_DONE,
            },
        )
    except Exception as exc:
        LOGGER.exception("Inspection %s failed", inspection_id)
        try:
            traffic_light.set("red")
        except Exception:
            pass
        storage.finalize_inspection(
            inspection_id,
            status=INSPECTION_ERROR,
            passed=False,
            light_color="RED",
            data={"mode": mode},
            error=str(exc),
        )
        broker.publish(inspection_id, "error", {"message": str(exc)})


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------

def _register_routes(app: FastAPI) -> None:
    static_dir = Path(__file__).resolve().parent / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse("/live", status_code=status.HTTP_307_TEMPORARY_REDIRECT)

    @app.get("/classic", response_class=FileResponse, include_in_schema=False)
    def dashboard() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.get("/live", response_class=FileResponse, include_in_schema=False)
    def live_dashboard() -> FileResponse:
        return FileResponse(static_dir / "live.html")

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "timestamp": time.time()}

    # -- Inspections ----------------------------------------------------------

    @app.post("/api/inspect", response_model=InspectResponse)
    def start_inspection(req: InspectRequest) -> InspectResponse:
        storage: Storage = app.state.storage
        if storage.get_reference(req.reference_id) is None:
            raise HTTPException(status_code=404, detail="Reference not found")
        inspection_id = str(uuid.uuid4())
        storage.create_inspection(
            inspection_id, reference_id=req.reference_id, mode=req.mode
        )
        runner: InspectionRunner = app.state.inspection_runner
        worker: BackgroundWorker = app.state.worker
        worker.submit(
            inspection_id,
            lambda iid, broker, _app=app, _ref=req.reference_id, _mode=req.mode: runner(
                _app, iid, _ref, _mode
            ),
        )
        return InspectResponse(
            inspection_id=inspection_id,
            status="pending",
            reference_id=req.reference_id,
            mode=req.mode,
        )

    @app.get("/api/inspections")
    def list_inspections(limit: int = 100) -> dict[str, Any]:
        storage: Storage = app.state.storage
        records = storage.list_inspections(limit=limit)
        return {"inspections": [_inspection_summary(r) for r in records]}

    @app.get("/api/inspections/{inspection_id}")
    def get_inspection(inspection_id: str) -> dict[str, Any]:
        storage: Storage = app.state.storage
        record = storage.get_inspection(inspection_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Inspection not found")
        return record.to_dict()

    @app.get("/api/inspections/{inspection_id}/image")
    def get_inspection_image(inspection_id: str, layer: str = "annotated") -> Response:
        storage: Storage = app.state.storage
        if layer not in VALID_LAYERS:
            raise HTTPException(
                status_code=400,
                detail=f"layer must be one of {list(VALID_LAYERS)}",
            )
        record = storage.get_inspection(inspection_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Inspection not found")
        blob = storage.read_image_bytes(inspection_id, layer)
        if blob is None:
            raise HTTPException(status_code=404, detail=f"No {layer} image for this inspection")
        return Response(content=blob, media_type="image/png")

    # -- SSE progress ---------------------------------------------------------

    @app.get("/api/events/{inspection_id}")
    async def inspection_events(inspection_id: str, request: Request) -> EventSourceResponse:
        storage: Storage = app.state.storage
        if storage.get_inspection(inspection_id) is None:
            raise HTTPException(status_code=404, detail="Inspection not found")
        broker: ProgressBroker = app.state.broker
        broker.register(inspection_id)

        async def event_stream():
            emitted = 0
            while True:
                if await request.is_disconnected():
                    return
                history = broker.history(inspection_id)
                while emitted < len(history):
                    event = history[emitted]
                    emitted += 1
                    yield {"event": event.stage, "data": json.dumps(event.to_dict())}
                    if event.stage in {"done", "error", "aborted"}:
                        return
                if broker.is_terminal(inspection_id):
                    return
                # A restarted process has persisted results but no in-memory
                # broker history. Replay its terminal state rather than hang.
                record = storage.get_inspection(inspection_id)
                if record is not None and record.status in {INSPECTION_DONE, INSPECTION_ERROR, INSPECTION_ABORTED}:
                    stage = "error" if record.status == INSPECTION_ERROR else "done"
                    yield {"event": stage, "data": json.dumps({
                        "inspection_id": inspection_id, "stage": stage,
                        "status": record.status, "passed": record.passed,
                        "light_color": record.light_color, "message": record.error,
                    })}
                    return
                await asyncio.sleep(0.1)

        return EventSourceResponse(event_stream())

    # -- References -----------------------------------------------------------

    @app.get("/api/references")
    def list_references() -> dict[str, Any]:
        storage: Storage = app.state.storage
        return {"references": storage.list_references()}

    @app.post("/api/references", status_code=status.HTTP_201_CREATED)
    def create_reference(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        try:
            # Validate shape via Pydantic before persisting.
            parsed = ReferenceCircuit(**body)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid reference: {exc}")
        storage: Storage = app.state.storage
        return storage.save_reference(parsed.model_dump())

    @app.post("/api/references/from-inspection/{inspection_id}", status_code=status.HTTP_201_CREATED)
    def reference_from_inspection(
        inspection_id: str,
        body: ReferenceCreateFromInspection,
    ) -> dict[str, Any]:
        storage: Storage = app.state.storage
        record = storage.get_inspection(inspection_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Inspection not found")
        if record.status != INSPECTION_DONE:
            raise HTTPException(
                status_code=400,
                detail="Can only save a reference from a completed inspection",
            )
        snapped = record.data.get("snapped_parts") or []
        parts: list[dict[str, Any]] = []
        for part in snapped:
            parts.append(
                {
                    "id": part["id"],
                    "type": part["type"],
                    "pins": dict(part.get("pins", {})),
                }
            )
        source_reference = storage.get_reference(record.reference_id) if record.reference_id else None
        rail_bindings = (source_reference["data"]["rails"] if source_reference
                         else {"VCC": "top_plus", "GND": "top_minus"})
        ref_data = {
            "name": body.name,
            "rails": rail_bindings,
            "parts": parts,
        }
        # Round-trip through the model to validate.
        parsed = ReferenceCircuit(**ref_data)
        return storage.save_reference(parsed.model_dump())

    # -- Calibration ----------------------------------------------------------

    @app.post("/api/calibrate")
    def calibrate() -> dict[str, Any]:
        storage: Storage = app.state.storage
        camera: BreadboardCamera | None = app.state.camera
        raw = (
            camera.capture_burst()
            if camera is not None
            else generate_synthetic_aruco_board_image()
        )
        if raw.ndim == 2:
            raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
        reg = BreadboardRegistration()
        try:
            result = reg.register(raw, enforce_quality=False)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Calibration failed: {exc}")
        centers = {
            str(marker_id): [float(x), float(y)]
            for marker_id, (x, y) in result.marker_centers.items()
        }
        data = {
            "quality": _quality_to_dict(result.quality),
            "marker_centers": centers,
            "has_homography": result.homography is not None,
        }
        return storage.save_calibration(data)

    @app.get("/api/calibration")
    def get_calibration() -> dict[str, Any]:
        storage: Storage = app.state.storage
        cal = storage.get_calibration()
        if cal is None:
            raise HTTPException(status_code=404, detail="No calibration on file")
        return cal

    # -- Preview --------------------------------------------------------------

    @app.get("/api/preview")
    def preview() -> Response:
        camera: BreadboardCamera | None = app.state.camera
        raw = (
            camera.capture_burst()
            if camera is not None
            else generate_synthetic_aruco_board_image()
        )
        if raw.ndim == 2:
            raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
        h, w = raw.shape[:2]
        target_w = 640
        if w == 0:
            raise HTTPException(status_code=500, detail="Empty preview frame")
        scale = target_w / w
        thumb = cv2.resize(raw, (target_w, max(1, int(h * scale))))
        ok, buf = cv2.imencode(".png", thumb)
        if not ok:
            raise HTTPException(status_code=500, detail="Could not encode preview")
        return Response(content=buf.tobytes(), media_type="image/png")


def _inspection_summary(record: InspectionRecord) -> dict[str, Any]:
    data = record.to_dict()
    # Trim verbose fields for the list endpoint.
    data.pop("snapped_parts", None)
    data.pop("actual_netlist", None)
    data.pop("findings", None)
    data.pop("raw_detections", None)
    return data


# -----------------------------------------------------------------------------
# Default app instance (used by uvicorn)
# -----------------------------------------------------------------------------

app = create_app()
