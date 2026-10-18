"""SQLite-backed persistence for inspections, references, and calibration state.

Images (raw/rectified/annotated) are stored on disk alongside the DB under
``<root>/images/``. The DB holds the structured verdict JSON plus pointers.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)

ImageLayer = str  # "raw" | "rectified" | "annotated"
VALID_LAYERS: tuple[ImageLayer, ...] = ("raw", "rectified", "annotated")

INSPECTION_PENDING = "pending"
INSPECTION_RUNNING = "running"
INSPECTION_DONE = "done"
INSPECTION_ERROR = "error"
INSPECTION_ABORTED = "aborted"


# -----------------------------------------------------------------------------
# Seeded reference circuits
# -----------------------------------------------------------------------------

LED_BLINK_REFERENCE: dict[str, Any] = {
    "name": "LED Blink",
    "rails": {"VCC": "top_plus", "GND": "top_minus"},
    "parts": [
        {"id": "R1", "type": "resistor", "pins": {"0": "VCC", "1": "n_bridge"}},
        {"id": "LED1", "type": "led", "pins": {"anode": "n_bridge", "cathode": "GND"}},
    ],
}

SERIES_PARALLEL_REFERENCE: dict[str, Any] = {
    "name": "Series-Parallel LEDs",
    "rails": {"VCC": "top_plus", "GND": "top_minus"},
    "parts": [
        {"id": "R1", "type": "resistor", "pins": {"0": "VCC", "1": "nA"}},
        {"id": "LED1", "type": "led", "pins": {"anode": "nA", "cathode": "GND"}},
        {"id": "R2", "type": "resistor", "pins": {"0": "VCC", "1": "nB"}},
        {"id": "LED2", "type": "led", "pins": {"anode": "nB", "cathode": "GND"}},
    ],
}

SEED_REFERENCES: dict[str, dict[str, Any]] = {
    "led_blink": LED_BLINK_REFERENCE,
    "series_parallel_leds": SERIES_PARALLEL_REFERENCE,
}


# -----------------------------------------------------------------------------
# Storage
# -----------------------------------------------------------------------------


@dataclass
class InspectionRecord:
    id: str
    created_at: str
    reference_id: str | None
    mode: str
    status: str
    passed: bool | None
    light_color: str | None
    data: dict[str, Any]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "inspection_id": self.id,
            "timestamp": self.created_at,
            "reference_id": self.reference_id,
            "mode": self.mode,
            "status": self.status,
            "passed": self.passed,
            "light_color": self.light_color,
            "error": self.error,
            "findings": [],
            "snapped_parts": [],
            "actual_netlist": {},
            **self.data,
        }


class Storage:
    """Thread-safe SQLite store with on-disk image blobs."""

    def __init__(self, root: str | os.PathLike | None = None) -> None:
        root_path = Path(root) if root is not None else Path(os.environ.get(
            "SHERLOCK_DATA_DIR", Path.home() / ".sherlock_ohms"
        ))
        self.root = root_path
        self.images_dir = root_path / "images"
        self.db_path = root_path / "sherlock.db"
        self.root.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_db()
        self._seed_defaults()

    # -- Connection helpers ---------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS circuit_references (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    is_seed INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS inspections (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    reference_id TEXT,
                    mode TEXT NOT NULL,
                    status TEXT NOT NULL,
                    passed INTEGER,
                    light_color TEXT,
                    error TEXT,
                    data_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS inspections_created_idx
                    ON inspections(created_at DESC);
                CREATE TABLE IF NOT EXISTS calibration (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    data_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )

    def _seed_defaults(self) -> None:
        with self._lock, self._connect() as conn:
            for ref_id, ref_data in SEED_REFERENCES.items():
                existing = conn.execute(
                    "SELECT 1 FROM circuit_references WHERE id = ?", (ref_id,)
                ).fetchone()
                if existing is not None:
                    continue
                conn.execute(
                    "INSERT INTO circuit_references (id, name, data_json, created_at, is_seed)"
                    " VALUES (?, ?, ?, ?, 1)",
                    (ref_id, ref_data["name"], json.dumps(ref_data), _now_iso()),
                )

    # -- References -----------------------------------------------------------

    def list_references(self) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT id, name, data_json, created_at, is_seed FROM circuit_references"
                " ORDER BY is_seed DESC, created_at DESC"
            ).fetchall()
        return [self._row_to_reference(row) for row in rows]

    def get_reference(self, reference_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT id, name, data_json, created_at, is_seed FROM circuit_references"
                " WHERE id = ?",
                (reference_id,),
            ).fetchone()
        return self._row_to_reference(row) if row is not None else None

    def save_reference(
        self,
        data: dict[str, Any],
        *,
        reference_id: str | None = None,
        is_seed: bool = False,
    ) -> dict[str, Any]:
        if "name" not in data or "rails" not in data or "parts" not in data:
            raise ValueError("Reference must include name, rails, and parts")
        ref_id = reference_id or _slugify(data["name"]) or str(uuid.uuid4())
        with self._lock, self._connect() as conn:
            # Avoid id collisions by appending a counter.
            final_id, counter = ref_id, 1
            while conn.execute(
                "SELECT 1 FROM circuit_references WHERE id = ?", (final_id,)
            ).fetchone() is not None:
                counter += 1
                final_id = f"{ref_id}_{counter}"
            conn.execute(
                "INSERT INTO circuit_references (id, name, data_json, created_at, is_seed)"
                " VALUES (?, ?, ?, ?, ?)",
                (final_id, data["name"], json.dumps(data), _now_iso(), int(is_seed)),
            )
            row = conn.execute(
                "SELECT id, name, data_json, created_at, is_seed FROM circuit_references"
                " WHERE id = ?",
                (final_id,),
            ).fetchone()
        return self._row_to_reference(row)

    @staticmethod
    def _row_to_reference(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "name": row["name"],
            "created_at": row["created_at"],
            "is_seed": bool(row["is_seed"]),
            "data": json.loads(row["data_json"]),
        }

    # -- Inspections ----------------------------------------------------------

    def create_inspection(
        self,
        inspection_id: str,
        *,
        reference_id: str | None,
        mode: str,
    ) -> InspectionRecord:
        record = InspectionRecord(
            id=inspection_id,
            created_at=_now_iso(),
            reference_id=reference_id,
            mode=mode,
            status=INSPECTION_PENDING,
            passed=None,
            light_color=None,
            data={},
        )
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO inspections (id, created_at, reference_id, mode, status,"
                " passed, light_color, error, data_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record.id,
                    record.created_at,
                    record.reference_id,
                    record.mode,
                    record.status,
                    None,
                    None,
                    None,
                    "{}",
                ),
            )
        return record

    def update_inspection_status(self, inspection_id: str, status: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE inspections SET status = ? WHERE id = ?",
                (status, inspection_id),
            )

    def finalize_inspection(
        self,
        inspection_id: str,
        *,
        status: str,
        passed: bool | None,
        light_color: str | None,
        data: dict[str, Any],
        error: str | None = None,
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE inspections SET status = ?, passed = ?, light_color = ?,"
                " error = ?, data_json = ? WHERE id = ?",
                (
                    status,
                    None if passed is None else int(passed),
                    light_color,
                    error,
                    json.dumps(data),
                    inspection_id,
                ),
            )

    def get_inspection(self, inspection_id: str) -> InspectionRecord | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM inspections WHERE id = ?", (inspection_id,)
            ).fetchone()
        return self._row_to_inspection(row) if row is not None else None

    def list_inspections(self, limit: int = 100) -> list[InspectionRecord]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM inspections ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._row_to_inspection(row) for row in rows]

    @staticmethod
    def _row_to_inspection(row: sqlite3.Row) -> InspectionRecord:
        return InspectionRecord(
            id=row["id"],
            created_at=row["created_at"],
            reference_id=row["reference_id"],
            mode=row["mode"],
            status=row["status"],
            passed=None if row["passed"] is None else bool(row["passed"]),
            light_color=row["light_color"],
            error=row["error"],
            data=json.loads(row["data_json"] or "{}"),
        )

    # -- Images ---------------------------------------------------------------

    def image_path(self, inspection_id: str, layer: ImageLayer) -> Path:
        if layer not in VALID_LAYERS:
            raise ValueError(f"Unknown image layer {layer!r}")
        return self.images_dir / f"{inspection_id}_{layer}.png"

    def save_image(self, inspection_id: str, layer: ImageLayer, image: np.ndarray) -> Path:
        path = self.image_path(inspection_id, layer)
        if image.dtype != np.uint8 or image.ndim not in (2, 3):
            raise ValueError("save_image expects a uint8 image")
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"Failed to write image to {path}")
        return path

    def read_image_bytes(self, inspection_id: str, layer: ImageLayer) -> bytes | None:
        path = self.image_path(inspection_id, layer)
        if not path.exists():
            return None
        return path.read_bytes()

    # -- Calibration ----------------------------------------------------------

    def save_calibration(self, data: dict[str, Any]) -> dict[str, Any]:
        record = {"data": data, "updated_at": _now_iso()}
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO calibration (id, data_json, updated_at) VALUES (1, ?, ?)"
                " ON CONFLICT(id) DO UPDATE SET data_json = excluded.data_json,"
                " updated_at = excluded.updated_at",
                (json.dumps(data), record["updated_at"]),
            )
        return record

    def get_calibration(self) -> dict[str, Any] | None:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT data_json, updated_at FROM calibration WHERE id = 1").fetchone()
        if row is None:
            return None
        return {"data": json.loads(row["data_json"]), "updated_at": row["updated_at"]}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z"


def _slugify(name: str) -> str:
    allowed = "abcdefghijklmnopqrstuvwxyz0123456789_"
    out: list[str] = []
    for ch in name.strip().lower():
        if ch in allowed:
            out.append(ch)
        elif ch in " -\t":
            out.append("_")
    result = "".join(out).strip("_")
    while "__" in result:
        result = result.replace("__", "_")
    return result
