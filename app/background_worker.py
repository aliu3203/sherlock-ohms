"""Background inspection worker + SSE progress broker.

A single-threaded executor runs inspections one at a time (the pipeline uses
OpenCV / torch under the hood, so parallel runs would just thrash). Progress
events are published to a per-inspection thread-safe queue; the FastAPI SSE
endpoint subscribes and streams them to the browser.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

LOGGER = logging.getLogger(__name__)

STAGES: tuple[str, ...] = (
    "capturing",
    "registering",
    "detecting",
    "checking",
    "done",
)
TERMINAL_STAGES: frozenset[str] = frozenset({"done", "error", "aborted"})


@dataclass
class ProgressEvent:
    inspection_id: str
    stage: str
    timestamp: float = field(default_factory=time.time)
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "inspection_id": self.inspection_id,
            "stage": self.stage,
            "timestamp": self.timestamp,
            **self.detail,
        }


class ProgressBroker:
    """Thread-safe fan-out of inspection progress events."""

    def __init__(self) -> None:
        self._queues: dict[str, queue.Queue[ProgressEvent | None]] = {}
        self._history: dict[str, list[ProgressEvent]] = {}
        self._terminal: set[str] = set()
        self._lock = threading.Lock()

    def register(self, inspection_id: str) -> None:
        with self._lock:
            self._queues.setdefault(inspection_id, queue.Queue())
            self._history.setdefault(inspection_id, [])

    def publish(
        self,
        inspection_id: str,
        stage: str,
        detail: dict[str, Any] | None = None,
    ) -> ProgressEvent:
        event = ProgressEvent(inspection_id, stage, detail=dict(detail or {}))
        with self._lock:
            self._history.setdefault(inspection_id, []).append(event)
            q = self._queues.setdefault(inspection_id, queue.Queue())
            if stage in TERMINAL_STAGES:
                self._terminal.add(inspection_id)
        q.put(event)
        if stage in TERMINAL_STAGES:
            # Sentinel so sync subscribers release.
            q.put(None)
        return event

    def history(self, inspection_id: str) -> list[ProgressEvent]:
        with self._lock:
            return list(self._history.get(inspection_id, []))

    def is_terminal(self, inspection_id: str) -> bool:
        with self._lock:
            return inspection_id in self._terminal

    def subscribe(
        self,
        inspection_id: str,
        *,
        poll_interval: float = 0.25,
    ):
        """Blocking generator yielding events for an inspection id."""
        self.register(inspection_id)
        # Replay history so late subscribers still see earlier stages.
        for past in self.history(inspection_id):
            yield past
        if self.is_terminal(inspection_id):
            return
        q = self._queues[inspection_id]
        seen = len(self.history(inspection_id))
        while True:
            try:
                event = q.get(timeout=poll_interval)
            except queue.Empty:
                if self.is_terminal(inspection_id):
                    break
                continue
            if event is None:
                break
            # Skip events we already replayed.
            if seen > 0:
                seen -= 1
                continue
            yield event
            if event.stage in TERMINAL_STAGES:
                break


InspectionJob = Callable[[str, "ProgressBroker"], None]


class BackgroundWorker:
    """Single-worker executor serializing inspection jobs."""

    def __init__(self, broker: ProgressBroker | None = None) -> None:
        self.broker = broker or ProgressBroker()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sherlock-worker")
        self._futures: dict[str, Future] = {}
        self._lock = threading.Lock()

    def submit(self, inspection_id: str, job: InspectionJob) -> Future:
        self.broker.register(inspection_id)

        def _runner() -> None:
            try:
                job(inspection_id, self.broker)
            except Exception as exc:
                LOGGER.exception("Inspection %s crashed", inspection_id)
                self.broker.publish(
                    inspection_id,
                    "error",
                    {"message": str(exc), "trace": traceback.format_exc()},
                )

        future = self._executor.submit(_runner)
        with self._lock:
            self._futures[inspection_id] = future
        return future

    def wait(self, inspection_id: str, timeout: float | None = None) -> None:
        with self._lock:
            future = self._futures.get(inspection_id)
        if future is None:
            return
        future.result(timeout=timeout)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
