"""Analyses requested from the page, run one at a time on a worker thread: the model is never shared.

The page submits a job, then polls it: queued (with how many are ahead) -> running (with the current step of its
plan) -> done (with the saved analysis id), error (with a code the page can word) or cancelled (only while queued).
"""

from __future__ import annotations

from jspace_demo.errors import InputError

import logging
import queue
import threading
import time
import uuid
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext

KEEP_SECONDS = 3600  # finished jobs are forgotten after an hour; their results stay in the history

log = logging.getLogger(__name__)


class Jobs:
    def __init__(
        self, run: Callable[[dict, Callable[[str], None]], str], gpu: Callable[[], AbstractContextManager] = nullcontext
    ):
        self._run, self._gpu = run, gpu
        self._queue: queue.Queue[tuple[str, dict]] = queue.Queue()
        self._jobs: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None

    def submit(self, params: dict, plan: list[str] | None = None) -> dict:
        """Queue a job. ``plan`` names the steps it will report, so the page can list them all from the start."""
        job_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._forget_old()
            self._jobs[job_id] = {
                "id": job_id,
                "status": "queued",
                "step": None,
                "plan": list(plan or []),
                "result": None,
                "error": None,
                "created": time.time(),
                "started": None,
                "finished": None,
            }
            self._queue.put((job_id, params))  # under the lock: the queue's order is the order "ahead" counts
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(target=self._loop, daemon=True, name="jspace-jobs")
                self._worker.start()
        return self.get(job_id)

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            ahead = sum(
                1 for j in self._jobs.values() if j["status"] in ("queued", "running") and j["created"] < job["created"]
            )
            return {**job, "ahead": ahead if job["status"] == "queued" else 0}

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job["status"] != "queued":
                return False
            job.update(status="cancelled", finished=time.time())
            return True

    def _loop(self) -> None:
        while True:
            job_id, params = self._queue.get()
            with self._lock:
                job = self._jobs.get(job_id)
                if job is None or job["status"] != "queued":
                    continue
                job.update(status="running", started=time.time())

            def step(name: str, job: dict = job) -> None:
                with self._lock:
                    job["step"] = name

            try:
                with self._gpu():
                    outcome = {"status": "done", "result": self._run(params, step)}
            except Exception as error:  # the page shows it; the worker keeps serving
                if not isinstance(error, InputError):
                    log.exception("job %s failed", job_id)
                outcome = {"status": "error", "error": describe(error)}
            with self._lock:
                job.update(outcome, step=None, finished=time.time())

    def _forget_old(self) -> None:
        cutoff = time.time() - KEEP_SECONDS
        for job_id in [k for k, j in self._jobs.items() if j["finished"] and j["finished"] < cutoff]:
            del self._jobs[job_id]


def describe(error: Exception) -> dict:
    """An error as the page can word it: the code and parameters of an input the user can fix (``InputError``),
    otherwise "internal" with the message."""
    if isinstance(error, InputError):
        return {"code": error.code, "params": dict(error.params), "message": str(error)}
    return {"code": "internal", "params": {}, "message": str(error) or type(error).__name__}
