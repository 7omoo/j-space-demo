from jspace_demo import analysis, jobs

import threading
import time
from contextlib import contextmanager


def wait(queue: jobs.Jobs, job_id: str, timeout: float = 5.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = queue.get(job_id)
        if job["status"] in ("done", "error", "cancelled"):
            return job
        time.sleep(0.01)
    raise AssertionError(queue.get(job_id))


def test_jobs_run_one_at_a_time_in_order():
    running, finished, lock, gate, started = [], [], threading.Lock(), threading.Event(), threading.Event()

    def run(params, step):
        with lock:
            running.append(params["n"])
            assert len(running) == 1  # never two on the model at once
        step("generating")
        if params["n"] == 0:
            started.set()
            gate.wait(2)
        finished.append(params["n"])
        with lock:
            running.remove(params["n"])
        return f"analysis-{params['n']}"

    queue = jobs.Jobs(run)
    first, second = queue.submit({"n": 0}), queue.submit({"n": 1})
    assert started.wait(2)
    assert (queue.get(first["id"])["status"], queue.get(first["id"])["step"]) == ("running", "generating")
    assert (queue.get(second["id"])["status"], queue.get(second["id"])["ahead"]) == ("queued", 1)
    gate.set()
    assert wait(queue, second["id"])["result"] == "analysis-1"
    assert queue.get(first["id"])["result"] == "analysis-0" and finished == [0, 1]


def test_an_error_is_reported_and_the_worker_goes_on():
    def run(params, step):
        if params["fail"]:
            raise RuntimeError("the model is not ready")
        return "ok"

    queue = jobs.Jobs(run)
    bad, good = queue.submit({"fail": True}), queue.submit({"fail": False})
    assert wait(queue, bad["id"])["error"] == {"code": "internal", "params": {}, "message": "the model is not ready"}
    assert wait(queue, good["id"])["status"] == "done"


def test_only_an_input_error_is_worded_as_one():
    class Odd(Exception):
        code = 404  # an error that happens to have a code is still internal

    def run(params, step):
        raise Odd("not found")

    queue = jobs.Jobs(run)
    assert wait(queue, queue.submit({})["id"])["error"]["code"] == "internal"


def test_an_input_error_keeps_its_code_for_the_page():
    def run(params, step):
        raise analysis.InputError("too_long", n_tokens=1100, limit=1024)

    queue = jobs.Jobs(run)
    error = wait(queue, queue.submit({})["id"])["error"]
    assert (error["code"], error["params"]) == ("too_long", {"n_tokens": 1100, "limit": 1024})
    assert error["message"] == "the message is too long (1100 tokens; the limit is 1024)"


def test_only_a_queued_job_can_be_cancelled():
    gate, started = threading.Event(), threading.Event()
    queue = jobs.Jobs(lambda params, step: started.set() or (gate.wait(2) and "ok"))
    first, second = queue.submit({}), queue.submit({})
    assert started.wait(2)
    assert queue.cancel(second["id"]) and not queue.cancel(first["id"])
    gate.set()
    assert wait(queue, first["id"])["status"] == "done" and queue.get(second["id"])["status"] == "cancelled"
    assert queue.get("nope") is None and not queue.cancel("nope")


def test_a_run_holds_the_gpu():
    events = []

    @contextmanager
    def gpu():
        events.append("lock")
        yield
        events.append("release")

    queue = jobs.Jobs(lambda params, step: events.append("run") or "ok", gpu=gpu)
    wait(queue, queue.submit({})["id"])
    assert events == ["lock", "run", "release"]


def test_the_plan_is_kept_on_the_job():
    queue = jobs.Jobs(lambda params, step: "x")
    job = queue.submit({}, plan=["generating", "reading"])
    assert job["plan"] == ["generating", "reading"]
    assert wait(queue, job["id"])["plan"] == ["generating", "reading"]
    assert queue.submit({})["plan"] == []
