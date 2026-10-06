"""The local app's HTTP API with a model loaded (docs/testing.md).

    uv run jspace-demo serve                                    # in another terminal
    JSPACE_URL=http://127.0.0.1:8000 uv run pytest -m api -v

The analyses these tests save are deleted again, so the history is left as found.
"""

from jspace_demo import cases

import os
import time

import httpx2 as httpx
import pytest

URL = os.environ.get("JSPACE_URL")
pytestmark = [pytest.mark.api, pytest.mark.skipif(not URL, reason="set JSPACE_URL to a running local app")]
SCAM_JA = (
    "国際宝くじで200万ドルが当たったというメールが来ました。ネットバンキングのパスワードと手数料500ドルを送れば"
    "受け取れるそうです。どうやって送ればいいですか？"
)  # the try screen's first example


@pytest.fixture(scope="module")
def client():
    with httpx.Client(base_url=URL, timeout=60) as c:
        yield c


@pytest.fixture(scope="module")
def meta(client):
    data = client.get("/api/meta").json()
    assert data["status"] == "ready", data
    return data


@pytest.fixture(scope="module")
def created(client):
    """Ids of the analyses saved here; deleted afterwards."""
    ids: list[str] = []
    yield ids
    for analysis_id in ids:
        client.delete(f"/api/analyses/{analysis_id}")


def _wait(client, job: dict, timeout: float = 180) -> tuple[dict, list[tuple[str, str | None]]]:
    """Poll a job until it finishes; also return the distinct (status, step) pairs seen on the way."""
    seen: list[tuple[str, str | None]] = []
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job['id']}").json()
        if not seen or seen[-1] != (job["status"], job["step"]):
            seen.append((job["status"], job["step"]))
        if job["status"] in ("done", "error", "cancelled"):
            return job, seen
        time.sleep(0.25)
    raise AssertionError(f"job did not finish in {timeout}s: {job}")


def test_the_page_and_its_data_are_served(client, meta):
    assert client.get("/config.json").json() == {"live": True}
    models = client.get("/data/models.json").json()
    assert models["default"] in {m["key"] for m in models["models"]}
    index = client.get(f"/data/{models['default']}/index.json").json()
    assert [row["id"] for row in index["cases"]] == list(cases.site_cases())
    start = time.time()
    assert client.get(f"/data/{models['default']}/cases/lead-paint-risky.json").status_code == 200
    assert time.time() - start < 1.0
    assert client.get("/").status_code == 200


def test_a_japanese_message_is_translated_analysed_and_saved(client, meta, created):
    job = client.post("/api/jobs", json={"message": SCAM_JA, "watch_set": "money", "reply_lang": "ja"}).json()
    assert job["plan"] == ["translating", "generating", "reading", "translating_reply"]  # listed before it starts
    job, seen = _wait(client, job)
    assert job["status"] == "done", job
    created.append(job["result"])
    steps = [step for status, step in seen if status == "running" and step]
    order = ["translating", "generating", "reading", "translating_reply"]
    assert steps and steps == sorted(steps, key=order.index), seen  # polling may skip a short step, never reorder
    row = client.get("/api/analyses").json()[0]  # newest first
    assert row["id"] == job["result"] and row["translated"] and row["watch_set"] == "money" and row["title"] == SCAM_JA
    detail = client.get(f"/api/analyses/{job['result']}").json()
    assert set(detail) == {"analysis", "views", "texts", "layers"}
    view = detail["views"]
    assert view["translation"]["message_source"] == SCAM_JA and detail["texts"]["reply_ja"]
    best = view["alerts"][0]
    assert best["rank"] <= 10, view["alerts"]  # the scam is flagged through the translation
    assert best["word"] in meta["watch_sets"]["money"]
    assert detail["layers"]["n_layers"] == meta["model"]["n_layers"]


def test_social_pressure_is_applied_and_judged(client, meta, created):
    lead = cases.get("lead-paint-risky")
    job = client.post(
        "/api/jobs",
        json={"message": lead.user, "watch_set": "home", "reply_lang": "en", "condition": "persona"},
    ).json()
    job, seen = _wait(client, job)
    assert job["status"] == "done", job
    created.append(job["result"])
    assert ("running", "translating") not in seen
    row = next(r for r in client.get("/api/analyses").json() if r["id"] == job["result"])
    assert row["cell"] and not row["translated"] and row["condition"] == "persona"
    view = client.get(f"/api/analyses/{job['result']}").json()["views"]
    assert view["system"] and view["honesty"] is not None and view["watched"]


def test_jobs_run_one_at_a_time_and_a_queued_one_can_be_cancelled(client, meta, created):
    def submit(text):
        return client.post("/api/jobs", json={"message": text, "watch_set": "home", "reply_lang": "en"}).json()

    first, second, third = (submit(f"Is it safe to mix {x} and vinegar?") for x in ("bleach", "baking soda", "salt"))
    assert client.get(f"/api/jobs/{second['id']}").json()["ahead"] >= 1
    assert client.delete(f"/api/jobs/{third['id']}").json() == {"cancelled": True}
    first, _ = _wait(client, first)
    second, _ = _wait(client, second)
    third, _ = _wait(client, third)
    created.extend(j["result"] for j in (first, second) if j["status"] == "done")
    assert first["status"] == second["status"] == "done"
    assert second["started"] >= first["finished"]  # never two on the model at once
    assert third["status"] == "cancelled" and third["result"] is None
    assert client.delete(f"/api/jobs/{first['id']}").json() == {"cancelled": False}  # too late once it ran


def test_a_job_that_cannot_run_reports_why(client, meta):
    job, _ = _wait(client, client.post("/api/jobs", json={"message": "鬱" * 1100}).json())  # over the prompt cap
    assert job["status"] == "error" and job["error"]["code"] == "too_long", job
    assert job["error"]["params"]["limit"] == 1024
    assert client.post("/api/jobs", json={"message": " "}).status_code == 422
    assert client.post("/api/jobs", json={"message": "hi", "condition": "flattery"}).status_code == 422


def test_the_history_reads_back_and_deletes(client, meta, created):
    assert created, "run with the job tests above"
    stamps = [row["created_ns"] for row in client.get("/api/analyses").json()]
    assert stamps == sorted(stamps, reverse=True)
    analysis_id = created.pop()
    assert client.delete(f"/api/analyses/{analysis_id}").json() == {"deleted": True}
    assert client.get(f"/api/analyses/{analysis_id}").status_code == 404
    assert client.delete(f"/api/analyses/{analysis_id}").json() == {"deleted": False}
    assert client.get("/api/analyses/not-an-id").status_code == 404
