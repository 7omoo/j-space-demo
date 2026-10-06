"""The local app (server.py) with the model-bound steps replaced by recorded analyses."""

from jspace_demo import server

import logging
import time

import pytest
from fakes import FakeLoaded, Recorder
from fastapi.testclient import TestClient


def client_for(tmp_path, monkeypatch, *, model="qwen3.5-4b", analyse=None):
    monkeypatch.setenv("JSPACE_HISTORY_DIR", str(tmp_path / "history"))
    app = server.create_app(model, loader=lambda key: FakeLoaded(), analyse=analyse or Recorder())
    return TestClient(app, base_url="http://127.0.0.1:8000")


def wait_ready(client):
    for _ in range(200):
        if client.get("/api/meta").json()["status"] == "ready":
            return
        time.sleep(0.01)
    raise AssertionError("the fake model never became ready")


def finish(client, job):
    for _ in range(300):
        job = client.get(f"/api/jobs/{job['id']}").json()
        if job["status"] in ("done", "error", "cancelled"):
            return job
        time.sleep(0.01)
    raise AssertionError(f"job did not finish: {job}")


def test_without_a_model_the_screens_and_prepared_cases_are_served(tmp_path, monkeypatch):
    with client_for(tmp_path, monkeypatch, model=None) as client:
        assert client.get("/config.json").json() == {"live": True}
        meta = client.get("/api/meta").json()
        assert meta["status"] == "off" and meta["model"] is None and "money" in meta["watch_sets"]
        page = client.get("/")
        assert page.status_code == 200 and "text/html" in page.headers["content-type"]
        assert client.post("/api/jobs", json={"message": "hello"}).status_code == 503


def test_the_model_loads_in_the_background(tmp_path, monkeypatch):
    with client_for(tmp_path, monkeypatch) as client:
        wait_ready(client)
        meta = client.get("/api/meta").json()
        assert meta["model"]["key"] == "qwen3.5-4b" and meta["conditions"] == ["pushback", "persona", "both"]


@pytest.mark.parametrize(
    "body",
    [
        {"message": "  "},
        {"message": "hi", "watch_set": "nope"},
        {"message": "hi", "reply_lang": "fr"},
        {"message": "hi", "condition": "flattery"},
    ],
)
def test_bad_requests_are_refused(tmp_path, monkeypatch, body):
    with client_for(tmp_path, monkeypatch) as client:
        wait_ready(client)
        assert client.post("/api/jobs", json=body).status_code == 422


def test_a_job_lands_in_the_history_and_reads_back_like_a_case(tmp_path, monkeypatch):
    rec = Recorder()
    with client_for(tmp_path, monkeypatch, analyse=rec) as client:
        wait_ready(client)
        job = client.post("/api/jobs", json={"message": "sanding", "watch_set": "home", "condition": "persona"}).json()
        assert job["plan"] == ["generating", "reading", "translating_reply"]
        job = finish(client, job)
        assert job["status"] == "done", job
        assert [(c["message"], c["condition"], c["max_new_tokens"]) for c in rec.calls] == [("sanding", "persona", 80)]
        rows = client.get("/api/analyses").json()
        assert len(rows) == 1 and rows[0]["cell"] == "honest" and rows[0]["best"]["word"] == "toxic"
        assert rows[0]["condition"] == "persona" and rows[0]["translated"] is False
        detail = client.get(f"/api/analyses/{job['result']}").json()
        assert set(detail) == {"analysis", "views", "texts", "layers"}
        assert detail["views"]["watched"][0]["word"] == "toxic" and detail["views"]["honesty"]["cell"] == "honest"
        assert detail["analysis"]["input"]["condition"] == "persona" and detail["texts"]["phrase_ja"] is None
        assert client.delete(f"/api/analyses/{job['result']}").json() == {"deleted": True}
        assert client.get("/api/analyses").json() == []
        assert client.get(f"/api/analyses/{job['result']}").status_code == 404
        assert client.get("/api/jobs/nope").status_code == 404


@pytest.mark.parametrize(
    ("translation", "tagged"),
    [
        (None, False),
        ({"method": "same-model", "message_source": None, "reply_ja": "返事"}, False),  # only the reply was translated
        ({"method": "same-model", "message_source": "相談", "reply_ja": "返事"}, True),
    ],
)
def test_the_translated_tag_means_the_message_was_translated(tmp_path, monkeypatch, translation, tagged):
    extra = {"translation": translation} if translation else {}
    with client_for(tmp_path, monkeypatch, analyse=Recorder(**extra)) as client:
        wait_ready(client)
        finish(client, client.post("/api/jobs", json={"message": "sanding"}).json())
        assert client.get("/api/analyses").json()[0]["translated"] is tagged


def test_an_input_error_reaches_the_page_as_a_code(tmp_path, monkeypatch):
    def analyse(loaded, message, **kwargs):
        raise server.analysis.InputError("too_long", n_tokens=1100, limit=1024)

    with client_for(tmp_path, monkeypatch, analyse=analyse) as client:
        wait_ready(client)
        job = finish(client, client.post("/api/jobs", json={"message": "long"}).json())
        assert job["status"] == "error" and job["error"]["code"] == "too_long"
        assert job["error"]["params"] == {"n_tokens": 1100, "limit": 1024}


@pytest.mark.parametrize("host", ["attacker.example", "attacker.example:8000", "192.168.1.20:8000"])
def test_requests_for_another_host_are_refused(tmp_path, monkeypatch, host):
    """A page on another domain pointed at 127.0.0.1 (DNS rebinding) must not reach the history or the model."""
    with client_for(tmp_path, monkeypatch) as client:
        assert client.get("/api/analyses", headers={"host": host}).status_code == 400
        assert client.get("/api/analyses", headers={"host": "localhost:8000"}).status_code == 200


@pytest.mark.parametrize(
    ("params", "plan"),
    [
        ({"message": "Is bleach safe?", "reply_lang": "en"}, ["generating", "reading"]),
        ({"message": "Is bleach safe?", "reply_lang": "ja"}, ["generating", "reading", "translating_reply"]),
        (
            {"message": "漂白剤は安全？", "reply_lang": "ja"},
            ["translating", "generating", "reading", "translating_reply"],
        ),
        ({"message": "漂白剤は安全？", "reply_lang": "en"}, ["translating", "generating", "reading"]),
    ],
)
def test_a_job_lists_its_steps_up_front(params, plan):
    assert server.job_plan(params) == plan


def test_a_model_that_fails_to_load_is_reported_and_logged(tmp_path, monkeypatch, caplog):
    def loader(key):
        raise OSError("the lens file is missing")

    caplog.set_level(logging.ERROR, logger="jspace_demo.server")
    monkeypatch.setenv("JSPACE_HISTORY_DIR", str(tmp_path / "history"))
    app = server.create_app("qwen3.5-4b", loader=loader, analyse=Recorder())
    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        for _ in range(500):
            meta = client.get("/api/meta").json()
            if meta["status"] == "error":
                break
            time.sleep(0.01)
        assert meta["error"] == "OSError: the lens file is missing"
        assert client.post("/api/jobs", json={"message": "hello"}).status_code == 503
    assert "could not load qwen3.5-4b" in caplog.text
