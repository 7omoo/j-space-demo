"""The local app: the screens, the prepared cases, and analyses of the user's own text.

    jspace-demo serve                    # http://127.0.0.1:8000, Qwen3.5-4B loaded in the background
    jspace-demo serve --model qwen3-8b   # another model (models.py)
    jspace-demo serve --no-model         # the screens and prepared cases only

The prepared cases are files under web/data/ that the page reads itself, exactly as on the static site. The
server adds only what needs a model or this machine: ``/config.json`` (tells the page it runs locally),
``/api/meta`` (the model's state), ``/api/jobs`` (the user's own analyses, one at a time) and
``/api/analyses`` (their history, kept on this machine).

It listens on 127.0.0.1 only and answers only requests addressed to this machine by name, so a web page cannot
reach the history or the model through a domain it points at 127.0.0.1 (DNS rebinding).
"""

from __future__ import annotations

from jspace_demo import analysis, cases, history, jobs, pressure, runtime, translate, views
from jspace_demo.glossary import Glossary
from jspace_demo.models import DEFAULT_MODEL, spec
from jspace_demo.paths import WEB_DIR

import logging
import threading
import traceback
from collections.abc import Callable
from contextlib import asynccontextmanager, contextmanager

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

HOST = "127.0.0.1"
LOCAL_NAMES = ["127.0.0.1", "localhost"]  # the Host header a request to this machine carries

Loader = Callable[[str], analysis.Loaded]
log = logging.getLogger(__name__)


class State:
    """The one model of this process, loaded in the background, and the lock that keeps it to one job."""

    def __init__(self, model_key: str | None):
        self.model_key = model_key
        self.loaded: analysis.Loaded | None = None
        self.status = "off" if model_key is None else "idle"  # off | idle | loading | ready | error
        self.error: str | None = None
        self.gpu = threading.Lock()

    def load(self, loader: Loader) -> None:
        self.status = "loading"
        try:
            self.loaded = loader(self.model_key)
            self.status = "ready"
        except Exception:
            log.exception("could not load %s", self.model_key)
            self.error = traceback.format_exc().strip().splitlines()[-1]
            self.status = "error"  # last: whoever sees the error state also sees its message

    @contextmanager
    def on_gpu(self):
        """One job on the model at a time; afterwards the allocator's cache goes back to the system."""
        with self.gpu:
            try:
                yield
            finally:
                if self.loaded is not None:
                    runtime.free_cached_memory(self.loaded.device)


class JobRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000, pattern=r"\S")
    watch_set: str = "general"
    reply_lang: str = Field("ja", pattern="^(ja|en)$")
    condition: str | None = None  # social pressure: one of pressure.CONDITIONS


def job_plan(params: dict) -> list[str]:
    """The steps a job will report, in order, so the page can list them all from the start."""
    steps = ["translating"] if translate.needs_english(params["message"]) else []
    steps += ["generating", "reading"]
    if params["reply_lang"] == "ja":
        steps.append("translating_reply")
    return steps


def create_app(
    model_key: str | None = DEFAULT_MODEL,
    *,
    loader: Loader = analysis.load,
    analyse: Callable[..., dict] = analysis.analyze_message,
) -> FastAPI:
    """The app. ``model_key=None`` serves the screens without a model; ``loader`` and ``analyse`` are the
    model-bound steps (tests replace them)."""
    state = State(model_key)
    glossary = Glossary.load()
    sets = cases.watch_sets()

    def run_job(params: dict, step: Callable[[str], None]) -> str:
        if state.loaded is None:
            raise RuntimeError(f"the model is not ready ({state.status})")
        concern = sets[params["watch_set"]]
        words = list(dict.fromkeys([*concern, *cases.generic_watch()]))
        payload = analyse(
            state.loaded,
            params["message"],
            watch_words=words,
            condition=params.get("condition"),
            reply_in_japanese=params["reply_lang"] == "ja",
            on_step=step,
        )
        view = views.build(payload, concern_words=concern, glossary=glossary)
        return history.save({"input": params, "payload": payload}, _history_summary(params, payload, view))

    queue = jobs.Jobs(run_job, gpu=state.on_gpu)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if model_key is not None:
            threading.Thread(target=state.load, args=(loader,), daemon=True).start()
        yield

    app = FastAPI(title="J-Space Demo", lifespan=lifespan)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=LOCAL_NAMES)

    @app.get("/config.json")
    def config() -> dict:
        return {"live": True}

    @app.get("/api/meta")
    def meta() -> dict:
        model = (
            state.loaded.info
            if state.loaded
            else ({"key": model_key, "label": spec(model_key).label} if model_key else None)
        )
        return {
            "status": state.status,
            "error": state.error,
            "model": model,
            "watch_sets": {name: list(words) for name, words in sets.items()},
            "conditions": list(pressure.CONDITIONS),
            "memory_gib": runtime.held_memory_gib(state.loaded.device) if state.loaded else None,
        }

    @app.post("/api/jobs")
    def submit_job(req: JobRequest) -> dict:
        if req.watch_set not in sets:
            raise HTTPException(status_code=422, detail=f"unknown watch set {req.watch_set!r}")
        if req.condition is not None and req.condition not in pressure.CONDITIONS:
            raise HTTPException(status_code=422, detail=f"unknown condition {req.condition!r}")
        if state.loaded is None:
            raise HTTPException(status_code=503, detail={"status": state.status, "error": state.error})
        params = req.model_dump()
        return queue.submit(params, plan=job_plan(params))

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str) -> dict:
        found = queue.get(job_id)
        if found is None:
            raise HTTPException(status_code=404, detail="no such job")
        return found

    @app.delete("/api/jobs/{job_id}")
    def cancel_job(job_id: str) -> dict:
        return {"cancelled": queue.cancel(job_id)}

    @app.get("/api/analyses")
    def analyses() -> list[dict]:
        return history.listing()

    @app.get("/api/analyses/{analysis_id}")
    def analysis_detail(analysis_id: str) -> dict:
        record = history.load(analysis_id)
        if record is None:
            raise HTTPException(status_code=404, detail="no such analysis")
        return history_file(record, sets, glossary)

    @app.delete("/api/analyses/{analysis_id}")
    def delete_analysis(analysis_id: str) -> dict:
        return {"deleted": history.delete(analysis_id)}

    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")  # last: the routes above come first
    return app


def history_file(record: dict, sets: dict, glossary: Glossary) -> dict:
    """A saved analysis in the shape of a prepared case file, so one case screen draws both."""
    params, payload = record["input"], record["payload"]
    # a watch set renamed since the analysis: judge by the words it was analysed with, less the generic list
    concern = sets.get(params["watch_set"]) or [w for w in payload["watch"]["words"] if w not in cases.generic_watch()]
    view = views.build(payload, concern_words=concern, glossary=glossary)
    translation = payload.get("translation") or {}
    texts = {
        "message_ja": translation.get("message_source"),
        "reply_ja": translation.get("reply_ja"),
        "phrase_ja": None,
    }
    return {
        "analysis": {"id": record["id"], "created": record["created"], "input": params},
        "views": view,
        "texts": texts,
        "layers": views.layers(payload),
    }


def _history_summary(params: dict, payload: dict, view: dict) -> dict:
    best = view["alerts"][0] if view["alerts"] else None
    return {
        "title": params["message"],
        "watch_set": params["watch_set"],
        "condition": params.get("condition"),
        "model": (payload.get("model") or {}).get("key"),
        "cell": view["honesty"]["cell"] if view["honesty"] else None,
        "best": best and {"word": best["word"], "rank": best["rank"], "ja": best.get("ja")},
        "translated": bool((payload.get("translation") or {}).get("message_source")),  # the message, not the reply
    }
