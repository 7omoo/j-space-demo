"""Stand-ins for the model-bound steps of the local app: a loaded model and an analysis that returns recorded
payloads. Shared by the server's unit tests and the browser tests of the local app's screens."""

import json
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures" / "qwen3.5-4b"


def recorded(case_id: str, **extra) -> dict:
    return {**json.loads((FIXTURES / f"{case_id}.json").read_text(encoding="utf-8")), **extra}


class FakeLoaded:
    key, device = "qwen3.5-4b", "cpu"
    info = {"key": "qwen3.5-4b", "label": "Qwen3.5-4B", "n_layers": 32}


class Recorder:
    """Stands in for analysis.analyze_message: returns recorded payloads and remembers how it was called."""

    def __init__(self, case_id="lead-paint-risky", **extra):
        self.calls, self.case_id, self.extra = [], case_id, extra

    def __call__(
        self, loaded, message, *, watch_words, condition=None, reply_in_japanese=False, max_new_tokens=80, on_step=None
    ):
        self.calls.append(
            {
                "message": message,
                "condition": condition,
                "max_new_tokens": max_new_tokens,
                "reply_in_japanese": reply_in_japanese,
            }
        )
        if on_step:
            on_step("generating")
        return recorded(self.case_id, **self.extra)
