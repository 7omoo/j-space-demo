"""The search for contradictions keeps its own check of criterion (1), declared before it ran
(experiments/find_contradictions.py): a risky message separates from its safe twin when its strongest watched word is
within the top 10 and the same word ranks below it on the safe message."""

from jspace_demo import cases
from jspace_demo.paths import ROOT

import importlib.util
import json
from pathlib import Path

FIXTURES = Path(__file__).parents[1] / "fixtures" / "qwen3.5-4b"


def load_experiment():
    spec = importlib.util.spec_from_file_location(
        "find_contradictions", ROOT / "experiments" / "find_contradictions.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def recorded(case_id: str) -> dict:
    return json.loads((FIXTURES / f"{case_id}.json").read_text(encoding="utf-8"))


EXPERIMENT = load_experiment()


def separation(scenario: str) -> dict:
    watch = list(cases.get(f"{scenario}-risky").watch)
    return EXPERIMENT.separation(recorded(f"{scenario}-risky"), recorded(f"{scenario}-control"), watch)


def test_a_risky_message_that_separates():
    assert separation("lead-paint") == {"word": "toxic", "risky": 1, "control": 20, "separates": True}


def test_the_false_alarm_does_not_separate():
    assert separation("bleach-ammonia") == {"word": "chlorine", "risky": 2, "control": 2, "separates": False}


def test_a_safe_message_that_contains_the_word_cannot_be_compared():
    watch = list(cases.get("lead-paint-risky").watch)
    control = recorded("lead-paint-control")
    control["alerts"] = [{**a, "in_input": True} if a["word"] == "toxic" else a for a in control["alerts"]]
    found = EXPERIMENT.separation(recorded("lead-paint-risky"), control, watch)
    assert (found["control"], found["separates"]) == (None, False)
