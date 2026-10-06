"""The prepared cases: what each scenario sends to the model, the words watched for it, and its Japanese texts.

A scenario is defined once in ``data/cases.json`` and yields its cases:

    explainer       one case, a plain-text prompt (the boot riddle)
    risk            "<id>-risky" and "<id>-control": the message, and a safe message of the same shape
    false_positive  the same pair, kept to show a topic that alarms on the safe message too
    pressure        for a scenario with ``"pressure": true``, "<id>-pushback", "<id>-persona" and "<id>-both":
                    the risky message under the conditions of ``pressure.py``

The site shows the pressure experiment (``site_cases``): each pressure scenario's risky message and its three
pressure cases, plus the explainer. The safe messages and the false-alarm scenario are analysed for the report
(docs/report.md, section 7) and not shown.

English is what the model reads. The Japanese texts are for the screens only; the Japanese replies depend on
the model, so they live per model in ``data/replies/<model>.json``.
"""

from __future__ import annotations

from jspace_demo import pressure
from jspace_demo.paths import PACKAGE_DATA

import json
from dataclasses import dataclass
from functools import cache, lru_cache


@dataclass(frozen=True)
class Case:
    id: str
    scenario: str
    kind: str  # explainer | risk | false_positive | pressure
    variant: str  # raw | risky | control | pushback | persona | both
    watch: tuple[str, ...]  # the scenario's watch words: what the screens judge by
    user: str | None = None  # the chat message as sent, pressure suffix included
    system: str | None = None
    raw: str | None = None  # a plain-text prompt, no chat template
    max_new_tokens: int = 80
    focus_token: str | None = None
    control: str | None = None  # a risky case: the id of its safe counterpart
    base: str | None = None  # a pressure case: the risky case it puts under pressure

    @property
    def analysis_watch(self) -> list[str]:
        """The words read out: the scenario's own first, then (except for the explainer) the generic list."""
        if self.kind == "explainer":
            return list(self.watch)
        return list(dict.fromkeys([*self.watch, *generic_watch()]))


@lru_cache(maxsize=1)
def catalogue() -> dict:
    return json.loads((PACKAGE_DATA / "cases.json").read_text(encoding="utf-8"))


def generic_watch() -> tuple[str, ...]:
    """A single list a business could watch across every conversation (read out, never judged by)."""
    return tuple(catalogue()["generic_watch"])


def watch_sets() -> dict[str, tuple[str, ...]]:
    """The word sets offered on the "try your own text" screen, by kind of business."""
    return {name: tuple(words) for name, words in catalogue()["watch_sets"].items()}


@lru_cache(maxsize=1)
def all_cases() -> dict[str, Case]:
    """Every case: what is analysed (the report uses them all). The site shows ``site_cases``."""
    out: dict[str, Case] = {}
    for s in catalogue()["scenarios"]:
        watch = tuple(s["watch"])
        if s["kind"] == "explainer":
            out[s["id"]] = Case(
                s["id"],
                s["id"],
                "explainer",
                "raw",
                watch,
                raw=s["raw"],
                max_new_tokens=s.get("max_new_tokens", 80),
                focus_token=s.get("focus_token"),
            )
            continue
        risky, control = f"{s['id']}-risky", f"{s['id']}-control"
        out[risky] = Case(risky, s["id"], s["kind"], "risky", watch, user=s["message"], control=control)
        out[control] = Case(control, s["id"], s["kind"], "control", watch, user=s["control"])
    for s in catalogue()["scenarios"]:
        if not s.get("pressure"):
            continue
        for condition in pressure.CONDITIONS:
            user, system = pressure.apply(s["message"], condition)
            case_id = f"{s['id']}-{condition}"
            out[case_id] = Case(
                case_id,
                s["id"],
                "pressure",
                condition,
                tuple(s["watch"]),
                user=user,
                system=system,
                base=f"{s['id']}-risky",
            )
    return out


@lru_cache(maxsize=1)
def site_cases() -> dict[str, Case]:
    """The cases the screens show, in list order: the explainer, then for each pressure scenario its risky message
    (asked without pressure) and its three pressure cases."""
    pressured = {s["id"] for s in catalogue()["scenarios"] if s.get("pressure")}
    shown = {cid: c for cid, c in all_cases().items() if c.kind == "explainer"}
    for scenario in (s["id"] for s in catalogue()["scenarios"] if s["id"] in pressured):
        for variant in ("risky", *pressure.CONDITIONS):
            shown[f"{scenario}-{variant}"] = all_cases()[f"{scenario}-{variant}"]
    return shown


def get(case_id: str) -> Case:
    try:
        return all_cases()[case_id]
    except KeyError:
        raise KeyError(f"no case {case_id!r}") from None


@cache
def replies_ja(model_key: str) -> dict[str, str]:
    """The Japanese renderings of one model's replies, by case id (empty until drafted)."""
    path = PACKAGE_DATA / "replies" / f"{model_key}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _scenario(case: Case) -> dict:
    return next(s for s in catalogue()["scenarios"] if s["id"] == case.scenario)


def titles(case: Case) -> dict:
    """A shown case's names, Japanese and English: its own title and its scenario's."""
    scenario = _scenario(case)
    variant = "risky" if case.kind == "pressure" else case.variant
    return {"title": scenario["variants"][variant]["title"], "scenario_title": scenario["title"]}


def texts(case: Case, model_key: str) -> dict:
    """What the case screen shows around a shown case, beyond the analysis itself."""
    data = catalogue()
    scenario = _scenario(case)
    variant = "risky" if case.kind == "pressure" else case.variant
    out = {
        "message": scenario["raw"] if variant == "raw" else scenario["message"],  # before any pressure is added
        "message_ja": scenario["variants"][variant].get("message_ja"),
        "reply_ja": replies_ja(model_key).get(case.id),
    }
    if case.kind == "pressure":
        pushed = case.variant in ("pushback", "both")
        out["pushback"] = pressure.PUSHBACK.strip() if pushed else None
        out["pushback_ja"] = data["pushback_ja"] if pushed else None
    return out


def phrase_ja(text: str) -> str | None:
    """The Japanese rendering of one phrase of a prepared message (where a concern first came up)."""
    return catalogue()["phrases_ja"].get(text)
