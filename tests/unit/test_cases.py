"""The prepared cases (cases.py, data/cases.json): integrity, the experiment's wording, and the Japanese texts of
the cases the site shows."""

from jspace_demo import cases, pressure, views, vocab

import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parents[1] / "fixtures" / "qwen3.5-4b"
ALL = cases.all_cases()
SHOWN = cases.site_cases()
SCENARIOS = cases.catalogue()["scenarios"]


def test_every_scenario_yields_its_cases():
    assert len(ALL) == 38 and len(set(ALL)) == len(ALL)
    for s in SCENARIOS:
        own = [c for c in ALL.values() if c.scenario == s["id"]]
        expected = {"explainer": 1, "risk": 2, "false_positive": 2}[s["kind"]] + (3 if s.get("pressure") else 0)
        assert len(own) == expected, s["id"]


def test_case_shapes():
    for c in ALL.values():
        if c.kind == "explainer":
            assert c.raw and c.user is None and c.system is None
        else:
            assert c.user and c.raw is None, c.id
        if c.variant == "risky":
            assert ALL[c.control].variant == "control" and ALL[c.control].scenario == c.scenario
        if c.kind == "pressure":
            base = ALL[c.base]
            assert base.variant == "risky" and c.watch == base.watch
            assert c.user.startswith(base.user) and (c.system is not None) == (c.variant in ("persona", "both"))
            assert (c.user != base.user) == (c.variant in ("pushback", "both"))


def test_pressure_uses_the_experiments_strings():
    for c in ALL.values():
        if c.kind == "pressure":
            assert (c.user, c.system) == pressure.apply(ALL[c.base].user, c.variant)


def test_the_bleach_scenario_is_not_put_under_pressure():
    """Its pressured reply gave a dangerous mixing ratio; the public cases are safe to show in full."""
    assert not any(c.scenario == "bleach-ammonia" and c.kind == "pressure" for c in ALL.values())


def test_watch_words_never_appear_in_what_the_model_reads():
    # A hit must mean the model brought the concept in itself, not that it copied the input.
    for c in ALL.values():
        text = " ".join(filter(None, (c.user, c.raw, c.system)))
        if c.kind == "explainer":
            continue
        assert not [w for w in c.watch if vocab.mentions(w, text)], c.id


def test_analysis_watch_puts_the_scenario_words_first():
    assert cases.get("boot-riddle").analysis_watch == ["Italy", "Euro"]
    words = cases.get("lottery-scam-risky").analysis_watch
    own = cases.get("lottery-scam-risky").watch
    assert words[: len(own)] == list(own)
    assert set(cases.generic_watch()) <= set(words) and len(words) == len(set(words))


def test_watch_sets_are_single_words_without_duplicates():
    for name, words in cases.watch_sets().items():
        assert words and len(words) == len(set(words)) and all(" " not in w for w in words), name


@pytest.mark.parametrize("case_id", list(ALL))
def test_the_recorded_analysis_is_of_this_case(case_id):
    """The fixtures (and the exported data) were measured with exactly the catalogue's text."""
    payload = json.loads((FIXTURES / f"{case_id}.json").read_text(encoding="utf-8"))
    c = ALL[case_id]
    assert payload["user_text"] == (c.raw if c.raw is not None else c.user)
    assert payload.get("system") == c.system
    assert set(payload["watch"]["words"]) | set(payload["watch"]["skipped"]) == set(c.analysis_watch)


def test_watch_words_qwen_cannot_read_are_the_known_ones():
    """The case screen says these are not watched (views' ``unread_words``); a tokenizer change would alter that."""
    skipped = {}
    for case_id in ALL:
        payload = json.loads((FIXTURES / f"{case_id}.json").read_text(encoding="utf-8"))
        for word in payload["watch"]["skipped"]:
            skipped.setdefault(ALL[case_id].scenario, set()).add(word)
    assert skipped == {"storm-hike": {"hypothermia"}, "ceo-wire": {"impersonation"}, "bleach-ammonia": {"fumes"}}


def test_every_shown_case_has_its_japanese_texts():
    for c in SHOWN.values():
        names, t = cases.titles(c), cases.texts(c, "qwen3.5-4b")
        assert all(names[k]["ja"] and names[k]["en"] for k in ("title", "scenario_title")), c.id
        assert t["message"] and t["message_ja"], c.id
        if c.kind == "pressure":
            assert bool(t["pushback_ja"]) == (c.variant in ("pushback", "both"))


def test_the_site_shows_each_pressure_scenario_under_four_conditions_and_the_explainer():
    shown = cases.site_cases()
    pressured = sorted({c.scenario for c in ALL.values() if c.kind == "pressure"})
    assert list(shown)[0] == "boot-riddle" and len(shown) == 1 + 4 * len(pressured) == 29
    for scenario in pressured:
        assert [cid for cid in shown if cid.startswith(scenario + "-")] == [
            f"{scenario}-{v}" for v in ("risky", "pushback", "persona", "both")
        ]
    assert not [cid for cid, c in shown.items() if c.variant == "control"]


@pytest.mark.parametrize("model_key", ["qwen3.5-4b", "qwen3-8b", "gemma-3-12b-it", "qwen3-14b"])
def test_every_shown_reply_has_a_japanese_rendering_and_nothing_else_does(model_key):
    replies = cases.replies_ja(model_key)
    assert set(replies) == set(SHOWN), sorted(set(replies) ^ set(SHOWN))
    assert all(replies.values())


def test_every_phrase_of_a_shown_message_has_a_japanese_rendering():
    missing = set()
    for case_id, c in SHOWN.items():
        if c.kind == "explainer":
            continue
        payload = json.loads((FIXTURES / f"{case_id}.json").read_text(encoding="utf-8"))
        missing |= {p["text"] for p in views.message.phrases(payload) if cases.phrase_ja(p["text"]) is None}
    assert not missing, sorted(missing)
