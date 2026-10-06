"""Preparing the cases (precompute.py): analysing into the cache, and exporting the files the screens read."""

from jspace_demo import cases, precompute

import json
import shutil
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parents[1] / "fixtures" / "qwen3.5-4b"


@pytest.fixture
def site(tmp_path, monkeypatch):
    """A cache filled from the recorded analyses, and an empty web/data, both under tmp_path."""
    cache = tmp_path / "cache"
    shutil.copytree(FIXTURES, cache / "qwen3.5-4b")
    monkeypatch.setattr(precompute, "analysis_cache", lambda key: cache / key)
    monkeypatch.setattr(precompute, "SITE_DATA", tmp_path / "data")
    return tmp_path / "data"


def test_export_writes_every_case_an_index_and_the_model_list(site):
    assert precompute.export(["qwen3.5-4b"]) == {"qwen3.5-4b": len(cases.site_cases())}
    models = json.loads((site / "models.json").read_text(encoding="utf-8"))
    assert models["default"] == "qwen3.5-4b" and [m["key"] for m in models["models"]] == ["qwen3.5-4b"]
    assert models["models"][0]["n_layers"] == 32 and models["models"][0]["band"] == [8, 25]
    index = json.loads((site / "qwen3.5-4b" / "index.json").read_text(encoding="utf-8"))
    assert [row["id"] for row in index["cases"]] == list(cases.site_cases())
    assert not list(site.rglob("*.partial"))


def test_the_safe_messages_are_analysed_for_the_report_but_not_exported(site):
    precompute.export(["qwen3.5-4b"])
    shipped = {path.stem for path in (site / "qwen3.5-4b" / "cases").glob("*.json")}
    assert "lead-paint-control" in cases.all_cases() and "lead-paint-control" not in shipped
    assert "bleach-ammonia-risky" not in shipped  # the false alarm has no pressure cases


def test_a_case_file_has_what_the_case_screen_draws(site):
    precompute.export(["qwen3.5-4b"])
    built = json.loads((site / "qwen3.5-4b" / "cases" / "lead-paint-risky.json").read_text(encoding="utf-8"))
    assert set(built) == {"case", "views", "texts", "layers"}
    assert built["case"]["honesty"] == {"cell": "honest"} and built["case"]["title"]["ja"]
    assert [w["word"] for w in built["views"]["watched"]] == ["toxic", "hazard", "unsafe", "poisoning"]
    texts = built["texts"]
    assert texts["reply_ja"].startswith("今すぐ作業を止めてください")
    assert texts["phrase_ja"] == "マスクは持っていない"  # where the concern came up, in Japanese
    assert built["layers"] is None  # the fixtures keep no per-layer tables (the explainer's do)
    explainer = json.loads((site / "qwen3.5-4b" / "cases" / "boot-riddle.json").read_text(encoding="utf-8"))
    assert explainer["layers"]["n_layers"] == 32 and set(explainer["layers"]["top"]) == {"j", "logit"}


def test_a_pressure_row_names_its_scenario_and_condition(site):
    precompute.export(["qwen3.5-4b"])
    index = json.loads((site / "qwen3.5-4b" / "index.json").read_text(encoding="utf-8"))
    row = next(r for r in index["cases"] if r["id"] == "warfarin-aspirin-persona")
    assert (row["scenario"], row["kind"], row["variant"]) == ("warfarin-aspirin", "pressure", "persona")
    assert row["honesty"]["cell"] == "sycophancy" and row["scenario_title"]["ja"]


def test_export_refuses_a_model_with_cases_missing(site, tmp_path):
    (tmp_path / "cache" / "qwen3.5-4b" / "lead-paint-risky.json").unlink()
    with pytest.raises(FileNotFoundError, match="1 cases not analysed"):
        precompute.export(["qwen3.5-4b"])


def test_a_removed_case_does_not_linger(site):
    precompute.export(["qwen3.5-4b"])
    stale = site / "qwen3.5-4b" / "cases" / "old-case.json"
    stale.write_text("{}")
    precompute.export(["qwen3.5-4b"])
    assert not stale.exists()


def test_compute_analyses_only_what_is_not_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(precompute, "analysis_cache", lambda key: tmp_path / key)
    calls = []
    monkeypatch.setattr(
        precompute,
        "analyse_case",
        lambda loaded, case: calls.append(case.id) or {"alerts": [], "case": {"id": case.id}},
    )
    loaded = type("L", (), {"key": "toy"})()
    seen = []
    precompute.compute(
        loaded, only=["lead-paint-risky", "boot-riddle"], on_case=lambda c, p, new: seen.append((c.id, new))
    )
    assert calls == ["boot-riddle", "lead-paint-risky"] and seen == [("boot-riddle", True), ("lead-paint-risky", True)]
    precompute.compute(loaded, only=["lead-paint-risky"], on_case=lambda c, p, new: seen.append((c.id, new)))
    assert calls == ["boot-riddle", "lead-paint-risky"] and seen[-1] == ("lead-paint-risky", False)  # from the cache
    precompute.compute(loaded, only=["lead-paint-risky"], refresh=True)
    assert calls[-1] == "lead-paint-risky" and len(calls) == 3


def test_a_mistyped_case_is_refused_before_any_work():
    from jspace_demo import cli

    assert cli.main(["precompute", "--model", "qwen3.5-4b", "--case", "lead-paint-riskyy"]) == 2
    with pytest.raises(ValueError, match="lead-paint-riskyy"):
        precompute.compute(None, only=["lead-paint-riskyy"])
