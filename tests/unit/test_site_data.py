"""The published data (web/data): what the static site ships for every model, checked as it is on disk."""

from jspace_demo import cases, precompute
from jspace_demo.glossary import Glossary
from jspace_demo.paths import ROOT, SITE_DATA
from jspace_demo.views import is_unshown

import json
from pathlib import Path

import pytest

MODELS = json.loads((SITE_DATA / "models.json").read_text(encoding="utf-8"))
KEYS = [m["key"] for m in MODELS["models"]]
FIXTURES = Path(__file__).parents[1] / "fixtures" / MODELS["default"]
GLOSSARY = Glossary.load()


def published(model_key: str, case_id: str) -> dict:
    return json.loads((SITE_DATA / model_key / "cases" / f"{case_id}.json").read_text(encoding="utf-8"))


def recorded(case_id: str) -> dict:
    return json.loads((FIXTURES / f"{case_id}.json").read_text(encoding="utf-8"))


def case_files(model_key: str) -> dict[str, dict]:
    folder = SITE_DATA / model_key / "cases"
    return {path.stem: json.loads(path.read_text(encoding="utf-8")) for path in sorted(folder.glob("*.json"))}


def shown_words(node) -> set[str]:
    """Every word a view shows: the text of each word-like entry, wherever it sits."""
    found: set[str] = set()
    if isinstance(node, dict):
        for field in ("text", "en", "word"):
            if isinstance(node.get(field), str):
                found.add(node[field])
        for value in node.values():
            found |= shown_words(value)
    elif isinstance(node, list):
        for value in node:
            found |= shown_words(value)
    return found


def test_the_default_model_is_listed_first():
    assert MODELS["default"] == KEYS[0]


@pytest.mark.parametrize("model_key", KEYS)
def test_every_model_has_every_case_and_an_index_that_matches(model_key):
    files = case_files(model_key)
    index = json.loads((SITE_DATA / model_key / "index.json").read_text(encoding="utf-8"))
    assert list(files) == sorted(cases.site_cases())
    assert [row["id"] for row in index["cases"]] == list(cases.site_cases())
    for row in index["cases"]:
        assert row == {key: files[row["id"]]["case"].get(key) for key in row}, row["id"]


REPORTED = {  # docs/report.md and the READMEs: sycophancy (of 21) under pressure, and (of 7) without it
    "qwen3.5-4b": (14, 0),
    "qwen3-8b": (8, 0),
    "gemma-3-12b-it": (15, 0),
    "qwen3-14b": (8, 1),
}


@pytest.mark.parametrize("model_key", KEYS)
def test_the_home_tables_counts_are_the_reported_ones(model_key):
    rows = json.loads((SITE_DATA / model_key / "index.json").read_text(encoding="utf-8"))["cases"]
    pressured = [r for r in rows if r["kind"] == "pressure"]
    plain = [r for r in rows if r["variant"] == "risky"]

    def agreeing(found):
        return sum(r["honesty"]["cell"] == "sycophancy" for r in found)

    assert (len(pressured), len(plain)) == (21, 7)
    assert (agreeing(pressured), agreeing(plain)) == REPORTED[model_key]


READINGS = json.loads((ROOT / "experiments" / "stance_readings.json").read_text(encoding="utf-8"))
UNCLEAR = {  # the same documents and the about page: of those, the replies a full reading took as unclear
    "qwen3.5-4b": (1, 0),
    "qwen3-8b": (1, 0),
    "gemma-3-12b-it": (2, 0),
    "qwen3-14b": (2, 1),
}


@pytest.mark.parametrize("model_key", KEYS)
def test_the_sycophantic_results_a_full_reading_took_as_unclear_are_the_reported_ones(model_key):
    """The rule reads a reply that goes along with a soft hedge as agreement. The documents say how many of the
    sycophantic results are such replies; none is a reply a full reading took as a warning."""
    rows = json.loads((SITE_DATA / model_key / "index.json").read_text(encoding="utf-8"))["cases"]
    sycophantic = [r for r in rows if r["honesty"] and r["honesty"]["cell"] == "sycophancy"]
    reading = {r["id"]: READINGS[model_key][r["id"]] for r in sycophantic}
    assert "warn" not in reading.values()

    pressured = [r for r in sycophantic if r["kind"] == "pressure"]
    plain = [r for r in sycophantic if r["variant"] == "risky"]

    def unclear(found):
        return sum(reading[r["id"]] == "unclear" for r in found)

    assert (unclear(pressured), unclear(plain)) == UNCLEAR[model_key]


def named_words(node) -> list[dict]:
    """Every word entry a view names in Japanese: the ones that carry a "ja" field."""
    found: list[dict] = []
    if isinstance(node, dict):
        if "ja" in node:
            found.append(node)
        for value in node.values():
            found += named_words(value)
    elif isinstance(node, list):
        for value in node:
            found += named_words(value)
    return found


@pytest.mark.parametrize("model_key", KEYS)
def test_every_word_a_screen_names_is_in_the_glossary(model_key):
    """A word without a Japanese name is in the glossary on purpose (shown as read), never just missing."""
    unlisted = set()
    for built in case_files(model_key).values():
        for item in named_words(built["views"]):
            text = item.get("en") or item.get("word") or item.get("text")
            if item["ja"] is None and GLOSSARY.key(text) not in GLOSSARY.entries:
                unlisted.add(text)
    assert not unlisted, sorted(unlisted)


@pytest.mark.parametrize("model_key", KEYS)
def test_no_screen_shows_an_unshown_token(model_key):
    for case_id, built in case_files(model_key).items():
        words = shown_words(built["views"])
        if built.get("layers"):
            layers = built["layers"]
            words |= {
                layers["words"][i] for grid in layers["top"].values() for row in grid for cell in row for i in cell
            }
        assert not {w for w in words if is_unshown(w)}, case_id


@pytest.mark.parametrize("case_id", list(cases.site_cases()))
def test_the_default_models_files_are_what_the_code_builds_now(case_id):
    """Rebuilt from the recorded analyses, the default model's files match what is published, so a change to the
    views or the cases without `jspace-demo export` fails here. The recordings keep the per-layer tables of the
    explainer only, so the expert view is compared there."""
    built = precompute.case_file(cases.get(case_id), recorded(case_id), MODELS["default"], Glossary.load())
    built = json.loads(json.dumps(built, ensure_ascii=False))  # tuples become lists, as on disk
    shipped = published(MODELS["default"], case_id)
    for part in ("case", "views", "texts"):
        assert built[part] == shipped[part], part
    if built["layers"] is not None:
        assert built["layers"] == shipped["layers"]


@pytest.mark.parametrize("model_key", KEYS)
def test_every_phrase_a_case_names_has_its_japanese(model_key):
    """Where the concern came up is named in Japanese too, under every model's tokenizer."""
    missing = [
        (case_id, built["views"]["trigger"]["phrase"]["text"])
        for case_id, built in case_files(model_key).items()
        if built["views"]["mode"] == "chat"  # the explainer is told on the About page, not on a case page
        and (built["views"]["trigger"] or {}).get("where") == "in_message"
        and not built["texts"]["phrase_ja"]
    ]
    assert not missing, missing
