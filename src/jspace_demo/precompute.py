"""Prepare the cases for the screens: analyse every case with a model, then export what the screens read.

    jspace-demo precompute --model qwen3.5-4b   # analyse what is not cached yet, then export
    jspace-demo export                          # export every model already analysed (no model needed)

compute writes  cache/analyses/<model>/<case>.json         the full payload of every case (large; not committed)
export writes   web/data/models.json                       the models with prepared cases
                web/data/<model>/index.json                one summary row per case the site shows (the home table)
                web/data/<model>/cases/<case>.json         one case: views, texts and the expert view's layers

Every case is analysed, the safe messages included, because the report uses them (docs/report.md); only the cases
the site shows are exported (``cases.site_cases``).

The local app and the static site read the same files, so the screens have one source for the prepared cases.
"""

from __future__ import annotations

from jspace_demo import analysis, cases, views
from jspace_demo.glossary import Glossary
from jspace_demo.models import MODELS, spec
from jspace_demo.paths import SITE_DATA, analysis_cache

import json
import os
import shutil
from collections.abc import Callable, Iterable
from pathlib import Path


def payload_path(model_key: str, case_id: str) -> Path:
    return analysis_cache(model_key) / f"{case_id}.json"


def load_payload(model_key: str, case_id: str) -> dict | None:
    path = payload_path(model_key, case_id)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def analyse_case(loaded: analysis.Loaded, case: cases.Case) -> dict:
    """The full payload of one prepared case."""
    if case.raw is not None:
        payload = analysis.analyze(
            loaded, raw=case.raw, watch_words=case.analysis_watch, max_new_tokens=case.max_new_tokens
        )
    else:
        payload = analysis.analyze(
            loaded,
            user=case.user,
            system=case.system,
            watch_words=case.analysis_watch,
            max_new_tokens=case.max_new_tokens,
        )
    payload["case"] = {"id": case.id, "variant": case.variant}
    return payload


def compute(
    loaded: analysis.Loaded,
    *,
    refresh: bool = False,
    only: Iterable[str] | None = None,
    on_case: Callable[[cases.Case, dict, bool], None] | None = None,
) -> None:
    """Analyse every case not cached yet for ``loaded``'s model (all of them with ``refresh``).

    ``on_case(case, payload, computed)`` is told about each case, cached or new.
    """
    wanted = set(only) if only else None
    unknown = sorted(wanted - set(cases.all_cases())) if wanted else []
    if unknown:
        raise ValueError(f"unknown case {', '.join(unknown)}")
    for case in cases.all_cases().values():
        if wanted is not None and case.id not in wanted:
            continue
        payload = None if refresh else load_payload(loaded.key, case.id)
        computed = payload is None
        if computed:
            payload = analyse_case(loaded, case)
            _write_json(payload_path(loaded.key, case.id), payload)
        if on_case:
            on_case(case, payload, computed)


def export(model_keys: Iterable[str] | None = None, *, glossary: Glossary | None = None) -> dict[str, int]:
    """Write the screens' files for each model (default: every model with a cache). Returns cases written per model.

    Raises before writing a model whose cache misses a case; models exported before it stay written."""
    glossary = glossary or Glossary.load()
    keys = list(model_keys) if model_keys else [k for k in MODELS if analysis_cache(k).exists()]
    written = {}
    for key in keys:
        missing = [cid for cid in cases.site_cases() if load_payload(key, cid) is None]
        if missing:
            raise FileNotFoundError(
                f"{key}: {len(missing)} cases not analysed yet (first: {missing[0]}); "
                f"run jspace-demo precompute --model {key}"
            )
        written[key] = _export_model(key, glossary)
    _write_models_index()
    return written


def case_file(case: cases.Case, payload: dict, model_key: str, glossary: Glossary) -> dict:
    """Everything the case screen needs for one case."""
    view = views.build(payload, concern_words=case.watch, glossary=glossary, focus_token=case.focus_token)
    texts = cases.texts(case, model_key)
    phrase = (view["trigger"] or {}).get("phrase")
    texts["phrase_ja"] = cases.phrase_ja(phrase["text"]) if phrase else None
    return {"case": summary_row(case, view), "views": view, "texts": texts, "layers": views.layers(payload)}


def summary_row(case: cases.Case, view: dict) -> dict:
    """One row of the home table: where the case sits and how its reply came out, without loading the case."""
    return {
        "id": case.id,
        "scenario": case.scenario,
        "kind": case.kind,
        "variant": case.variant,
        **cases.titles(case),
        "honesty": {"cell": view["honesty"]["cell"]} if view["honesty"] else None,
    }


def _export_model(model_key: str, glossary: Glossary) -> int:
    out = SITE_DATA / model_key
    if out.exists():
        shutil.rmtree(out)  # a case removed from the catalogue must not linger
    rows = []
    for case in cases.site_cases().values():
        built = case_file(case, load_payload(model_key, case.id), model_key, glossary)
        _write_json(out / "cases" / f"{case.id}.json", built)
        rows.append(built["case"])
    sample = load_payload(model_key, next(iter(cases.site_cases())))
    _write_json(out / "index.json", {"model": _model_info(model_key, sample), "cases": rows})
    return len(rows)


def _model_info(model_key: str, payload: dict) -> dict:
    s = spec(model_key)
    info = payload.get("model") or {}
    return {
        "key": s.key,
        "label": s.label,
        "hf_name": s.hf_name,
        "licence": s.licence,
        "n_layers": info.get("n_layers") or payload["n_layers"],
        "band": payload["band"],
        "vocab_size": payload["vocab_size"],
    }


def _write_models_index() -> None:
    """``models.json``: the models whose cases are exported, in the switcher's order."""
    rows = []
    for key in MODELS:
        index = SITE_DATA / key / "index.json"
        if index.exists():
            rows.append(json.loads(index.read_text(encoding="utf-8"))["model"])
    _write_json(SITE_DATA / "models.json", {"default": rows[0]["key"] if rows else None, "models": rows})


def _write_json(path: Path, content: dict) -> None:
    """Write whole or not at all: a reader never sees half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(content, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    os.replace(partial, path)
