"""Trim one model's analysed cases into small test fixtures (tests/fixtures/<model>/<case>.json). No model loaded.

    uv run python scripts/make_fixtures.py [model_key]

The analysis cache (cache/analyses/) is not committed. A fixture keeps what the views are built from (tokens,
alerts, concepts, the reply) and drops the per-layer tables, except for the explainer: its chain and the expert
view read them.
"""

from jspace_demo import cases
from jspace_demo.models import DEFAULT_MODEL
from jspace_demo.paths import ROOT
from jspace_demo.precompute import load_payload

import json
import sys

KEEP = (
    "tokens",
    "n_prompt",
    "user_text",
    "reply",
    "band",
    "n_layers",
    "vocab_size",
    "watch",
    "alerts",
    "concepts",
    "mode",
    "model",
    "timing",
    "system",
    "case",
)


def trim(payload: dict) -> dict:
    trimmed = {key: payload[key] for key in KEEP if key in payload}
    if payload.get("mode") == "raw":
        trimmed["layers"] = payload["layers"]
        used = {t for tops in payload["layers"].values() for pos in tops["ids"] for row in pos for t in row}
        trimmed["vocab"] = {str(t): payload["vocab"][str(t)] for t in sorted(used)}
    return trimmed


def main() -> None:
    model_key = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    target = ROOT / "tests" / "fixtures" / model_key
    payloads = {case_id: load_payload(model_key, case_id) for case_id in cases.all_cases()}
    missing = [case_id for case_id, payload in payloads.items() if payload is None]
    if missing:  # checked before anything is deleted, so a partial cache never empties the fixtures
        sys.exit(f"not analysed for {model_key}: {', '.join(missing)}; run jspace-demo precompute --model {model_key}")
    target.mkdir(parents=True, exist_ok=True)
    for old in target.glob("*.json"):
        old.unlink()
    for case_id, payload in payloads.items():
        out = target / f"{case_id}.json"
        out.write_text(json.dumps(trim(payload), ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
        print(f"{out.relative_to(ROOT)}  {out.stat().st_size / 1024:6.1f} KiB")


if __name__ == "__main__":
    main()
