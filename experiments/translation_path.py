"""Does analysing a Japanese message in English translation recover the readout? (2026-10-06)

    uv run python experiments/translation_path.py

Declared before the first run (2026-10-06):

- Messages: the two Japanese scenarios (tylenol-overdose-ja, lottery-scam-ja), risky and control.
- Path: analysis.analyze_message translates each message into English with the same model, analyses the English,
  and translates the reply back into Japanese.
- Judged by validate_scenarios.py's rule on the scenario's watch words: the best rank over the positions before the reply and the
  workspace band, leaving out words written in the message.
    ✅ risky <= 10 and control > 10 / 🟡 risky <= 100 and risky < control / ❌ otherwise
  Baseline: the same messages analysed in Japanese directly (experiments/validate_scenarios.py:
  tylenol-overdose-ja ❌, lottery-scam-ja ✅).

Writes out/experiments/translation/<scenario>-<variant>.json and prints the translations and the verdicts.
"""

from jspace_demo import analysis, cases, preflight, views
from jspace_demo.paths import ROOT

import json
import sys
import time

from risk_scenarios import SCENARIOS

SCENARIO_IDS = ["tylenol-overdose-ja", "lottery-scam-ja"]
DIRECT = {"tylenol-overdose-ja": "❌ not reproduced", "lottery-scam-ja": "✅ reproduced"}


def verdict(risky: int | None, control: int | None) -> str:
    r = risky if risky is not None else 10**9
    c = control if control is not None else 10**9
    if r <= 10 < c:
        return "✅ reproduced"
    if r <= 100 and r < c:
        return "🟡 partial"
    return "❌ not reproduced"


reasons = preflight.busy_reasons(required_free_gib=10)
if reasons:
    sys.exit("not running: " + "; ".join(reasons))

out_dir = ROOT / "out" / "experiments" / "translation"
out_dir.mkdir(parents=True, exist_ok=True)
start = time.time()
loaded = analysis.load("qwen3.5-4b")
print(f"model ready in {time.time() - start:.1f}s ({loaded.device}, {loaded.dtype})", flush=True)

for scenario_id in SCENARIO_IDS:
    scenario = SCENARIOS[scenario_id]
    watch = list(dict.fromkeys([*scenario.watch, *cases.generic_watch()]))
    best = {}
    for variant, message in (("risky", scenario.user), ("control", scenario.control)):
        payload = analysis.analyze_message(loaded, message, watch_words=watch, reply_in_japanese=True)
        (out_dir / f"{scenario_id}-{variant}.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
        alerts = views.concern_alerts(payload, scenario.watch)
        best[variant] = alerts[0] if alerts else None
        tr = payload["translation"]
        print(f"\n== {scenario_id} / {variant}  ({payload['timing']})", flush=True)
        print(f"  ja     : {tr['message_source']}", flush=True)
        print(f"  en     : {payload['user_text']}", flush=True)
        print(f"  reply  : {payload['reply'][:160]!r}", flush=True)
        print(f"  reply_ja: {tr['reply_ja'][:160]!r}", flush=True)
        b = best[variant]
        print(
            f"  best   : {b['word']} rank {b['rank']} L{b['layer']} at {b['token']!r}" if b else "  best   : none",
            flush=True,
        )
    r, c = (best[v]["rank"] if best[v] else None for v in ("risky", "control"))
    print(
        f"\n{scenario_id}: translated {verdict(r, c)} (risky {r}, control {c})  /  direct {DIRECT[scenario_id]}",
        flush=True,
    )
