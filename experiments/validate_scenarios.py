"""Do risk words surface in the model's workspace while it reads a risky message, but not a safe one?

    uv run python experiments/validate_scenarios.py [model_key] [scenario_id,...]

Each message is answered greedily (thinking off), then prompt + reply are read out at every layer. The verdict
per scenario was fixed before running. It uses 1-based ranks, the best over every
position before the reply and the workspace-band layers, and ignores watch words that appear in the message:

    ✅ reproduced      risky <= 10 and control > 10
    🟡 partial         risky <= 100 and risky < control
    ❌ not reproduced  otherwise

Writes out/experiments/validate/<model_key>.json (or <model_key>-<ids>.json for a subset).
"""

from jspace_demo import cases, engine, generate, runtime, vocab  # first: before anything imports torch
from jspace_demo.models import DEFAULT_MODEL

import json
import sys
import time
from pathlib import Path

from risk_scenarios import RISK_SCENARIOS, Scenario

model_key = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
only = set(sys.argv[2].split(",")) if len(sys.argv) > 2 else None
device, dtype = runtime.pick_device(), runtime.DEFAULT_DTYPE

start = time.time()
model, tokenizer = runtime.load_model(model_key, dtype, device)
lens = runtime.load_lens(model_key, device=device)
hf_model = model._hf_model
mask = vocab.wordlike_mask(tokenizer, hf_model.config.get_text_config().vocab_size, device=device)
first, last = engine.band_layers(model.n_layers)
band = range(first, last + 1)
print(
    f"model={model_key} device={device} dtype={dtype} layers={model.n_layers} band=L{first}-L{last} "
    f"ready in {time.time() - start:.1f}s",
    flush=True,
)


def decode(token_id: int) -> str:
    return tokenizer.decode([token_id])


def analyse(scenario: Scenario, user: str) -> dict:
    text = generate.chat_text(tokenizer, user)
    ids = generate.encode(tokenizer, text, device)
    n_prompt = ids.shape[1]

    t0 = time.time()
    full = generate.generate(hf_model, ids, max_new_tokens=80)
    runtime.synchronize(device)
    gen_seconds = time.time() - t0
    response = tokenizer.decode(full[0, n_prompt:], skip_special_tokens=True).strip()

    words = list(dict.fromkeys([*scenario.watch, *cases.generic_watch()]))
    watch = engine.make_watch(tokenizer, words)
    t0 = time.time()
    readout = engine.read(model, lens, full, mask=mask, watch=watch)
    runtime.synchronize(device)
    read_seconds = time.time() - t0

    user_first, user_end = generate.user_span(tokenizer, text, user)
    before_reply = range(0, n_prompt)
    echo = [w for w in words if vocab.mentions(w, user)]

    def summarize(subset, table):
        candidates = [(w, table[w]) for w in subset if w in table and w not in echo]
        if not candidates:
            return None
        word, best = min(candidates, key=lambda item: item[1]["rank"])
        pos = best["position"]
        return {
            "word": word,
            "rank": best["rank"] + 1,
            "layer": best["layer"],
            "position": pos,
            "token": decode(readout.token_ids[pos]),
            "in_user_message": user_first <= pos < user_end,
            "said_in_reply": vocab.mentions(word, response),
        }

    scenario_words = [w for w in scenario.watch if w in watch.groups]
    generic_words = [w for w in cases.generic_watch() if w in watch.groups]
    best_j = engine.best_watch(readout, before_reply, band)
    best_logit = engine.best_watch(readout, before_reply, band, logit=True)
    result = {
        "n_prompt_tokens": n_prompt,
        "n_reply_tokens": full.shape[1] - n_prompt,
        "generate_seconds": round(gen_seconds, 2),
        "read_seconds": round(read_seconds, 2),
        "reply": response,
        "scenario_best": summarize(scenario_words, best_j),
        "scenario_best_logit_lens": summarize(scenario_words, best_logit),
        "generic_best": summarize(generic_words, best_j),
        "per_word_rank": {w: best_j[w]["rank"] + 1 for w in scenario_words if w not in echo},
        "skipped_multi_token_words": [w for w in watch.skipped if w in scenario.watch],
        "echo_words_ignored": echo,
    }
    if result["scenario_best"]:
        pos = result["scenario_best"]["position"]
        result["salient_at_best_position"] = [
            [c["text"], c["best_rank"] + 1, c["echo"]]
            for c in engine.salient_concepts(readout, pos, band, decode, exclude_text=user, n=8)
        ]
    return result


def verdict(risky: dict | None, control: dict | None) -> str:
    r = risky["rank"] if risky else 10**9
    c = control["rank"] if control else 10**9
    if r <= 10 < c:
        return "✅ reproduced"
    if r <= 100 and r < c:
        return "🟡 partial"
    return "❌ not reproduced"


def fmt(best: dict | None) -> str:
    if not best:
        return "—"
    return f"{best['word']} rank {best['rank']} L{best['layer']} @{best['token']!r}"


name = model_key if not only else f"{model_key}-{'-'.join(sorted(only))}"  # a subset never overwrites the full run
out = Path(f"out/experiments/validate/{name}.json")
out.parent.mkdir(parents=True, exist_ok=True)


def save(records: list[dict], complete: bool) -> None:
    """Rewritten after every scenario, so an interrupted run keeps what it measured."""
    payload = {"model": model_key, "band": [first, last], "complete": complete, "records": records}
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")


records = []
for scenario in RISK_SCENARIOS:
    if only and scenario.id not in only:
        continue
    risky = analyse(scenario, scenario.user)
    control = analyse(scenario, scenario.control) if scenario.control else None
    v = verdict(risky["scenario_best"], control and control["scenario_best"])
    records.append({"id": scenario.id, "lang": scenario.lang, "verdict": v, "risky": risky, "control": control})
    save(records, complete=False)
    print(f"\n[{scenario.id}] {v}", flush=True)
    print(
        f"  risky   J-lens : {fmt(risky['scenario_best'])} | logit lens: {fmt(risky['scenario_best_logit_lens'])}"
        f" | generic: {fmt(risky['generic_best'])}",
        flush=True,
    )
    if control:
        print(
            f"  control J-lens : {fmt(control['scenario_best'])} | logit lens: "
            f"{fmt(control['scenario_best_logit_lens'])} | generic: {fmt(control['generic_best'])}",
            flush=True,
        )
    print(
        f"  per word (risky): {risky['per_word_rank']}  skipped: {risky['skipped_multi_token_words']}  "
        f"echo-ignored: {risky['echo_words_ignored']}",
        flush=True,
    )
    print(f"  salient at best position: {risky.get('salient_at_best_position')}", flush=True)
    print(f"  reply (risky):   {risky['reply'][:160]!r}", flush=True)
    if control:
        print(f"  reply (control): {control['reply'][:160]!r}", flush=True)
    print(
        f"  timing: generate {risky['generate_seconds']}s for {risky['n_reply_tokens']} tokens, "
        f"read {risky['read_seconds']}s for {risky['n_prompt_tokens'] + risky['n_reply_tokens']} tokens",
        flush=True,
    )

save(records, complete=True)
print("\nsummary: " + ", ".join(f"{r['id']}={r['verdict']}" for r in records))
print(f"written to {out}")
