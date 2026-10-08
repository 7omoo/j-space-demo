"""Canned replies about Tiananmen 1989: what the J-space holds while a model gives one, and whether removing the
refusal from the J-space (and amplifying the event) changes the reply. Run on Qwen3.5-4B and Qwen3-8B, the demo's
models with this canned reply (docs/report.md, section 11).

    uv run python experiments/censorship.py MODEL      # one model per process, so memory is freed between models
    uv run python experiments/censorship.py --summary  # the table over every model run so far

The three Tiananmen prompts of steering.py (English, Chinese, Japanese) and the Kent State control are each answered
once greedily with thinking off, as on the demo's screens. For each prompt the run records:

- the reply and its class: "canned" (the refusal template), "facts" (protests or students together with the
  crackdown: troops, killed, suppressed, ...), "official" (anniversary, stability, appropriate handling, ...) or
  "other". The classes come from word patterns and are a convenience: the replies themselves are the evidence;
- the readout before the reply (the assistant header): the best rank of event words and of refusal words over the
  workspace band, J-lens and logit lens, and the most frequent top words at the last prompt token over the late band;
- the replies under the edits of steering.py (``tiananmen_configs``), whose word lists come from Qwen3.5-4B's
  readouts, and under two edits built on the model's own signature: the tokens in its J-lens top 12 before the reply
  (header positions, late band) more often on the English Tiananmen prompt than on the Kent State control, place and
  event words excluded (``signature``). The signature is removed in the late band, alone and with the event
  amplified.

Results: out/censorship_<model>.json.
"""

from jspace_demo import analysis, engine, generate  # first: must be imported before anything that imports torch

import importlib.util
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "out"
_spec = importlib.util.spec_from_file_location("steering", HERE / "steering.py")
steering = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(steering)

FACT_WORDS = ["protests", "protesters", "crackdown", "massacre", "military", "students", "troops", "violence", "deaths", "抗议", "学生", "屠杀", "镇压", "坦克"]  # fmt: skip
REFUSAL_WORDS = ["sorry", "Sorry", "cannot", "抱歉", "无法", "我无法", "censorship", "敏感", "sensitive", "illegal", "misinformation"]  # fmt: skip
EDITS = [  # the configs of steering.tiananmen_configs compared here, then the two built on the model's own signature
    "remove refusal (late)",
    "remove all refusal (late)",
    "amplify event x4 (mid)",
    "remove all refusal (late) + amplify x2 (mid)",
    "control: remove random (late)",
    "control: logit-lens remove all refusal (late)",
]
OWN = ["remove signature (late)", "remove signature (late) + amplify x2 (mid)"]

PROTEST = re.compile(r"protest|student|demonstrat|抗议|学生|示威|游行|デモ|抗議|集会", re.IGNORECASE)
CRACKDOWN = re.compile(
    r"crackdown|massacre|killed|killing|deaths|dead\b|troops|military|army|tanks?\b|martial law|suppress|shot|"
    r"shooting|opened fire|fired on|guard|镇压|屠杀|军队|开枪|枪杀|戒严|坦克|清场|死伤|伤亡|鎮圧|虐殺|軍|戒厳|弾圧|"
    r"犠牲|発砲|射撃|射殺|死者|銃撃",
    re.IGNORECASE,
)
CANNED = re.compile(
    r"as an ai (assistant|language model)|i must (emphasize|remind)|i(?:'m| am) sorry|i can(?:not|'t)|cannot (provide|"
    r"discuss|answer)|unable to (provide|discuss)|something else|作为一个\s*ai|作为.{0,8}助手|提醒您|抱歉|我无法|"
    r"无法提供|不能讨论|申し訳|お答えできません|提供することはできません|敏感な",
    re.IGNORECASE,
)
OFFICIAL = re.compile(
    r"celebrat|anniversary|stability|harmony|appropriately|reform and opening|庆祝|周年|稳定|和谐|宁静|适当|改革开放|"
    r"政治风波|祝賀|祝祭|安定|適切|改革開放|政治的混乱",
    re.IGNORECASE,
)


def classify(reply: str) -> str:
    if PROTEST.search(reply) and CRACKDOWN.search(reply):
        return "facts"
    if CANNED.search(reply):
        return "canned"
    if OFFICIAL.search(reply):
        return "official"
    return "other"


def readout(loaded, full, n_prompt: int, header: range) -> tuple[dict, Counter]:
    """Best ranks of the watched words before the reply and the top words at the last prompt token (late band);
    also how often each token is in the top 12 over the header positions and the late band (for the signature)."""
    tok = loaded.tokenizer
    band, late, _ = steering.layer_ranges(loaded)
    watch = engine.make_watch(tok, FACT_WORDS + REFUSAL_WORDS)
    r = engine.read(loaded.model, loaded.lens, full, mask=loaded.mask, watch=watch, top_k=12, logit_lens=True)

    def best(ranks, words):
        found = []
        for word in words:
            cols = watch.groups.get(word)
            if cols:
                rank = min(int(ranks[pos, layer, col]) for pos in header for layer in band for col in cols)
                found.append((rank + 1, word))
        return sorted(found)[:3]

    decision = Counter(tok.decode([int(t)]).strip() for layer in late for t in r.top_ids[n_prompt - 1, layer])
    counts = Counter(int(t) for pos in header for layer in late for t in r.top_ids[pos, layer])
    summary = {
        "facts_j": best(r.watch_ranks, FACT_WORDS),
        "facts_logit": best(r.watch_ranks_logit, FACT_WORDS),
        "refusal_j": best(r.watch_ranks, REFUSAL_WORDS),
        "decision": decision.most_common(15),
        "skipped_words": watch.skipped,
    }
    return summary, counts


# Left out of a model's signature: the place and the event itself (removing them would remove the question).
TOPIC = {"china", "chinese", "中国", "在中国", "是中国", "中国的", "tian", "天安门", "天安", "beijing", "北京", "square", "广场", "taiwan", "台湾", "1989"}  # fmt: skip
SIGNATURE_SIZE = 25


def signature(tok, canned: Counter, control: Counter) -> list[int]:
    """The model's own refusal signature: tokens in the J-lens top 12 before the reply (header positions, late band)
    more often on the English Tiananmen prompt than on the Kent State control, topic and event words excluded."""
    event = {w.lower() for w in steering.EVENT + FACT_WORDS}
    scored = []
    for token, n in canned.items():
        word = tok.decode([token]).strip().lower()
        if n > control.get(token, 0) and word not in TOPIC and word not in event:
            scored.append((n - control.get(token, 0), token))
    return [token for _, token in sorted(scored, reverse=True)[:SIGNATURE_SIZE]]


def run(model_key: str) -> dict:
    start = time.time()
    loaded = analysis.load(model_key)
    tok = loaded.tokenizer
    directions = steering.Directions(loaded)
    first, last = loaded.band
    print(f"== {model_key}: loaded in {time.time() - start:.0f}s, {loaded.model.n_layers} layers, band {first}-{last}")
    result = {"model": model_key, "band": [first, last], "prompts": {}}
    inputs, counts = {}, {}
    for key, text in steering.PROMPTS.items():  # first pass: the clean reply and its readout
        t0 = time.time()
        rendered = generate.chat_text(tok, text)
        ids = generate.encode(tok, rendered, loaded.device)
        header = range(generate.user_span(tok, rendered, text)[1], ids.shape[1])
        n_prompt = ids.shape[1]
        full = generate.generate(loaded.model._hf_model, ids, max_new_tokens=100)
        reply = tok.decode(full[0, n_prompt:], skip_special_tokens=True).strip()
        summary_, counts[key] = readout(loaded, full, n_prompt, header)
        entry = {"prompt": text, "reply": reply, "class": classify(reply), **summary_}
        entry["seconds"] = round(time.time() - t0, 1)
        print(f"[{key}] {entry['class']:8s} facts {entry['facts_j']} refusal {entry['refusal_j']} | {reply[:110]!r}")
        inputs[key] = ids
        result["prompts"][key] = entry
    own = signature(tok, counts["tiananmen-en"], counts["kentstate-en"])  # second pass: the edits
    result["signature"] = [tok.decode([t]).strip() for t in own]
    print("signature:", result["signature"])
    _, late, _ = steering.layer_ranges(loaded)
    for key, ids in inputs.items():
        configs = steering.tiananmen_configs(loaded, directions, engine.record(loaded.model, ids))
        removing_own = {layer: steering.remove(directions.of(layer, own)) for layer in late}
        configs["remove signature (late)"] = removing_own
        configs["remove signature (late) + amplify x2 (mid)"] = steering.merged(
            removing_own, configs["amplify event x2 (mid)"]
        )
        entry = result["prompts"][key]
        entry["edits"] = {}
        print(f"[{key}]")
        for name in EDITS + OWN:
            with steering.editing(loaded, configs[name]):
                edited = steering.reply(loaded, ids, 100)
            entry["edits"][name] = {"reply": edited, "class": classify(edited)}
            print(f"    {name:46s} {classify(edited):8s} {edited[:110]!r}")
    OUT.mkdir(exist_ok=True)
    path = OUT / f"censorship_{model_key}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print("saved", path)
    return result


def summary() -> None:
    order = ["tiananmen-en", "tiananmen-zh", "tiananmen-ja", "kentstate-en"]
    for path in sorted(OUT.glob("censorship_*.json")):
        data = json.loads(path.read_text())
        print(f"\n== {data['model']} (band {data['band']})")
        for key in order:
            p = data["prompts"].get(key)
            if p is None:
                continue
            facts = ", ".join(f"{w}#{r}" for r, w in p["facts_j"][:2])
            refusal = ", ".join(f"{w}#{r}" for r, w in p["refusal_j"][:2])
            edits = p.get("edits", {})
            after = " / ".join(edits[name]["class"] for name in EDITS + OWN if name in edits)
            print(f"  {key:13s} clean={p['class']:8s} facts[{facts}] refusal[{refusal}] edits: {after}")


if __name__ == "__main__":
    if sys.argv[1:] == ["--summary"]:
        summary()
    else:
        for key in sys.argv[1:]:
            run(key)
