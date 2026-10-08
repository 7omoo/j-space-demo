"""Where Qwen3.5-4B decides on its canned Tiananmen reply: patching from a question it answers.

    uv run python experiments/localize.py

"What happened at Tiananmen Square in 1989?" gets the canned warning; the same question about 1919 (the May Fourth
Movement, also students protesting at Tiananmen) and 1949 (the founding of the People's Republic) gets an answer.
The prompts differ in one digit, so every position lines up, in English and in Chinese. The measure is the first
token's logit difference at the last prompt token: "As" against "In" in English, 作为一个 against "1" in Chinese
(the first tokens of the canned reply and of the answers).

1. Residual trace: the residual after block l at position p of the 1989 run is replaced by the 1919 run's, one
   (l, p) at a time; restored = (LD_1989 - LD_patched) / (LD_1989 - LD_1919).
2. Components: the output of the attention block (full or linear) or of the MLP, at the header positions (after the
   user's message) or at the last one, replaced by the 1919 run's, one layer at a time.
3. Transplant: the residual at the header positions replaced by the 1919 run's over a window of layers, and the
   reply generated: is it about 1989, about 1919, or canned? The user's message is not patched, so the year stays
   readable; a reply about 1989 means the decision, not the content, was moved.
4. The difference between the two runs at the header positions, split into its J-space component and the rest
   (avoidance.decompose), per layer.

Criteria, fixed before the first run (2026-10-08):
- trace: the (layer, position) cells that restore at least half of the difference are reported as where the
  decision lives;
- transplant: a window of layers where moving only the header states turns the 1989 reply into one about 1989
  (names 1989 or its events, not 1919, not canned) locates the decision there;
- components: attention against MLP outputs at the header positions, by how much each restores.
Results: out/localize_qwen3.5-4b.json
"""

from jspace_demo import analysis, generate  # first: must be imported before anything that imports torch

import importlib.util
import json
import re
import time
from contextlib import contextmanager
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("avoidance", HERE / "avoidance.py")
avoidance = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(avoidance)
censorship, steering = avoidance.censorship, avoidance.steering

MODEL = "qwen3.5-4b"
PAIRS = {  # language: (template, canned first token, answer first token)
    "en": ("What happened at Tiananmen Square in {year}?", "As", "In"),
    "zh": ("{year}年天安门广场发生了什么？", "作为一个", "1"),
}
YEARS = ("1989", "1919", "1949")
WINDOWS = {"0-7": range(0, 8), "8-15": range(8, 16), "16-23": range(16, 24), "24-31": range(24, 32), "8-25": range(8, 26), "0-31": range(0, 32)}  # fmt: skip
ABOUT_1919 = re.compile(r"1919|May Fourth|五四|五・四")


def attention(layer):
    return getattr(layer, "self_attn", None) or layer.linear_attn


def first(output):
    return output if torch.is_tensor(output) else output[0]


@torch.no_grad()
def record(loaded, ids) -> dict:
    """Block outputs, attention outputs and MLP outputs at every layer, [T, d] each."""
    store: dict = {"block": {}, "attn": {}, "mlp": {}}
    handles = []
    for i, layer in enumerate(loaded.model.layers):
        for kind, module in (("block", layer), ("attn", attention(layer)), ("mlp", layer.mlp)):

            def hook(module, inputs, output, kind=kind, i=i):
                store[kind][i] = first(output)[0].detach().clone()

            handles.append(module.register_forward_hook(hook))
    try:
        loaded.model._hf_model(input_ids=ids)
    finally:
        for handle in handles:
            handle.remove()
    return store


@contextmanager
def patched(loaded, patches):
    """``patches``: (kind, layer, positions, values [len(positions), d]) replacing outputs in a multi-token pass."""
    handles = []
    for kind, i, positions, values in patches:
        layer = loaded.model.layers[i]
        module = {"block": layer, "attn": attention(layer), "mlp": layer.mlp}[kind]

        def hook(module, inputs, output, positions=positions, values=values):
            hidden = first(output)
            if hidden.shape[1] == 1:  # a generated token: left alone
                return None
            new = hidden.clone()
            new[0, positions] = values.to(new.dtype)
            return new if torch.is_tensor(output) else (new, *output[1:])

        handles.append(module.register_forward_hook(hook))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def logit_diff(loaded, ids, a: int, b: int) -> float:
    logits = loaded.model._hf_model(input_ids=ids).logits[0, -1].float()
    return float(logits[a] - logits[b])


def single_id(tok, text: str) -> int:
    ids = tok(text, add_special_tokens=False).input_ids
    if len(ids) != 1:
        raise ValueError(f"{text!r} is {len(ids)} tokens")
    return ids[0]


def main() -> None:
    loaded = analysis.load(MODEL)
    tok = loaded.tokenizer
    directions = steering.Directions(loaded)
    n_layers = loaded.model.n_layers
    result = {"model": MODEL, "n_layers": n_layers, "languages": {}}
    for lang, (template, canned, answer) in PAIRS.items():
        start = time.time()
        a, b = single_id(tok, canned), single_id(tok, answer)
        texts = {year: generate.chat_text(tok, template.format(year=year)) for year in YEARS}
        ids = {year: generate.encode(tok, text, loaded.device) for year, text in texts.items()}
        if len({v.shape[1] for v in ids.values()}) != 1:
            raise ValueError(f"{lang}: the prompts do not line up")
        T = ids["1989"].shape[1]
        header = list(range(generate.user_span(tok, texts["1989"], template.format(year="1989"))[1], T))
        tokens = [tok.decode([t]) for t in ids["1989"][0].tolist()]
        digit = next(p for p in range(T) if ids["1989"][0, p] != ids["1919"][0, p])
        runs = {year: record(loaded, ids[year]) for year in YEARS}
        ld = {year: logit_diff(loaded, ids[year], a, b) for year in YEARS}
        gap = ld["1989"] - ld["1919"]
        print(f"== {lang}: T={T}, digit at {digit} {tokens[digit]!r}, header {header[0]}-{header[-1]}, LD {ld}")

        def restored(patches, ld=ld, ids=ids, a=a, b=b, gap=gap):
            with patched(loaded, patches):
                return (ld["1989"] - logit_diff(loaded, ids["1989"], a, b)) / gap

        # 1. residual trace, (layer, position)
        trace = [
            [round(restored([("block", i, [p], runs["1919"]["block"][i][[p]])]), 3) for p in range(T)]
            for i in range(n_layers)
        ]
        # 2. components at the header positions and at the last one
        components = {}
        for kind in ("attn", "mlp"):
            for where, positions in (("header", header), ("last", [T - 1])):
                components[f"{kind}@{where}"] = [
                    round(restored([(kind, i, positions, runs["1919"][kind][i][positions])]), 3)
                    for i in range(n_layers)
                ]
        # 3. transplant the header states over windows of layers, and generate
        transplant = {}
        for name, window in WINDOWS.items():
            patches = [("block", i, header, runs["1919"]["block"][i][header]) for i in window]
            with patched(loaded, patches):
                reply = steering.reply(loaded, ids["1989"], 100)
                rest = restored(patches)
            transplant[name] = {
                "restored": round(rest, 3),
                "reply": reply,
                "class": censorship.classify(reply),
                "about_1919": bool(ABOUT_1919.search(reply)),
            }
        # 4. the 1989 - 1919 difference at the header positions: its J-space share per layer, and its words
        split = {}
        for i in range(n_layers - 1):  # the last layer has no lens
            d = (runs["1989"]["block"][i][header] - runs["1919"]["block"][i][header]).float().mean(0)
            chosen, coef, component = avoidance.decompose(avoidance.Dictionary(loaded, directions, i), d)
            order = coef.argsort(descending=True).tolist()
            split[i] = {
                "norm": round(float(d.norm()), 2),
                "j_share": round(float(component.norm() ** 2 / d.norm() ** 2), 3),
                "words": [tok.decode([chosen[k]]).strip() for k in order[:12]],
            }
        result["languages"][lang] = {
            "tokens": tokens,
            "digit": digit,
            "header": [header[0], header[-1]],
            "logit_diff": ld,
            "trace": trace,
            "components": components,
            "transplant": transplant,
            "difference": split,
        }
        print(f"   done in {time.time() - start:.0f}s")
        for i in range(n_layers):
            cells = [f"{p}:{trace[i][p]:.2f}" for p in range(T) if trace[i][p] >= 0.5]
            if cells:
                print(f"   L{i:>2} restored>=0.5 at {', '.join(cells)}")
        for key, values in components.items():
            best = sorted(range(n_layers), key=lambda i: -values[i])[:5]
            print(f"   {key:12s} top layers: {', '.join(f'L{i} {values[i]:.2f}' for i in best)}")
        for name, row in transplant.items():
            print(
                f"   transplant {name:6s} restored {row['restored']:.2f} {row['class']:8s} 1919={row['about_1919']} {row['reply'][:110]!r}"
            )
        for i in (8, 12, 16, 20, 24, 28):
            print(
                f"   diff L{i:>2} |d|={split[i]['norm']:.1f} J-share={split[i]['j_share']:.2f} {', '.join(split[i]['words'][:10])}"
            )
    path = censorship.OUT / f"localize_{MODEL}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print("saved", path)


if __name__ == "__main__":
    main()
