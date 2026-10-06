"""The J-lens against the logit lens on the boot riddle, compared with the CPU fp32 reference (Qwen3.5-4B).

    uv run python experiments/compare_lenses.py [device] [dtype]

Prints the four-layer comparison and writes the full sweep over every fitted layer to out/compare_boot.txt. The
criterion, fixed before the first run: at layers 9-13 Italy ranks under 10 with the J-lens and over 1,000 with the
logit lens at the "boot" token.
"""

from jspace_demo import health, runtime, vocab  # first: must be imported before anything that imports torch
from jspace_demo.models import DEFAULT_MODEL

import sys
from pathlib import Path

device = sys.argv[1] if len(sys.argv) > 1 else runtime.pick_device()
dtype = sys.argv[2] if len(sys.argv) > 2 else runtime.DEFAULT_DTYPE
model, tokenizer = runtime.load_model(DEFAULT_MODEL, dtype, device)
lens = runtime.load_lens(DEFAULT_MODEL, device=device)
print(f"device={device} dtype={dtype} {lens!r}")

prompt = health.BOOT_PROMPT
REF = health.REFERENCE_4B
ITALY = REF["italy_id"]
n = model.n_layers
four = [n // 4, n // 2, n // 4 * 3, n - 2]
jl, model_logits, ids = lens.apply(model, prompt, positions=[-2, -1])
ll, _, _ = lens.apply(model, prompt, positions=[-2, -1], use_jacobian=False)
mask = vocab.wordlike_mask(tokenizer, int(model_logits.shape[-1]))


def top(logits, k=5, wordlike=False):
    if wordlike:
        logits = logits.masked_fill(~mask, float("-inf"))
    return [tokenizer.decode([t]) for t in logits.topk(k).indices]


BOOT, LAST = 0, 1  # row index into the [n_positions, vocab] tensors: positions=[-2, -1]
print(f"read position -2 = {tokenizer.decode([int(ids[0, -2])])!r}, -1 = {tokenizer.decode([int(ids[0, -1])])!r}")
print("\n== four layers at the boot position (top-5, raw vocab) ==")
for layer in four:
    got = top(jl[layer][BOOT])
    ref = REF["boot_jlens_top5"].get(layer)
    print(f"L{layer:>2} logit-lens: {top(ll[layer][BOOT])}")
    print(f"L{layer:>2} J-lens:     {got}")
    if ref:
        print(
            f"L{layer:>2} reference:  {ref}  top1 {'match' if got[0] == ref[0] else 'DIFF'}, "
            f"overlap {len(set(got) & set(ref))}/5"
        )
print(f"model next-token top-5 at boot: {top(model_logits[BOOT])}")
print(f"model next-token top-5 at the final position: {top(model_logits[LAST])}")

rows = []
for layer in lens.source_layers:
    j, lg = jl[layer][BOOT], ll[layer][BOOT]
    rows.append(
        f"L{layer:>2} | Italy rank J={health.rank_of(j, ITALY):>6} "
        f"logit={health.rank_of(lg, ITALY):>6}\n"
        f"     J-lens  top10 raw : {top(j, 10)}\n"
        f"     J-lens  top5 word : {top(j, 5, True)}\n"
        f"     logit   top10 raw : {top(lg, 10)}\n"
        f"     logit   top5 word : {top(lg, 5, True)}\n"
        f"     final position J-lens top3 word : {top(jl[layer][LAST], 3, True)}"
    )
out = Path("out/compare_boot.txt")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(f"device={device} dtype={dtype}\nprompt={prompt!r}\n\n" + "\n".join(rows) + "\n", encoding="utf-8")
print(f"\nfull sweep written to {out}")

print("\n== Italy rank (0-based) vs reference ==")
for layer, ref_rank in REF["italy_jlens_rank"].items():
    print(
        f"L{layer:>2} J-lens {health.rank_of(jl[layer][BOOT], ITALY):>6} (ref {ref_rank:>5}) | "
        f"logit {health.rank_of(ll[layer][BOOT], ITALY):>6} "
        f"(ref {REF['italy_logit_rank'][layer]:>6})"
    )
band = range(9, 14)
j_ranks = {layer: health.rank_of(jl[layer][BOOT], ITALY) for layer in band}
l_ranks = {layer: health.rank_of(ll[layer][BOOT], ITALY) for layer in band}
ok = all(r < 10 for r in j_ranks.values()) and all(r > 1000 for r in l_ranks.values())
print(f"criterion L9-13: J-lens < 10 everywhere and logit lens > 1000 everywhere -> {'PASS' if ok else 'FAIL'}")
print(f"  J-lens {j_ranks}\n  logit  {l_ranks}")
