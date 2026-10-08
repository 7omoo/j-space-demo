"""Which attention heads carry Qwen3.5-4B's decision to give the canned Tiananmen reply (after localize.py).

    uv run python experiments/heads.py

localize.py found the decision at the header positions (after the user's message) from layer 8 on, with the
attention of layer 19 the strongest single block. Here, for the eight full-attention layers (3, 7, ..., 31; the
others are linear attention), each head's output at the header positions (its slice of the input to o_proj) is
replaced by the 1919 run's, and the first-token logit difference measured as in localize.py. For the heads that
restore most:

- where they read: their attention weights from the header positions, in the 1989 run;
- what they write: the 1989 - 1919 difference of their contribution (o_proj applied to their slice), split into its
  J-space component and the rest, with the component's words;
- together: the top 1, 3 and 5 heads replaced at once, and the reply generated.

Criteria, fixed before the first run (2026-10-08): a head restoring at least 0.2 alone carries part of the
decision; the top three together restoring at least 0.5 and turning the reply away from the canned one would put
the decision in a few heads.
Results: out/heads_qwen3.5-4b.json
"""

from jspace_demo import analysis, generate  # first: must be imported before anything that imports torch

import importlib.util
import json
import time
from contextlib import contextmanager
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("localize", HERE / "localize.py")
localize = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(localize)
avoidance, censorship, steering = localize.avoidance, localize.censorship, localize.steering


def full_attention_layers(loaded) -> list[int]:
    return [i for i, layer in enumerate(loaded.model.layers) if getattr(layer, "self_attn", None) is not None]


@torch.no_grad()
def record_heads(loaded, ids, layers) -> dict[int, torch.Tensor]:
    """The input of o_proj (the heads' outputs, concatenated) at each full-attention layer, [T, heads * head_dim]."""
    store, handles = {}, []
    for i in layers:

        def hook(module, args, i=i):
            store[i] = args[0][0].detach().clone()

        handles.append(loaded.model.layers[i].self_attn.o_proj.register_forward_pre_hook(hook))
    try:
        loaded.model._hf_model(input_ids=ids)
    finally:
        for handle in handles:
            handle.remove()
    return store


@contextmanager
def heads_patched(loaded, patches, head_dim: int):
    """``patches``: (layer, head, positions, values [len(positions), head_dim]) replacing that head's output."""
    by_layer: dict[int, list] = {}
    for i, h, positions, values in patches:
        by_layer.setdefault(i, []).append((h, positions, values))
    handles = []
    for i, items in by_layer.items():

        def hook(module, args, items=items):
            x = args[0]
            if x.shape[1] == 1:  # a generated token: left alone
                return None
            x = x.clone()
            for h, positions, values in items:
                x[0, positions, h * head_dim : (h + 1) * head_dim] = values.to(x.dtype)
            return (x, *args[1:])

        handles.append(loaded.model.layers[i].self_attn.o_proj.register_forward_pre_hook(hook))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def attention_from(loaded, ids, layer: int, queries: list[int], layers: list[int]) -> torch.Tensor:
    """Attention weights [heads, T] of one full-attention layer, averaged over the query positions (eager)."""
    hf = loaded.model._hf_model
    previous = hf.config._attn_implementation
    hf.set_attn_implementation("eager")
    try:
        out = hf(input_ids=ids, output_attentions=True)
    finally:
        hf.set_attn_implementation(previous)
    attentions = [a for a in out.attentions if a is not None]
    weights = attentions[layers.index(layer)] if len(attentions) == len(layers) else out.attentions[layer]
    return weights[0, :, queries, :].float().mean(1)  # [heads, T]


def main() -> None:
    loaded = analysis.load(localize.MODEL)
    tok = loaded.tokenizer
    directions = steering.Directions(loaded)
    cfg = loaded.model._hf_model.config.get_text_config()
    n_heads, head_dim = cfg.num_attention_heads, cfg.head_dim
    layers = full_attention_layers(loaded)
    result = {"model": localize.MODEL, "full_attention_layers": layers, "languages": {}}
    for lang, (template, canned, answer) in localize.PAIRS.items():
        start = time.time()
        a, b = localize.single_id(tok, canned), localize.single_id(tok, answer)
        texts = {y: generate.chat_text(tok, template.format(year=y)) for y in ("1989", "1919")}
        ids = {y: generate.encode(tok, t, loaded.device) for y, t in texts.items()}
        T = ids["1989"].shape[1]
        header = list(range(generate.user_span(tok, texts["1989"], template.format(year="1989"))[1], T))
        tokens = [tok.decode([t]) for t in ids["1989"][0].tolist()]
        heads = {y: record_heads(loaded, ids[y], layers) for y in ids}
        ld = {y: localize.logit_diff(loaded, ids[y], a, b) for y in ids}
        gap = ld["1989"] - ld["1919"]

        def restored(patches, ld=ld, ids=ids, a=a, b=b, gap=gap):
            with heads_patched(loaded, patches, head_dim):
                return (ld["1989"] - localize.logit_diff(loaded, ids["1989"], a, b)) / gap

        def patch(i, h, heads_=heads, header_=header):
            return (i, h, header_, heads_["1919"][i][header_, h * head_dim : (h + 1) * head_dim])

        single = {(i, h): round(restored([patch(i, h)]), 3) for i in layers for h in range(n_heads)}
        ranked = sorted(single, key=lambda k: -single[k])
        together = {}
        for k in (1, 3, 5, 8):
            chosen = ranked[:k]
            patches = [patch(i, h) for i, h in chosen]
            with heads_patched(loaded, patches, head_dim):
                reply = steering.reply(loaded, ids["1989"], 100)
            together[k] = {
                "heads": [f"L{i}.H{h}" for i, h in chosen],
                "restored": round(restored(patches), 3),
                "reply": reply,
                "class": censorship.classify(reply),
            }
        top = {}
        for i, h in ranked[:4]:
            weights = attention_from(loaded, ids["1989"], i, header, layers)[h]
            reads = sorted(range(T), key=lambda p: -float(weights[p]))[:5]
            o_proj = loaded.model.layers[i].self_attn.o_proj.weight[:, h * head_dim : (h + 1) * head_dim].float()
            diff = (
                heads["1989"][i][header, h * head_dim : (h + 1) * head_dim].float()
                - heads["1919"][i][header, h * head_dim : (h + 1) * head_dim].float()
            ).mean(0) @ o_proj.T
            row = {"restored": single[(i, h)], "reads": [(p, tokens[p], round(float(weights[p]), 3)) for p in reads]}
            if i in loaded.lens.jacobians:  # the last layer has no lens
                chosen, coef, component = avoidance.decompose(avoidance.Dictionary(loaded, directions, i), diff)
                order = coef.argsort(descending=True).tolist()
                row["writes_j_share"] = round(float(component.norm() ** 2 / diff.norm() ** 2), 3)
                row["writes"] = [tok.decode([chosen[k]]).strip() for k in order[:12]]
            top[f"L{i}.H{h}"] = row
        result["languages"][lang] = {
            "tokens": tokens,
            "header": [header[0], header[-1]],
            "logit_diff": ld,
            "single": {f"L{i}.H{h}": v for (i, h), v in single.items()},
            "together": together,
            "top": top,
        }
        print(f"== {lang}: LD {ld}, done in {time.time() - start:.0f}s")
        print("   top heads:", ", ".join(f"L{i}.H{h} {single[(i, h)]:.2f}" for i, h in ranked[:10]))
        for k, row in together.items():
            print(f"   top {k}: restored {row['restored']:.2f} {row['class']:8s} {row['reply'][:110]!r}")
        for name, row in top.items():
            reads = ", ".join(f"{p}:{t.strip() or repr(t)} {w:.2f}" for p, t, w in row["reads"])
            print(f"   {name} ({row['restored']:.2f}) reads [{reads}]")
            if "writes" in row:
                print(f"      writes J-share {row['writes_j_share']:.2f}: {', '.join(row['writes'][:10])}")
    path = censorship.OUT / f"heads_{localize.MODEL}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print("saved", path)


if __name__ == "__main__":
    main()
