"""Did a model and its lens load correctly? A quick check against readouts any working setup shows.

Every model is checked for a lens that fits it, finite readouts and " Paris" after "The capital of France is".
The boot riddle ("the country shaped like a boot") is read out too: Italy should surface under the J-lens at the
"boot" token in the middle layers (on Qwen3.5-4B the logit lens misses it there; on larger models it finds it at
fewer layers).

Qwen3.5-4B is also compared with CPU fp32 reference values (computed 2026-10-06 on Linux x86: torch 2.14.1,
transformers 5.18.0, jlens 581d398, lens revision qwen-n1000). Running the same weights in bf16 and fp16 on the CPU
gave the same top-1 token at all 31 lens layers.

The reference ranks below are 0-based, as upstream prints them; the result reports 1-based ranks (1 = top), like the
screens.
"""

from __future__ import annotations

from jspace_demo import engine

import torch

FRANCE_PROMPT = "The capital of France is"
BOOT_PROMPT = "Fact: The currency used in the country shaped like a boot is"
BOOT_LAYERS = [8, 16, 24, 30]  # the layers the 4B reference lists

REFERENCE_4B = {  # the parts of the CPU fp32 reference that the check compares
    "model": "qwen3.5-4b",
    "italy_id": 14898,  # ' Italy'
    "boot_jlens_top5": {
        8: [" `", " boots", " *", " `\\", " heel"],
        16: ["?", "？", "'?", "____", " Italy"],
        24: ["-shaped", " shape", " shaped", "shape", "形状"],
        30: [" is", " shape", " shaped", "-shaped", " heel"],
    },
}


def rank_of(logits: torch.Tensor, token_id: int) -> int:
    """Full-vocabulary rank of a token (0 = top)."""
    return int((logits > logits[token_id]).sum())


def _italy_id(tokenizer) -> int | None:
    for form in (" Italy", "Italy"):
        ids = tokenizer(form, add_special_tokens=False).input_ids
        if len(ids) == 1:
            return ids[0]
    return None


def check(model_key: str, model, lens, tokenizer) -> dict:
    """Run the checks; ``result["passed"]`` is the verdict and the rest says why."""

    def top5(logits):
        return [tokenizer.decode([t]) for t in logits.topk(5).indices]

    out: dict = {"model": model_key, "n_layers": model.n_layers, "d_model": model.d_model}
    out["lens_prompts"] = lens.n_prompts  # how many prompts the lens was averaged over
    # a Jacobian for every layer but the last, whose readout is the model's own output
    out["lens_fits"] = model.d_model == lens.d_model and set(lens.source_layers) == set(range(model.n_layers - 1))
    _, france, _ = lens.apply(model, FRANCE_PROMPT, layers=[lens.source_layers[0]], positions=[-1])
    out["france_model_top5"] = top5(france[0])

    first, last = engine.band_layers(model.n_layers)
    band = [layer for layer in lens.source_layers if first <= layer <= last]
    jl, model_logits, ids = lens.apply(model, BOOT_PROMPT, layers=band, positions=[-2])
    ll, _, _ = lens.apply(model, BOOT_PROMPT, layers=band, positions=[-2], use_jacobian=False)
    out["boot_read_token"] = tokenizer.decode([int(ids[0, -2])])
    out["boot_model_top5"] = top5(model_logits[0])
    italy = _italy_id(tokenizer)
    if italy is not None:
        out["italy_best_jlens_rank"] = 1 + min(rank_of(jl[layer][0], italy) for layer in band)
        out["italy_best_logit_rank"] = 1 + min(rank_of(ll[layer][0], italy) for layer in band)
    out["finite"] = all(bool(torch.isfinite(t).all()) for t in (france, model_logits, *jl.values(), *ll.values()))

    passed = out["lens_fits"] and out["finite"] and out["france_model_top5"][0] == " Paris"
    if model_key == REFERENCE_4B["model"]:
        out["reference"] = _compare_4b(model, lens, tokenizer, top5)
        passed = passed and out["reference"]["passed"]
    out["passed"] = bool(passed)
    return out


def _compare_4b(model, lens, tokenizer, top5) -> dict:
    ref = REFERENCE_4B
    layers = sorted(set(BOOT_LAYERS) | {12})
    jl, _, ids = lens.apply(model, BOOT_PROMPT, layers=layers, positions=[-2])
    jlens_top5 = {layer: top5(jl[layer][0]) for layer in BOOT_LAYERS}
    italy_l12 = rank_of(jl[12][0], ref["italy_id"])  # 0-based, like the reference
    out = {
        "boot_jlens_top5": jlens_top5,
        "top1_matches": sum(jlens_top5[layer][0] == ref["boot_jlens_top5"][layer][0] for layer in BOOT_LAYERS),
        "italy_jlens_rank_L12": italy_l12 + 1,
    }
    out["passed"] = tokenizer.decode([int(ids[0, -2])]) == " boot" and out["top1_matches"] >= 3 and italy_l12 <= 3
    return out
