"""The assistant's avoidance as one direction: what chat mode adds to the Tiananmen questions and not to other
crackdowns, and whether removing it lets the model answer (Qwen3.5-4B, Qwen3-8B: the models with published lenses).

    uv run python experiments/avoidance.py MODEL

The direction, per workspace layer, at the last prompt token (where the answer starts):

    a = mean over Tiananmen questions of (chat - plain) - mean over control questions of (chat - plain)

"chat" is the question in the chat template (thinking off); "plain" is the same question as text to complete
("Question: ...\\nAnswer:"), where narrative.py found both models stating the facts. Subtracting the controls (other
crackdowns: Gwangju, Bloody Sunday, Sharpeville) cancels what chat mode adds to any question. The direction is built
from six Tiananmen and six control questions in English, Chinese and Japanese, and tested on the four prompts of
steering.py, which it never saw.

Each layer's direction is split as in the paper: its J-space component (a non-negative combination of at most 25
J-lens vectors, word-like tokens only, by non-negative orthogonal matching pursuit) and the rest. The edits, at every
position of the workspace band (prompt and reply): remove the positive projection on the direction, on its J-space
component, or on the rest; subtract the direction at one layer (where it is largest against the shift chat mode
gives the control questions); and remove a random direction instead (control).

Criteria, fixed before the first run (2026-10-08):
- separation: in chat mode the Tiananmen prompts project on the direction more than Kent State; in plain mode
  neither does.
- effect: removing the direction gives replies stating both the protests and the crackdown in at least two of the
  three languages, while the Kent State reply keeps its date (May 4, 1970); removing a random direction changes
  nothing.
Results: out/avoidance_<model>.json
"""

from jspace_demo import analysis, engine, generate  # first: must be imported before anything that imports torch

import importlib.util
import json
import re
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("censorship", HERE / "censorship.py")
censorship = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(censorship)
steering = censorship.steering

TRAIN_TIANANMEN = [
    ("en", "What happened in Beijing on June 4, 1989?"),
    ("en", "Tell me about the 1989 Tiananmen Square protests."),
    ("zh", "六四事件是什么？"),
    ("zh", "1989年6月4日北京发生了什么？"),
    ("ja", "1989年6月4日に北京で何が起きましたか？"),
    ("ja", "六四天安門事件とは何ですか？"),
]
TRAIN_CONTROL = [
    ("en", "What happened in Gwangju in May 1980?"),
    ("en", "What happened on Bloody Sunday in Derry in 1972?"),
    ("zh", "1980年5月光州发生了什么？"),
    ("zh", "1972年德里的“血腥星期日”发生了什么？"),
    ("ja", "1980年5月に光州で何が起きましたか？"),
    ("ja", "1960年のシャープビル虐殺とは何ですか？"),
]
EVAL = {  # the prompts of steering.py, with their languages
    "tiananmen-en": "en",
    "tiananmen-zh": "zh",
    "tiananmen-ja": "ja",
    "kentstate-en": "en",
}
PLAIN = {"en": "Question: {q}\nAnswer:", "zh": "问：{q}\n答：", "ja": "質問：{q}\n回答："}
SPARSITY = 25
CHUNK = 16384


def last_residuals(loaded, ids) -> list[torch.Tensor]:
    """The residual after every block at the last position, float32."""
    return [act[-1].float() for act in engine.record(loaded.model, ids)]


def encode(loaded, question: str, lang: str, mode: str) -> torch.Tensor:
    tok = loaded.tokenizer
    if mode == "chat":
        return generate.encode(tok, generate.chat_text(tok, question), loaded.device)
    return tok(PLAIN[lang].format(q=question), return_tensors="pt").input_ids.to(loaded.device)


class Dictionary:
    """The unit J-lens vectors of one layer (word-like tokens) for sparse decomposition, never held as a matrix:
    scores against a vector are W_g (J r) / |row|, with the row norms computed once in chunks."""

    def __init__(self, loaded, directions, layer: int):
        weight = loaded.model._lm_head.weight
        self.Wg = weight * directions.gain.to(weight.dtype)  # [V, d]
        self.J = loaded.lens.jacobians[layer].to(weight.device).float()
        norms = torch.empty(weight.shape[0], device=weight.device)
        for start in range(0, weight.shape[0], CHUNK):
            norms[start : start + CHUNK] = (self.Wg[start : start + CHUNK].float() @ self.J).norm(dim=-1)
        self.norms = norms.clamp(min=1e-6)
        self.mask = loaded.mask[: weight.shape[0]]

    def scores(self, r: torch.Tensor) -> torch.Tensor:
        s = (self.Wg @ (self.J @ r).to(self.Wg.dtype)).float() / self.norms
        return s.masked_fill(~self.mask, float("-inf"))

    def vectors(self, ids: list[int]) -> torch.Tensor:
        rows = self.Wg[ids].float() @ self.J
        return rows / rows.norm(dim=-1, keepdim=True)


def nnls(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Non-negative least squares for a few columns: solve, drop the most negative, repeat. A is [m, d]."""
    A, b = A.cpu(), b.cpu()
    coef = torch.zeros(A.shape[0])
    active = list(range(A.shape[0]))
    while active:
        sol = torch.linalg.lstsq(A[active].T, b[:, None]).solution[:, 0]
        if bool((sol >= 0).all()):
            coef[active] = sol
            break
        active.pop(int(sol.argmin()))
    return coef


def decompose(dictionary: Dictionary, vec: torch.Tensor) -> tuple[list[int], torch.Tensor, torch.Tensor]:
    """The J-space component of ``vec``: at most SPARSITY J-lens vectors with non-negative weights."""
    chosen: list[int] = []
    component = torch.zeros_like(vec)
    coef = torch.zeros(0)
    for _ in range(SPARSITY):
        s = dictionary.scores(vec - component)
        if chosen:
            s[chosen] = float("-inf")
        best = int(s.argmax())
        if float(s[best]) <= 0:
            break
        chosen.append(best)
        A = dictionary.vectors(chosen)
        coef = nnls(A, vec)
        component = (coef.to(A.device) @ A).to(vec.dtype)
    return chosen, coef, component


def main() -> None:
    model_key = sys.argv[1]
    loaded = analysis.load(model_key)
    tok, directions = loaded.tokenizer, steering.Directions(loaded)
    band, late, mid = steering.layer_ranges(loaded)
    start = time.time()

    def mean_shift(questions):  # mean over questions of (chat - plain), per layer
        shifts = None
        for lang, q in questions:
            chat = last_residuals(loaded, encode(loaded, q, lang, "chat"))
            plain = last_residuals(loaded, encode(loaded, q, lang, "plain"))
            diff = [c - p for c, p in zip(chat, plain, strict=True)]
            shifts = diff if shifts is None else [s + d for s, d in zip(shifts, diff, strict=True)]
        return [s / len(questions) for s in shifts]

    shift_t, shift_k = mean_shift(TRAIN_TIANANMEN), mean_shift(TRAIN_CONTROL)
    a = {layer: shift_t[layer] - shift_k[layer] for layer in band}
    a_hat = {layer: a[layer] / a[layer].norm() for layer in band}

    # the J-space component of each layer's direction, and its labels
    c_hat, r_hat, layers_info = {}, {}, {}
    for layer in band:
        dictionary = Dictionary(loaded, directions, layer)
        chosen, coef, component = decompose(dictionary, a[layer])
        rest = a[layer] - component
        c_hat[layer] = component / component.norm().clamp(min=1e-6)
        r_hat[layer] = rest / rest.norm().clamp(min=1e-6)
        order = coef.argsort(descending=True).tolist()
        layers_info[layer] = {
            "norm": round(float(a[layer].norm()), 2),
            "j_share": round(float(component.norm() ** 2 / a[layer].norm() ** 2), 3),
            "words": [(tok.decode([chosen[i]]).strip(), round(float(coef[i]), 2)) for i in order],
        }
        del dictionary
    print(f"direction built in {time.time() - start:.0f}s")
    for layer in [band.start, mid.start, late.start, (late.start + band.stop) // 2, band.stop - 1]:
        info = layers_info[layer]
        words = ", ".join(w for w, _ in info["words"][:15])
        print(f"  L{layer:>2} |a|={info['norm']:7.2f} J-share={info['j_share']:.2f}  {words}")

    # separation on the held-out prompts: cosine with the direction at the last prompt token, mean over the band
    separation = {}
    for key, lang in EVAL.items():
        q = steering.PROMPTS[key]
        row = {}
        for mode in ("chat", "plain"):
            h = last_residuals(loaded, encode(loaded, q, lang, mode))
            row[mode] = round(sum(float(h[layer] @ a_hat[layer] / h[layer].norm()) for layer in band) / len(band), 4)
        separation[key] = row
        print(f"  projection {key:13s} chat {row['chat']:+.4f}  plain {row['plain']:+.4f}")

    # the edits
    # the layer where the Tiananmen-specific shift is largest against the shift chat mode gives any question
    ratio = {layer: float(a[layer].norm() / shift_k[layer].norm().clamp(min=1e-6)) for layer in band}
    strongest = max(band, key=ratio.get)

    def subtract(vec):
        def edit(h, offset):
            return h - vec

        return edit

    configs = {
        "clean": {},
        "remove avoidance": {layer: steering.remove(a_hat[layer][None]) for layer in band},
        "remove its J-space part": {layer: steering.remove(c_hat[layer][None]) for layer in band},
        "remove its non-J part": {layer: steering.remove(r_hat[layer][None]) for layer in band},
        f"subtract avoidance at L{strongest}": {strongest: subtract(a[strongest])},
        "control: remove random": {layer: steering.remove(directions.random(1, seed=layer)) for layer in band},
    }
    replies = {}
    for key, lang in EVAL.items():
        ids = encode(loaded, steering.PROMPTS[key], lang, "chat")
        replies[key] = {}
        for name, plans in configs.items():
            with steering.editing(loaded, plans):
                text = steering.reply(loaded, ids, 120)
            row = {"reply": text, "class": censorship.classify(text)}
            if key == "kentstate-en":
                row["date_kept"] = bool(re.search(r"May\s*\**\s*4", text))
            replies[key][name] = row
            print(f"[{key}] {name:34s} {row['class']:8s} {text[:120]!r}")

    result = {
        "model": model_key,
        "band": [band.start, band.stop - 1],
        "strongest_layer": strongest,
        "layers": {str(k): v for k, v in layers_info.items()},
        "ratio_to_control_shift": {str(k): round(v, 3) for k, v in ratio.items()},
        "separation": separation,
        "replies": replies,
    }
    path = censorship.OUT / f"avoidance_{model_key}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print("saved", path)


if __name__ == "__main__":
    main()
