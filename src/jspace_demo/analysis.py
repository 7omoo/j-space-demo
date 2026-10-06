"""One analysis: answer a message once, read every layer once (J-lens and logit lens), and shape the result.

The model-bound steps (generation, ``engine.read``) are kept apart from :func:`build_payload`, which only turns a
Readout into plain JSON, so the payload can be tested without the real model. The payload is the input of every
screen (``views.build``) and of the prepared cases (``precompute``).
"""

from __future__ import annotations

from jspace_demo import engine, generate, pressure, runtime, translate, vocab
from jspace_demo.errors import InputError
from jspace_demo.models import DEFAULT_MODEL, spec

import time
from collections.abc import Callable
from dataclasses import dataclass

import torch

LAYER_TOP_K = 8  # word-like readouts kept per (position, layer)
WATCH_TOP = 100  # a watched word's rank is kept per position wherever it is within the top 100
MAX_PROMPT_TOKENS = 1024  # generation and the per-layer readout both grow with length; longer prompts are refused
MAX_NEW_TOKENS = 80

StepCallback = Callable[[str], None] | None


@dataclass
class Loaded:
    """A model, its tokenizer and lens, and what the readout needs about them."""

    key: str
    model: object  # jlens LensModel
    tokenizer: object
    lens: object
    mask: torch.Tensor  # word-like tokens
    device: str
    dtype: str
    vocab_size: int
    band: tuple[int, int]  # the workspace layers ("thinking"), between reading and speaking

    @property
    def info(self) -> dict:
        return {
            "key": self.key,
            "hf_name": spec(self.key).hf_name,
            "label": spec(self.key).label,
            "n_layers": self.model.n_layers,
            "band": list(self.band),
            "vocab_size": self.vocab_size,
            "device": self.device,
            "dtype": self.dtype,
        }


def load(model_key: str = DEFAULT_MODEL, device: str | None = None, dtype: str | None = None) -> Loaded:
    device = device or runtime.pick_device()
    dtype = dtype or runtime.DEFAULT_DTYPE
    model, tokenizer = runtime.load_model(model_key, dtype, device)
    lens = runtime.load_lens(model_key, device=device)
    vocab_size = model._hf_model.config.get_text_config().vocab_size
    return Loaded(
        key=model_key,
        model=model,
        tokenizer=tokenizer,
        lens=lens,
        mask=vocab.wordlike_mask(tokenizer, vocab_size, device=device),
        device=device,
        dtype=dtype,
        vocab_size=vocab_size,
        band=engine.band_layers(model.n_layers),
    )


def analyze(
    loaded: Loaded,
    *,
    user: str | None = None,
    raw: str | None = None,
    watch_words=(),
    max_new_tokens: int = MAX_NEW_TOKENS,
    system: str | None = None,
    on_step: StepCallback = None,
) -> dict:
    """Answer ``user`` (chat, optionally under a ``system`` prompt) or continue ``raw`` (plain text), then read out
    prompt + reply at every layer. ``on_step(name)`` is told when each stage starts (the page's progress)."""
    if (user is None) == (raw is None):
        raise ValueError("pass exactly one of user= / raw=")
    if raw is not None and system:
        raise ValueError("a system prompt needs a chat message (user=)")
    if user is not None:
        user = user.strip()  # chat templates trim the message, so its span is found on the trimmed text
    if not (user if user is not None else raw.strip()):
        raise InputError("empty")
    tok = loaded.tokenizer
    if user is not None:
        text = generate.chat_text(tok, user, system=system)
        ids = generate.encode(tok, text, loaded.device)
        user_span = generate.user_span(tok, text, user, system=system)
    else:
        ids = tok(raw, return_tensors="pt").input_ids.to(loaded.device)  # the model's own special tokens (BOS)
        user_span = (0, ids.shape[1])
    n_prompt = ids.shape[1]
    if n_prompt > MAX_PROMPT_TOKENS:
        raise InputError("too_long", n_tokens=n_prompt, limit=MAX_PROMPT_TOKENS)

    _step(on_step, "generating")
    start = time.time()
    full = generate.generate(loaded.model._hf_model, ids, max_new_tokens=max_new_tokens)
    runtime.synchronize(loaded.device)
    generate_seconds = time.time() - start
    reply = tok.decode(full[0, n_prompt:], skip_special_tokens=True).strip()

    _step(on_step, "reading")
    start = time.time()
    readout = engine.read(
        loaded.model, loaded.lens, full, mask=loaded.mask, watch=engine.make_watch(tok, watch_words), logit_lens=True
    )
    runtime.synchronize(loaded.device)
    read_seconds = time.time() - start

    payload = build_payload(
        readout,
        decode=lambda t: tok.decode([t]),
        decode_many=lambda token_ids: tok.decode(token_ids),
        special_ids=set(tok.all_special_ids),
        n_prompt=n_prompt,
        user_span=user_span,
        user_text=user if user is not None else raw,
        reply=reply,
        band=loaded.band,
    )
    payload.update(
        mode="chat" if user is not None else "raw",
        model=loaded.info,
        timing={
            "generate_seconds": round(generate_seconds, 1),
            "read_seconds": round(read_seconds, 1),
            "n_tokens": int(full.shape[1]),
        },
    )
    if system:
        payload["system"] = system
    return payload


def analyze_message(
    loaded: Loaded,
    message: str,
    *,
    watch_words=(),
    condition: str | None = None,
    reply_in_japanese: bool = False,
    max_new_tokens: int = MAX_NEW_TOKENS,
    on_step: StepCallback = None,
) -> dict:
    """A chat message as the "try your own text" screen sends it.

    A Japanese message is analysed in English translation (the 4B model reads English more clearly), and with
    ``reply_in_japanese`` the reply is translated back. Both translations use the model already loaded, and
    ``payload["translation"]`` records them so the screen can say so. ``condition`` adds the experiment's social
    pressure (``pressure.CONDITIONS``) to the message as analysed.
    """
    message = message.strip()
    if not message:
        raise InputError("empty")
    source, translate_seconds = None, 0.0
    if translate.needs_english(message):
        n_tokens = len(loaded.tokenizer(message, add_special_tokens=False).input_ids)
        if n_tokens > MAX_PROMPT_TOKENS:
            raise InputError("too_long", n_tokens=n_tokens, limit=MAX_PROMPT_TOKENS)
        _step(on_step, "translating")
        start = time.time()
        english, complete = translate.translate(loaded, message, "en")
        translate_seconds += time.time() - start
        if not english or not complete:  # never analyse a translation that was cut short
            raise InputError("translation_failed")
        source, message = message, english
    user, system = pressure.apply(message, condition)
    payload = analyze(
        loaded, user=user, watch_words=watch_words, system=system, max_new_tokens=max_new_tokens, on_step=on_step
    )
    if condition is not None:
        payload["pressure"] = {"condition": condition, "message": message}
    reply_ja = None
    if reply_in_japanese and payload["reply"]:
        _step(on_step, "translating_reply")
        start = time.time()
        reply_ja, _ = translate.translate(loaded, payload["reply"], "ja")  # the reply itself stops at 80 tokens
        translate_seconds += time.time() - start
    if source is not None or reply_ja is not None:
        payload["translation"] = {"method": "same-model", "message_source": source, "reply_ja": reply_ja}
        payload["timing"]["translate_seconds"] = round(translate_seconds, 1)
    return payload


def _step(on_step: StepCallback, name: str) -> None:
    if on_step is not None:
        on_step(name)


REPLACEMENT_CHAR = "�"


def display_texts(token_ids: list[int], decode, decode_many, special_ids=frozenset(), max_run: int = 8) -> list[str]:
    """Per-token text for the screens.

    A character split across byte-level tokens (an emoji, a rare CJK character) decodes to U+FFFD piece by piece.
    Such a run is decoded together, shown on its first token, and the rest of the run is left empty.
    """
    texts = [decode(t) for t in token_ids]
    i = 0
    while i < len(texts):
        if REPLACEMENT_CHAR not in texts[i] or token_ids[i] in special_ids:
            i += 1
            continue
        end = i + 1
        joint = decode_many(token_ids[i:end])
        while (
            REPLACEMENT_CHAR in joint and end < len(texts) and end - i < max_run and token_ids[end] not in special_ids
        ):
            end += 1
            joint = decode_many(token_ids[i:end])
        if REPLACEMENT_CHAR not in joint:
            texts[i:end] = [joint] + [""] * (end - i - 1)
        i = end
    return texts


def build_payload(
    readout: engine.Readout,
    *,
    decode,
    special_ids: set[int],
    n_prompt: int,
    user_span: tuple[int, int],
    user_text: str,
    reply: str,
    band: tuple[int, int],
    decode_many=None,
) -> dict:
    """Shape a Readout as plain JSON. Ranks become 1-based here, and only here."""
    n_pos = len(readout.token_ids)
    decode_many = decode_many or (lambda token_ids: "".join(decode(t) for t in token_ids))
    shown = display_texts(readout.token_ids, decode, decode_many, special_ids)
    first, last = band
    band_layers = range(first, last + 1)
    before_reply = range(0, n_prompt)

    tokens = []
    for i, tid in enumerate(readout.token_ids):
        if tid in special_ids:
            kind = "special"
        elif i >= n_prompt:
            kind = "reply"
        elif user_span[0] <= i < user_span[1]:
            kind = "user"
        else:
            kind = "template"
        tokens.append({"t": shown[i], "k": kind})

    best = engine.best_watch(readout, before_reply, band_layers)
    by_position = engine.watch_by_position(readout, before_reply, band_layers, below=WATCH_TOP)
    has_logit = readout.watch_ranks_logit is not None
    best_logit = engine.best_watch(readout, before_reply, band_layers, logit=True) if has_logit else {}
    alerts = []
    for word, b in best.items():
        alerts.append(
            {
                "word": word,
                "rank": b["rank"] + 1,
                "layer": b["layer"],
                "position": b["position"],
                "token": shown[b["position"]] or decode(readout.token_ids[b["position"]]),
                "in_input": vocab.mentions(word, user_text),
                "said_in_reply": vocab.mentions(word, reply),
                "logit_rank": best_logit[word]["rank"] + 1 if word in best_logit else None,
            }
        )

    echo_text = f"{user_text} {reply}"
    concepts = [
        [
            {"t": c["text"], "r": c["best_rank"] + 1, "l": c["best_layer"], "echo": c["echo"]}
            for c in engine.salient_concepts(readout, pos, band_layers, decode, exclude_text=echo_text, n=6)
        ]
        for pos in range(n_pos)
    ]

    layers = {"j": _layer_tops(readout.top_ids, readout.top_ranks)}
    if readout.logit_top_ids is not None:
        layers["logit"] = _layer_tops(readout.logit_top_ids, readout.logit_top_ranks)

    used = {int(t) for t in readout.top_ids[:, :, :LAYER_TOP_K].ravel()}
    if readout.logit_top_ids is not None:
        used |= {int(t) for t in readout.logit_top_ids[:, :, :LAYER_TOP_K].ravel()}
    used |= set(readout.token_ids)
    return {
        "tokens": tokens,
        "n_prompt": n_prompt,
        "user_text": user_text,
        "reply": reply,
        "band": [first, last],
        "n_layers": len(readout.layers),
        "vocab_size": readout.vocab_size,
        "watch": {
            "words": readout.watch.words,
            "skipped": readout.watch.skipped,
            # [position, rank, layer] before the reply, wherever the word is within the top WATCH_TOP
            "positions": {w: [[p, r + 1, la] for p, r, la in found] for w, found in by_position.items()},
        },
        "alerts": alerts,
        "concepts": concepts,
        "layers": layers,
        "vocab": {str(t): decode(t) for t in sorted(used)},
    }


def _layer_tops(top_ids, top_ranks) -> dict:
    """``[position][layer][k]`` ids and 1-based ranks of the word-like top readouts."""
    return {"ids": top_ids[:, :, :LAYER_TOP_K].tolist(), "ranks": (top_ranks[:, :, :LAYER_TOP_K] + 1).tolist()}
