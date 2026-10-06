"""Translate with the model already loaded: a Japanese message into English before analysis, a reply into Japanese.

The 4B model's readouts are clearer in English (the Japanese version of the boot riddle does not resolve), so a
Japanese message is analysed in English translation and the screens say so. Greedy decoding with thinking off,
so the same input always gives the same translation. No second model, no network.
"""

from __future__ import annotations

from jspace_demo import generate

import re

_JAPANESE = re.compile(r"[぀-ヿ一-鿿]")  # kana or kanji
_INSTRUCTIONS = {
    "en": "Translate the following Japanese text into natural English. Keep every number, name and unit exactly "
    "as written. Output only the translation.",
    "ja": "Translate the following English text into natural Japanese. Keep every number, name and unit exactly "
    "as written. Output only the translation.",
}


def needs_english(text: str) -> bool:
    """Whether a message should be translated before analysis: it contains kana or kanji."""
    return bool(_JAPANESE.search(text))


def translate(loaded, text: str, target: str) -> tuple[str, bool]:
    """``text`` translated into ``target`` ("en" or "ja") by the loaded chat model, and whether the model ended the
    translation itself (``False``: it ran out of room and was cut short)."""
    chat = generate.chat_text(loaded.tokenizer, f"{_INSTRUCTIONS[target]}\n\n{text.strip()}")
    ids = generate.encode(loaded.tokenizer, chat, loaded.device)
    # English takes about 1.1 times the tokens of the Japanese; twice the input leaves room either way
    budget = 2 * len(loaded.tokenizer(text, add_special_tokens=False).input_ids) + 32
    new = generate.generate(loaded.model._hf_model, ids, max_new_tokens=budget)[0, ids.shape[1] :]
    ended = len(new) < budget or int(new[-1]) in _end_ids(loaded.model._hf_model)
    return loaded.tokenizer.decode(new, skip_special_tokens=True).strip(), ended


def _end_ids(hf_model) -> set[int]:
    ids = hf_model.generation_config.eos_token_id
    return set(ids) if isinstance(ids, list) else {ids}
