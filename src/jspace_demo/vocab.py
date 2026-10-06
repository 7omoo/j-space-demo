"""Vocabulary helpers: the word-like token mask (cached on disk) and whether a text mentions a word."""

from __future__ import annotations

from jspace_demo.paths import CACHE_DIR

import re

import numpy as np
import torch
from jlens import vis as _vis

_CJK = re.compile(r"[぀-ヿ一-鿿]")


def wordlike_mask(tokenizer, vocab_size: int, device: str | torch.device = "cpu") -> torch.Tensor:
    """``True`` for tokens that decode to word characters: the same rule as upstream ``compute_slice``.

    Building it decodes every vocabulary entry one by one, so the result is cached under ``cache/`` and also
    seeded into upstream's in-process cache, which ``compute_slice`` reads.
    """
    name = str(getattr(tokenizer, "name_or_path", "tokenizer")).replace("/", "--")
    path = CACHE_DIR / f"wordlike_{name}_{vocab_size}.npy"
    if path.exists():
        mask = torch.from_numpy(np.load(path))
    else:
        mask = _vis._meaningful_token_mask(tokenizer, vocab_size, torch.device("cpu")).cpu()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(path, mask.numpy())
    _vis._MEANINGFUL_MASK_CACHE[(id(tokenizer), vocab_size)] = mask
    return mask.to(device)


def mentions(word: str, text: str) -> bool:
    """Whether ``text`` contains ``word``: whole-word and case-insensitive, plain substring for CJK words."""
    if _CJK.search(word):
        return word in text
    return re.search(rf"(?<![A-Za-z]){re.escape(word)}(?![A-Za-z])", text, re.IGNORECASE) is not None
