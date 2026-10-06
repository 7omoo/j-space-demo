"""Chat prompt construction, token spans, and greedy generation."""

from __future__ import annotations

import torch


def chat_text(tokenizer, user: str, *, system: str | None = None, think: bool = False) -> str:
    """The chat-template string up to the start of the assistant turn (thinking off by default)."""
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=think)


def encode(tokenizer, text: str, device: str | torch.device) -> torch.Tensor:
    """``[1, T]`` ids. The chat template already carries its special tokens, so none are added here."""
    return tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)


def char_span_to_tokens(tokenizer, text: str, start: int, end: int) -> tuple[int, int]:
    """Token index range [first, last) covering the characters [start, end) of ``text``."""
    offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    inside = [i for i, (s, e) in enumerate(offsets) if e > s and s >= start and e <= end + 1]
    return (inside[0], inside[-1] + 1) if inside else (0, 0)


# Private-use characters (U+E000) no chat template rewrites, used to find where a template puts the
# user's message.
_SENTINEL = "\ue000jspace-user\ue000"


def user_span(tokenizer, text: str, user: str, *, system: str | None = None, think: bool = False) -> tuple[int, int]:
    """Token index range of the user's message inside ``text`` (rendered by :func:`chat_text` with the same args).

    The message is located right after the template's own header rather than by searching for it: templates
    trim surrounding whitespace, and a short message like "user" also occurs inside the header.
    """
    start = len(chat_text(tokenizer, _SENTINEL, system=system, think=think).split(_SENTINEL)[0])
    if text[start : start + len(user)] != user:
        raise ValueError("the chat template changed the message; strip surrounding whitespace before rendering")
    return char_span_to_tokens(tokenizer, text, start, start + len(user))


@torch.no_grad()
def generate(hf_model, input_ids: torch.Tensor, *, max_new_tokens: int = 80) -> torch.Tensor:
    """Greedy continuation; returns prompt + new tokens as ``[1, T + n]``."""
    return hf_model.generate(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
    )
