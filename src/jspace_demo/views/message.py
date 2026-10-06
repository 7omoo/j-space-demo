"""The message as tokens, words and phrases, with character spans for highlighting it on screen."""

from __future__ import annotations

import re

MAX_PHRASE_WORDS = 8
_CLAUSE_END = re.compile(r"[,.;:!?、。！？]\s*$")
_CONJUNCTIONS = frozenset({"but", "and", "so", "because", "or", "while", "although", "though"})
_PREPOSITIONS = frozenset(
    {
        "of",
        "in",
        "on",
        "at",
        "for",
        "with",
        "from",
        "to",
        "into",
        "under",
        "over",
        "off",
        "about",
        "by",
        "after",
        "before",
        "near",
        "without",
    }
)
_CJK = re.compile(r"[぀-ヿ一-鿿]")


def positions(payload: dict) -> list[int]:
    """Token positions of the user's message (not the chat template around it)."""
    return [i for i, t in enumerate(payload["tokens"]) if t["k"] == "user"]


def spans(payload: dict) -> dict[int, tuple[int, int]] | None:
    """Character span in ``user_text`` of each message token, or ``None`` if the tokens do not spell the text."""
    text, cursor, out = payload["user_text"], 0, {}
    for pos in positions(payload):
        piece = payload["tokens"][pos]["t"]
        if not text.startswith(piece, cursor):
            return None
        out[pos] = (cursor, cursor + len(piece))
        cursor += len(piece)
    return out if cursor == len(text) else None


def phrases(payload: dict) -> list[dict]:
    """The message cut into short phrases: after punctuation, before a conjunction, long runs at a preposition."""
    clauses, current = [], []
    for word in _words(payload):
        if word["text"].strip().casefold() in _CONJUNCTIONS and len(current) >= 2:
            clauses.append(current)
            current = []
        current.append(word)
        if _CLAUSE_END.search(word["text"]):
            clauses.append(current)
            current = []
    if current:
        clauses.append(current)
    char_spans = spans(payload)
    out = []
    for group in (g for clause in clauses for g in _split_long(clause)):
        first, last = group[0]["positions"][0], group[-1]["positions"][-1]
        raw = "".join(w["text"] for w in group)
        span = None
        if char_spans:
            lead, trail = len(raw) - len(raw.lstrip()), len(raw) - len(raw.rstrip())
            span = [char_spans[first][0] + lead, char_spans[last][1] - trail]
        out.append({"positions": [first, last], "text": raw.strip(), "char_span": span})
    return out


def _words(payload: dict) -> list[dict]:
    """Message tokens grouped into words: a word starts at a token that begins with a space (each CJK token alone)."""
    words: list[dict] = []
    for pos in positions(payload):
        text = payload["tokens"][pos]["t"]
        joins = bool(words) and (
            text == "" or not (text[:1].isspace() or _CJK.search(text) or _CJK.search(words[-1]["text"]))
        )
        if joins:
            words[-1]["positions"].append(pos)
            words[-1]["text"] += text
        else:
            words.append({"positions": [pos], "text": text})
    return words


def _split_long(words: list[dict]) -> list[list[dict]]:
    if len(words) <= MAX_PHRASE_WORDS:
        return [words]
    cuts = [i for i in range(2, len(words) - 1) if words[i]["text"].strip().casefold() in _PREPOSITIONS]
    if not cuts:
        return [words]
    middle = len(words) / 2
    cut = min(cuts, key=lambda i: abs(i - middle))
    return _split_long(words[:cut]) + _split_long(words[cut:])
