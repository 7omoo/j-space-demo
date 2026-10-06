"""What each screen shows, built from one analysis payload (``analysis.build_payload``). No model, no I/O.

The screens read only these views. Ranks, layers and positions are copied from the payload and never
recomputed, so every number on a screen can be traced back to the readout (tests/unit/test_views.py checks this
on recorded analyses).

Screens judge by *concern words*: the business's own list, which for a prepared case is its scenario's watch
words. The generic list is not used for judging, because it also lights up on safe messages (a car sale lights
up "urgent", vitamins light up "overdose").
"""

from __future__ import annotations

from jspace_demo.glossary import Glossary
from jspace_demo.views.chain import chain, focus_position, presence
from jspace_demo.views.concern import concern_alerts, trigger, watched
from jspace_demo.views.honesty import honesty, stance
from jspace_demo.views.layers import layers
from jspace_demo.views.words import STRONG, WEAK, alert, concern_keys, is_unshown, strength

from collections.abc import Sequence

__all__ = [
    "STRONG",
    "WEAK",
    "build",
    "chain",
    "concern_alerts",
    "honesty",
    "is_unshown",
    "layers",
    "stance",
    "strength",
    "trigger",
    "watched",
]


def build(
    payload: dict,
    *,
    concern_words: Sequence[str] | None = None,
    glossary: Glossary | None = None,
    focus_token: str | None = None,
) -> dict:
    """Every screen's data for one analysis. ``focus_token``: for the explainer, the prompt token whose layers tell
    the story (the riddle's " boot").
    """
    glossary = glossary or Glossary.load()
    words = list(payload["watch"]["words"] if concern_words is None else concern_words)
    concern = concern_keys(glossary, words)
    alerts = concern_alerts(payload, words)
    chat = payload.get("mode", "chat") == "chat"
    found = trigger(payload, alerts, glossary, concern)
    chains = focus = None
    if not chat and "layers" in payload:  # the explainer: the focus token (else the trigger), then the last token
        start = focus_position(payload, focus_token) if focus_token else None
        if start is None:  # no focus token, or this tokenizer splits it: where a concern word first came up
            start = (found or {}).get("position")
        positions = dict.fromkeys(p for p in (start, payload["n_prompt"] - 1) if p is not None)
        chains = [
            {"position": p, "token": payload["tokens"][p]["t"], "steps": chain(payload, glossary, p)} for p in positions
        ]
        if start is not None:
            focus = {
                "position": start,
                "token": payload["tokens"][start]["t"],
                "words": [{"word": w, **presence(payload, start, w)} for w in words],
            }
    view = {
        "mode": "chat" if chat else "raw",
        "model": (payload.get("model") or {}).get("key"),
        "vocab_size": payload.get("vocab_size"),
        "message": payload["user_text"],  # what was analysed (the English translation for a Japanese message)
        "reply": payload["reply"],
        # the reply ran into the token limit: the model had not ended it (its last token is not a special one)
        "reply_cut": chat and bool(payload["reply"]) and payload["tokens"][-1]["k"] == "reply",
        "translation": payload.get("translation"),  # {"message_source", "reply_ja", "method"} when translated
        "system": payload.get("system"),
        "concern_words": words,
        "unread_words": [w for w in words if w in payload["watch"]["skipped"]],  # no single token: never watched
        "honesty": honesty(payload, alerts) if chat else None,
        "watched": watched(alerts) if chat else None,
        "trigger": found,
        "chains": chains,
        "focus": focus,
        "alerts": [alert(a) for a in alerts],
    }
    _name_alerts(view, glossary)
    return view


def _name_alerts(view: dict, glossary: Glossary) -> None:
    """Give every concern word a screen shows its Japanese name (in place)."""
    found = [*view["alerts"], *(view["watched"] or []), (view["honesty"] or {}).get("word")]
    for item in filter(None, found):
        item["ja"] = glossary.ja(item["word"])
