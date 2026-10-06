"""The concern: how strongly the watched words came up before the reply, and where in the message the first did."""

from __future__ import annotations

from jspace_demo.glossary import Glossary
from jspace_demo.views import message
from jspace_demo.views.words import WEAK, alert, concept, concepts_at, plain_key, strong_positions

from collections.abc import Sequence

WATCHED_SHOWN = 4
CONCEPTS_AT_TRIGGER = 4  # at least; every watched word that came up there is shown


def concern_alerts(payload: dict, words: Sequence[str] | None = None) -> list[dict]:
    """The payload's alerts for ``words`` (default: every watch word), best first.

    Words written in the message itself are left out: the model was told them. Ties keep the order of ``words``,
    which is the business's own priority.
    """
    order = {w: i for i, w in enumerate(payload["watch"]["words"] if words is None else words)}
    found = [a for a in payload["alerts"] if a["word"] in order and not a["in_input"]]
    return sorted(found, key=lambda a: (a["rank"], order[a["word"]]))


def watched(alerts: list[dict]) -> list[dict]:
    """The watched words a screen lists, best first: those within the top 100, at most four."""
    return [alert(a) for a in alerts if a["rank"] <= WEAK][:WATCHED_SHOWN]


def trigger(payload: dict, alerts: list[dict], glossary: Glossary, concern: frozenset[str]) -> dict | None:
    """Where the concern came up: the first position at which a concern word reached the top 10, with the words that
    did there; without one, where the best concern word peaked."""
    if not alerts:
        return None
    strong = strong_positions(payload, [a["word"] for a in alerts])
    if strong:
        pos, hits = next(iter(strong.items()))
    else:
        best = alerts[0]
        pos, hits = best["position"], [(best["rank"], best["word"], best["layer"])]
    words = [word for _, word, _ in hits]
    # the watched words that came up here (one beyond the top 100 did not), then what else the model held there
    held = [
        concept({"t": word, "r": rank, "l": layer}, glossary, concern) for rank, word, layer in hits if rank <= WEAK
    ]
    named = {plain_key(glossary, c["text"]) for c in held}
    for c in concepts_at(payload, pos, glossary):
        if plain_key(glossary, c["t"]) not in named:
            held.append(concept(c, glossary, concern))
    in_message = message.positions(payload)
    if pos in in_message:
        where = "in_message"
    else:
        where = "before_message" if in_message and pos < in_message[0] else "after_message"
    # every position of the message lies in one phrase: the one being read when the concern came up
    phrase = next((p for p in message.phrases(payload) if p["positions"][0] <= pos <= p["positions"][1]), None)
    return {
        "position": pos,
        "token": payload["tokens"][pos]["t"],
        "where": where,
        "phrase": phrase and {"text": phrase["text"], "char_span": phrase["char_span"]},
        "words": words,
        "concepts": held[: max(CONCEPTS_AT_TRIGGER, len(words))],
    }
