"""The explainer's views: the top word at each layer at one position (a reasoning step by step), and where a
word stands at one position under each lens."""

from __future__ import annotations

from jspace_demo.glossary import Glossary
from jspace_demo.views.words import is_stop, is_unshown


def chain(payload: dict, glossary: Glossary, position: int | None = None) -> list[dict]:
    """The top word at each layer from the workspace band up, at ``position`` (default: the last prompt token).

    Runs of the same word become one step. A single layer that falls back to an earlier word between two runs of
    the same word is absorbed (currency, euro, *currency*, euro -> currency, euro); a word seen for the first time
    is kept even for one layer. An unshown token gives way to the next word at its layer; function words are
    dropped.
    """
    position = payload["n_prompt"] - 1 if position is None else position
    ids = payload["layers"]["j"]["ids"][position]
    steps: list[dict] = []
    for layer in range(payload["band"][0], len(ids)):
        shown = [text for text in (payload["vocab"][str(t)] for t in ids[layer]) if not is_unshown(text)]
        if not shown:
            continue
        text = shown[0]
        key = glossary.key(text)
        if steps and steps[-1]["key"] == key:
            steps[-1]["to"] = layer
        else:
            steps.append({"key": key, "text": text.strip(), "from": layer, "to": layer})
    i = 1
    while i < len(steps) - 1:
        relapse = steps[i]["key"] in {s["key"] for s in steps[: i - 1]}
        if relapse and steps[i]["from"] == steps[i]["to"] and steps[i - 1]["key"] == steps[i + 1]["key"]:
            steps[i - 1]["to"] = steps[i + 1]["to"]
            del steps[i : i + 2]
        else:
            i += 1
    return [
        {
            "text": s["text"],
            "en": glossary.english(s["text"]),
            "ja": glossary.ja(s["text"]),
            "from": s["from"],
            "to": s["to"],
        }
        for s in steps
        if not is_stop(glossary, s["text"])
    ]


def focus_position(payload: dict, token: str) -> int | None:
    """The last prompt position whose token is ``token`` (as written, leading space included)."""
    found = [i for i, t in enumerate(payload["tokens"][: payload["n_prompt"]]) if t["t"] == token]
    return found[-1] if found else None


def presence(payload: dict, position: int, word: str) -> dict:
    """Where ``word`` stands at ``position`` in the workspace band, per lens ("j", "logit").

    For each lens: the first and last band layer at which the word is among the readout's best eight word-like
    tokens, how many layers that is, and its best rank there (the readout's own full-vocabulary rank). ``None``
    for a lens that never has it there.
    """
    first, last = payload["band"]
    out = {}
    for lens, table in payload["layers"].items():
        found = [
            (layer, rank)
            for layer in range(first, last + 1)
            for t, rank in zip(table["ids"][position][layer], table["ranks"][position][layer], strict=True)
            if payload["vocab"][str(t)].strip().casefold() == word.casefold()
        ]
        layers = sorted({layer for layer, _ in found})
        out[lens] = (
            {"from": layers[0], "to": layers[-1], "layers": len(layers), "rank": min(r for _, r in found)}
            if found
            else None
        )
    return out
