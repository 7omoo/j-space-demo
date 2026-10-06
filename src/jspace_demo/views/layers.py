"""Every layer at every token, for the expert view: the top words under the J-lens and under the logit lens.

The payload keeps the eight best word-like readouts per position and layer. The expert view shows the best three
that are not unshown tokens (``words.UNSHOWN``), packed as indices into one word list per analysis, so a case file
stays small enough to ship for every model.
"""

from __future__ import annotations

from jspace_demo.views.words import is_unshown

TOP = 3


def layers(payload: dict, top: int = TOP) -> dict | None:
    """``top[lens][position][layer]`` lists indices into ``words``, best first (lens: "j" or "logit").

    ``None`` when the payload kept no per-layer tables (a trimmed test fixture).
    """
    if "layers" not in payload:
        return None
    words: list[str] = []
    index: dict[str, int] = {}

    def ref(token_id: int) -> int:
        text = payload["vocab"][str(token_id)]
        if text not in index:
            index[text] = len(words)
            words.append(text)
        return index[text]

    def best(cell: list[int]) -> list[int]:  # choose first, then pack: only shown words enter the list
        return [ref(t) for t in [t for t in cell if not is_unshown(payload["vocab"][str(t)])][:top]]

    tops = {
        lens: [[best(cell) for cell in row] for row in payload["layers"][lens]["ids"]]
        for lens in ("j", "logit")
        if lens in payload["layers"]
    }
    return {
        "n_layers": payload["n_layers"],
        "band": payload["band"],
        "n_prompt": payload["n_prompt"],
        "tokens": [{"t": t["t"], "k": t["k"]} for t in payload["tokens"]],
        "words": words,
        "top": tops,
    }
