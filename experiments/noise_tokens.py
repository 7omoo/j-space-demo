"""Which tokens top the readout whatever the message? Evidence for the screens' unshown tokens. No model loaded.

    uv run python experiments/noise_tokens.py [min_percent]

For every model with analysed cases (cache/analyses/<model>/), counts the positions at which each token is among
the three best J-lens readouts at some layer of the workspace band, over every position of every case. A token
that does so at a large share of all positions, across messages about paint, banks and hiking alike, says nothing
about any one message. Prints the tokens above min_percent (default 2) per model, marking those in
jspace_demo.views.words.UNSHOWN, and how many positions any unshown token reached.
"""

from jspace_demo import cases
from jspace_demo.models import MODELS
from jspace_demo.paths import analysis_cache
from jspace_demo.precompute import load_payload
from jspace_demo.views.words import STOPWORDS, is_unshown

import sys
from collections import Counter

TOP = 3


def band_tops(payload: dict) -> list[set[str]]:
    """Per position: the tokens among the TOP best J-lens readouts at any layer of the band."""
    first, last = payload["band"]
    vocab = payload["vocab"]
    return [
        {vocab[str(t)] for layer in range(first, last + 1) for t in row[layer][:TOP]}
        for row in payload["layers"]["j"]["ids"]
    ]


def main() -> None:
    min_share = float(sys.argv[1]) / 100 if len(sys.argv) > 1 else 0.02
    for model in (key for key in MODELS if analysis_cache(key).is_dir()):
        freq, positions, hit = Counter(), 0, 0
        for case in cases.all_cases().values():
            for tops in band_tops(load_payload(model, case.id)):
                freq.update(tops)
                positions += 1
                hit += any(is_unshown(t) for t in tops)
        print(f"{model}: {positions} positions; an unshown token among the top {TOP} at {hit / positions:.0%}")
        for token, n in freq.most_common():
            if n / positions < min_share:
                break
            if token.strip().casefold() in STOPWORDS:
                continue
            mark = "unshown" if is_unshown(token) else ""
            print(f"  {n / positions:6.1%}  {token!r:<32} {mark}")


if __name__ == "__main__":
    main()
