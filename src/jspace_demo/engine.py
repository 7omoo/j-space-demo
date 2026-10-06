"""One forward pass -> every layer's J-lens and logit-lens readout for a token sequence.

Built only from upstream parts (``ActivationRecorder``, ``JacobianLens.transport``, ``LensModel.unembed``);
upstream itself is not modified. It follows ``jlens.vis.compute_slice`` but keeps what the demo needs:
the word-like top-k per (position, layer), full-vocabulary ranks of watched tokens, and the same for the
logit lens so the two can be compared. Ranks are 0-based here; the UI shows them 1-based.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from jlens.hooks import ActivationRecorder

# Fractions of model depth that bound the "workspace" band, where readouts carry abstract content rather than
# input fragments (before) or the imminent next token (after). For Qwen3.5-4B (32 layers) this gives layers 8..25,
# which holds the boot riddle's hidden step (Italy is the top word at "boot" from layer 10 to 21; docs/report.md).
BAND_START, BAND_END = 0.25, 0.80


def band_layers(n_layers: int, start: float = BAND_START, end: float = BAND_END) -> tuple[int, int]:
    """Inclusive (first, last) layer index of the workspace band."""
    last = n_layers - 1
    return round(start * last), round(end * last)


@dataclass
class Watch:
    """Watched words resolved to single-token ids. A word can map to several ids (with / without a leading space)."""

    words: list[str]
    ids: list[int]
    groups: dict[str, list[int]]  # word -> column indices into ``ids``
    skipped: list[str] = field(default_factory=list)  # words with no single-token form


def make_watch(tokenizer, words) -> Watch:
    ids: list[int] = []
    groups: dict[str, list[int]] = {}
    skipped: list[str] = []
    for word in dict.fromkeys(w.strip() for w in words if w.strip()):
        cols = []
        for form in (word, " " + word):
            tok = tokenizer(form, add_special_tokens=False).input_ids
            if len(tok) == 1 and tok[0] not in ids:
                ids.append(tok[0])
                cols.append(len(ids) - 1)
            elif len(tok) == 1:
                cols.append(ids.index(tok[0]))
        if cols:
            groups[word] = cols
        else:
            skipped.append(word)
    return Watch(words=list(groups), ids=ids, groups=groups, skipped=skipped)


@dataclass
class Readout:
    """Readouts for one token sequence. Arrays are indexed [position, layer, ...]."""

    token_ids: list[int]
    layers: list[int]  # every layer; the last is the model's own output (J = I)
    top_ids: np.ndarray  # [T, L, k] word-like top-k under the J-lens
    top_ranks: np.ndarray  # [T, L, k] their full-vocabulary ranks
    watch: Watch
    watch_ranks: np.ndarray  # [T, L, W] J-lens ranks of the watched ids
    logit_top_ids: np.ndarray | None = None  # same as above under the logit lens (J = I at every layer)
    logit_top_ranks: np.ndarray | None = None
    watch_ranks_logit: np.ndarray | None = None
    vocab_size: int = 0


def _count_greater(neg_sorted: torch.Tensor, vals: torch.Tensor) -> torch.Tensor:
    """Per row, how many logits are strictly greater than each of ``vals``, given the row-wise ascending
    sort of the negated logits (so the count is a left insertion point)."""
    return torch.searchsorted(neg_sorted, (-vals).contiguous(), right=False).to(torch.int32).cpu()


def _layer_readout(
    logits: torch.Tensor, mask: torch.Tensor, k: int, watch_ids: torch.Tensor, search_width: int = 4096
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Word-like top-k (ids, ranks) and watched-token ranks for one layer, from a single descending sort.

    The word-like top-k is read off the first ``search_width`` sorted entries; rows that hold fewer than ``k``
    word-like tokens there fall back to a masked top-k.
    """
    values, order = logits.sort(dim=-1, descending=True)
    neg_sorted = -values  # ascending
    head = order[:, :search_width]
    wordlike = mask[head]
    keep = wordlike & (wordlike.cumsum(-1) <= k)
    if bool((keep.sum(-1) == k).all()):
        cols = keep.nonzero()[:, 1].view(-1, k)
        top_ids = head.gather(1, cols)
    else:
        top_ids = logits.masked_fill(~mask, float("-inf")).topk(k, dim=-1).indices
    top_ranks = _count_greater(neg_sorted, logits.gather(1, top_ids))
    watch_ranks = (
        _count_greater(neg_sorted, logits[:, watch_ids])
        if watch_ids.numel()
        else torch.empty(logits.shape[0], 0, dtype=torch.int32)
    )
    return top_ids.cpu().numpy(), top_ranks.numpy(), watch_ranks.numpy()


def record(model, input_ids: torch.Tensor) -> list[torch.Tensor]:
    """One forward pass; the residual after every block, as ``[T, d_model]`` tensors on the model's device."""
    layers = list(range(model.n_layers))
    with torch.no_grad(), ActivationRecorder(model.layers, at=layers) as recorder:
        model.forward(input_ids)
        return [recorder.activations[layer].detach()[0] for layer in layers]


# Positions read out per pass. A pass holds about 20 bytes x vocabulary per position (logits, sorted values, sort
# order), so 256 positions stay near 1.3 GB at Qwen3.5's 248k vocabulary however long the text is.
POSITION_CHUNK = 256


@torch.no_grad()
def _read_layers(model, lens, acts, *, mask, watch: Watch, top_k: int, use_jacobian: bool, chunk: int = POSITION_CHUNK):
    n_pos, n_layers = acts[0].shape[0], len(acts)
    watch_ids = torch.tensor(watch.ids, dtype=torch.long, device=acts[0].device)
    top_ids = np.zeros((n_pos, n_layers, top_k), dtype=np.int32)
    top_ranks = np.zeros_like(top_ids)
    watch_ranks = np.zeros((n_pos, n_layers, len(watch.ids)), dtype=np.int32)
    vocab_size = 0
    for layer in range(n_layers):
        transport = use_jacobian and layer in lens.jacobians  # the final layer has no J: it is the model's output
        for start in range(0, n_pos, chunk):
            rows = slice(start, start + chunk)
            residual = acts[layer][rows].float()
            if transport:
                residual = lens.transport(residual, layer)
            logits = model.unembed(residual).float()
            vocab_size = int(logits.shape[-1])
            top_ids[rows, layer], top_ranks[rows, layer], watch_ranks[rows, layer] = _layer_readout(
                logits, mask, top_k, watch_ids
            )
            del logits
    return top_ids, top_ranks, watch_ranks, vocab_size


def read(
    model,
    lens,
    input_ids: torch.Tensor,
    *,
    mask: torch.Tensor,
    watch: Watch | None = None,
    top_k: int = 10,
    logit_lens: bool = True,
    keep_activations: bool = False,
):
    """Run ``model`` once on ``input_ids`` ([1, T]) and read out every layer.

    ``mask`` is the word-like vocabulary mask (on the model's device). The logit lens doubles the cost, so it
    can be skipped here and added later with :func:`add_logit_lens` from kept activations. Returns the
    :class:`Readout`, plus the per-layer residuals when ``keep_activations`` is set (to check a readout against
    another implementation).
    """
    watch = watch or Watch(words=[], ids=[], groups={})
    acts = record(model, input_ids)
    top_ids, top_ranks, watch_ranks, vocab_size = _read_layers(
        model, lens, acts, mask=mask, watch=watch, top_k=top_k, use_jacobian=True
    )
    readout = Readout(
        token_ids=input_ids[0].tolist(),
        layers=list(range(model.n_layers)),
        top_ids=top_ids,
        top_ranks=top_ranks,
        watch=watch,
        watch_ranks=watch_ranks,
        vocab_size=vocab_size,
    )
    if logit_lens:
        add_logit_lens(model, lens, readout, acts, mask=mask)
    return (readout, acts) if keep_activations else readout


def add_logit_lens(model, lens, readout: Readout, acts, *, mask: torch.Tensor) -> Readout:
    """Fill the logit-lens fields of ``readout`` (J = I at every layer) from kept activations."""
    top_k = readout.top_ids.shape[-1]
    ids, ranks, watch_ranks, _ = _read_layers(
        model, lens, acts, mask=mask, watch=readout.watch, top_k=top_k, use_jacobian=False
    )
    readout.logit_top_ids, readout.logit_top_ranks, readout.watch_ranks_logit = ids, ranks, watch_ranks
    return readout


# --------------------------------------------------------------------------- #
# Aggregation over a Readout (pure numpy, no model needed)
# --------------------------------------------------------------------------- #


def best_watch(
    readout: Readout, positions: range | list[int], layers: range | list[int], *, logit: bool = False
) -> dict[str, dict]:
    """For each watched word: its best (lowest) rank over the given cells, and where it occurred.

    Returns {word: {"rank", "position", "layer", "token_id"}}; words are ordered best first.
    """
    ranks = readout.watch_ranks_logit if logit else readout.watch_ranks
    positions, layers = list(positions), list(layers)
    if not positions or not layers or ranks.shape[-1] == 0:
        return {}
    sub = ranks[np.ix_(positions, layers)]  # [P, Ls, W]
    out = {}
    for word, cols in readout.watch.groups.items():
        block = sub[:, :, cols]  # [P, Ls, variants]
        p, li, v = np.unravel_index(block.argmin(), block.shape)
        out[word] = {
            "rank": int(block[p, li, v]),
            "position": positions[p],
            "layer": layers[li],
            "token_id": readout.watch.ids[cols[v]],
        }
    return dict(sorted(out.items(), key=lambda item: item[1]["rank"]))


def watch_by_position(
    readout: Readout, positions: range | list[int], layers: range | list[int], *, below: int
) -> dict[str, list[tuple[int, int, int]]]:
    """For each watched word: (position, rank, layer) at every position where its best rank over ``layers`` is
    below ``below`` (0-based ranks), in position order. Where a word first came up, not only where it peaked."""
    positions, layers = list(positions), list(layers)
    if not positions or not layers or readout.watch_ranks.shape[-1] == 0:
        return {}
    sub = readout.watch_ranks[np.ix_(positions, layers)]  # [P, Ls, W]
    out = {}
    for word, cols in readout.watch.groups.items():
        block = sub[:, :, cols].min(axis=2)  # [P, Ls]: the best of the word's token forms
        best_layer = block.argmin(axis=1)
        best = block[np.arange(len(positions)), best_layer]
        out[word] = [(positions[i], int(best[i]), layers[int(best_layer[i])]) for i in np.flatnonzero(best < below)]
    return out


def salient_concepts(
    readout: Readout,
    position: int,
    layers: range | list[int],
    decode,
    *,
    exclude_text: str = "",
    n: int = 5,
    logit: bool = False,
) -> list[dict]:
    """Concepts most consistently near the top across ``layers`` at one position.

    Each word-like top-k entry scores 1 / (rank + 1), summed over layers (the ``compute_slice`` tracking score,
    per position). Tokens that echo ``exclude_text`` (case-insensitive substring), and ASCII fragments of at most two
    characters, are flagged as ``echo`` so the UI can show what the model brought in by itself.
    """
    top_ids = readout.logit_top_ids if logit else readout.top_ids
    top_ranks = readout.logit_top_ranks if logit else readout.top_ranks
    scores: dict[int, float] = {}
    best: dict[int, tuple[int, int]] = {}
    for layer in layers:
        for tid, rank in zip(top_ids[position, layer], top_ranks[position, layer], strict=True):
            tid, rank = int(tid), int(rank)
            scores[tid] = scores.get(tid, 0.0) + 1.0 / (rank + 1)
            if tid not in best or rank < best[tid][0]:
                best[tid] = (rank, layer)
    haystack = exclude_text.lower()
    out = []
    for tid in sorted(scores, key=scores.__getitem__, reverse=True)[:n]:
        text = decode(tid)
        stripped = text.strip().lower()
        out.append(
            {
                "token_id": tid,
                "text": text,
                "score": round(scores[tid], 3),
                "best_rank": best[tid][0],
                "best_layer": best[tid][1],
                "echo": (len(stripped) <= 2 and stripped.isascii()) or (bool(stripped) and stripped in haystack),
            }
        )
    return out
