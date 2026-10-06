from jspace_demo import engine

import numpy as np
import pytest
import torch
from conftest import FakeTokenizer, make_lens
from jlens.vis import _ranks_of

PROMPT = "the quick brown fox jumps over the lazy dog"


def _read(model, lens, **kw):
    ids = model.encode(PROMPT)
    mask = torch.ones(32, dtype=torch.bool)
    return engine.read(model, lens, ids, mask=mask, top_k=5, **kw), ids


def test_band_layers():
    assert engine.band_layers(32) == (8, 25)
    assert engine.band_layers(36) == (9, 28)


def test_read_shapes_and_final_row_is_model_output(tiny):
    lens = make_lens(tiny)
    watch = engine.Watch(words=["a", "b"], ids=[3, 7], groups={"a": [0], "b": [1]})
    readout, ids = _read(tiny, lens, watch=watch)
    n_pos = ids.shape[1]
    assert readout.top_ids.shape == (n_pos, tiny.n_layers, 5)
    assert readout.watch_ranks.shape == (n_pos, tiny.n_layers, 2)
    assert readout.layers[-1] == tiny.n_layers - 1
    final = tiny.forward(ids).last_hidden_state[0]
    expected_top1 = tiny.unembed(final).argmax(-1).numpy()
    np.testing.assert_array_equal(readout.top_ids[:, -1, 0], expected_top1)
    np.testing.assert_array_equal(readout.top_ranks[:, :, 0], 0)  # unmasked: the top entry has rank 0


def test_identity_lens_equals_logit_lens(tiny):
    watch = engine.Watch(words=["a"], ids=[5], groups={"a": [0]})
    readout, _ = _read(tiny, make_lens(tiny, identity=True), watch=watch)
    np.testing.assert_array_equal(readout.top_ids, readout.logit_top_ids)
    np.testing.assert_array_equal(readout.watch_ranks, readout.watch_ranks_logit)


def test_jacobian_is_applied(tiny):
    readout, _ = _read(tiny, make_lens(tiny))
    early = slice(0, tiny.n_layers - 1)
    assert not np.array_equal(readout.top_ids[:, early], readout.logit_top_ids[:, early])
    # The final row has no J, so both lenses read the model's own output there.
    np.testing.assert_array_equal(readout.top_ids[:, -1], readout.logit_top_ids[:, -1])


def test_layer_readout_matches_masked_topk_and_falls_back():
    gen = torch.Generator().manual_seed(3)
    logits = torch.randn(5, 40, generator=gen, dtype=torch.float64)
    mask = torch.zeros(40, dtype=torch.bool)
    mask[::3] = True  # every third token is "word-like"
    watch = torch.tensor([2, 9])
    expected_ids = logits.masked_fill(~mask, float("-inf")).topk(4, dim=-1).indices
    for width in (40, 3):  # 3: too narrow to hold 4 word-like tokens, so the masked top-k fallback runs
        ids, ranks, watch_ranks = engine._layer_readout(logits, mask, 4, watch, search_width=width)
        np.testing.assert_array_equal(ids, expected_ids.numpy())
        np.testing.assert_array_equal(ranks, _ranks_of(logits, expected_ids).numpy())
        np.testing.assert_array_equal(watch_ranks, _ranks_of(logits, watch).numpy())


def test_chunked_readout_is_identical(tiny):
    lens = make_lens(tiny)
    acts = engine.record(tiny, tiny.encode(PROMPT))
    watch = engine.Watch(words=["a"], ids=[5], groups={"a": [0]})
    kwargs = dict(mask=torch.ones(32, dtype=torch.bool), watch=watch, top_k=5, use_jacobian=True)
    whole = engine._read_layers(tiny, lens, acts, **kwargs)
    for chunk in (1, 3, 7):  # uneven chunks, including a last chunk shorter than the rest
        pieces = engine._read_layers(tiny, lens, acts, chunk=chunk, **kwargs)
        for a, b in zip(whole[:3], pieces[:3], strict=True):
            np.testing.assert_array_equal(a, b)


def _cell_top(model, lens, acts, position: int, layer: int, k: int) -> list[int]:
    """The top-k tokens at one (position, layer), computed directly from the kept residual."""
    residual = acts[layer][position : position + 1].float()
    if layer in lens.jacobians:
        residual = lens.transport(residual, layer)
    return model.unembed(residual).float().topk(k, dim=-1).indices[0].tolist()


def test_one_cell_computed_directly_matches_read(tiny):
    lens = make_lens(tiny)
    ids = tiny.encode(PROMPT)
    mask = torch.ones(32, dtype=torch.bool)
    readout, acts = engine.read(tiny, lens, ids, mask=mask, top_k=5, keep_activations=True)
    for layer in (0, 2, tiny.n_layers - 1):
        assert _cell_top(tiny, lens, acts, 4, layer, k=5) == readout.top_ids[4, layer].tolist()


def test_make_watch_groups_variants_and_skips_multi_token():
    tok = FakeTokenizer({"danger": 10, " danger": 11, " scam": 12})
    watch = engine.make_watch(tok, ["danger", "scam", "blackmail", "danger"])
    assert watch.ids == [10, 11, 12]
    assert watch.groups == {"danger": [0, 1], "scam": [2]}
    assert watch.skipped == ["blackmail"]


def _manual_readout(watch_ranks, top_ids=None, top_ranks=None):
    n_pos, n_layers, _ = watch_ranks.shape
    watch = engine.Watch(words=["danger", "scam"], ids=[10, 11, 12], groups={"danger": [0, 1], "scam": [2]})
    top_ids = top_ids if top_ids is not None else np.zeros((n_pos, n_layers, 3), dtype=np.int32)
    top_ranks = top_ranks if top_ranks is not None else np.zeros_like(top_ids)
    return engine.Readout(
        token_ids=list(range(n_pos)),
        layers=list(range(n_layers)),
        top_ids=top_ids,
        top_ranks=top_ranks,
        watch=watch,
        watch_ranks=watch_ranks,
    )


def test_best_watch_takes_min_over_variants_and_window():
    ranks = np.full((4, 5, 3), 500, dtype=np.int32)
    ranks[2, 3, 1] = 4  # " danger" at position 2, layer 3
    ranks[1, 1, 0] = 9  # "danger" at position 1, layer 1
    ranks[3, 4, 2] = 0  # "scam" at position 3, layer 4 (outside the window below)
    readout = _manual_readout(ranks)
    best = engine.best_watch(readout, positions=range(0, 3), layers=range(0, 4))
    assert list(best) == ["danger", "scam"]
    assert best["danger"] == {"rank": 4, "position": 2, "layer": 3, "token_id": 11}
    assert best["scam"]["rank"] == 500
    assert engine.best_watch(readout, positions=[], layers=range(5)) == {}


def test_salient_concepts_scores_and_flags_echo():
    top_ids = np.zeros((1, 3, 3), dtype=np.int32)
    top_ranks = np.zeros_like(top_ids)
    top_ids[0] = [[100, 101, 102], [100, 103, 101], [101, 100, 104]]
    top_ranks[0] = [[0, 1, 5], [2, 3, 4], [0, 9, 9]]
    readout = _manual_readout(np.zeros((1, 3, 3), dtype=np.int32), top_ids, top_ranks)
    names = {100: " overdose", 101: " Tylenol", 102: " is", 103: " liver", 104: " ok"}
    out = engine.salient_concepts(readout, 0, range(3), names.__getitem__, exclude_text="I took 8000mg of Tylenol")
    assert [c["token_id"] for c in out][:2] == [101, 100]  # 1/2+1/5+1 = 1.7 beats 1+1/3+1/10
    assert out[0]["score"] == pytest.approx(1.7, abs=1e-3)
    flags = {c["token_id"]: c["echo"] for c in out}
    assert flags[101] is True  # Tylenol is in the input
    assert flags[100] is False  # overdose is not
    assert flags[104] is True  # two ASCII characters: too short to count as a concept
    assert {c["token_id"]: c["best_layer"] for c in out}[100] == 0


def test_two_cjk_characters_count_as_a_concept():
    top_ids = np.array([[[200, 201]]], dtype=np.int32)
    readout = _manual_readout(np.zeros((1, 1, 3), dtype=np.int32), top_ids, np.zeros_like(top_ids))
    names = {200: "粉尘", 201: " ok"}
    flags = {c["text"]: c["echo"] for c in engine.salient_concepts(readout, 0, range(1), names.__getitem__)}
    assert flags == {"粉尘": False, " ok": True}
