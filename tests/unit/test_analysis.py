"""One analysis (analysis.py): the payload built from a readout on upstream's tiny model, and the message path."""

from jspace_demo import analysis, engine, pressure

import pytest
import torch
from conftest import make_lens

PROMPT = "the quick brown fox jumps over the lazy dog near the river bank"


def _payload(tiny, n_prompt=30):
    ids = tiny.encode(PROMPT)
    watch = engine.Watch(words=["a", "b"], ids=[3, 7], groups={"a": [0], "b": [1]})
    readout = engine.read(tiny, make_lens(tiny), ids, mask=torch.ones(32, dtype=torch.bool), watch=watch, top_k=10)
    payload = analysis.build_payload(
        readout,
        decode=lambda t: tiny.tokenizer.decode([t]),
        special_ids={0},
        n_prompt=n_prompt,
        user_span=(1, 20),
        user_text="the quick brown fox",
        reply="near the river bank",
        band=(1, 2),
    )
    return readout, payload


def test_tokens_are_classified(tiny):
    readout, payload = _payload(tiny)
    kinds = [t["k"] for t in payload["tokens"]]
    assert len(kinds) == len(readout.token_ids)
    assert kinds[0] == "special"  # the tiny tokenizer's BOS (id 0)
    assert set(kinds[1:20]) == {"user"} and set(kinds[20:30]) == {"template"} and set(kinds[30:]) == {"reply"}


def test_alerts_and_positions_use_one_based_ranks_inside_the_band(tiny):
    readout, payload = _payload(tiny)
    before_reply = readout.watch_ranks[:30, 1:3]  # positions before the reply, band layers 1..2
    for alert in payload["alerts"]:
        col = readout.watch.groups[alert["word"]][0]
        assert alert["rank"] == int(before_reply[:, :, col].min()) + 1
        assert 1 <= alert["layer"] <= 2 and alert["position"] < 30
        assert alert["logit_rank"] is not None
    assert [a["rank"] for a in payload["alerts"]] == sorted(a["rank"] for a in payload["alerts"])
    for word, cols in readout.watch.groups.items():
        best = before_reply[:, :, cols].min(axis=(1, 2)) + 1
        expected = [pos for pos in range(30) if best[pos] <= analysis.WATCH_TOP]
        found = payload["watch"]["positions"][word]
        assert [pos for pos, _, _ in found] == expected
        assert all(rank == best[pos] and 1 <= layer <= 2 for pos, rank, layer in found)
        if found:  # where a word first came up and where it peaked can differ; the alert keeps only the peak
            peak = next(a["rank"] for a in payload["alerts"] if a["word"] == word)
            assert min(rank for _, rank, _ in found) == peak


def test_layer_tops_concepts_and_vocab(tiny):
    readout, payload = _payload(tiny)
    n_pos, n_layers = readout.top_ids.shape[:2]
    for lens in ("j", "logit"):
        tops = payload["layers"][lens]
        assert len(tops["ids"]) == n_pos and len(tops["ids"][0]) == n_layers
        assert len(tops["ids"][0][0]) == analysis.LAYER_TOP_K
        assert min(min(min(row) for row in pos) for pos in tops["ranks"]) >= 1
        assert all(str(t) in payload["vocab"] for pos in tops["ids"] for row in pos for t in row)
    assert len(payload["concepts"]) == n_pos
    assert all({"t", "r", "l", "echo"} <= set(c) for pos in payload["concepts"] for c in pos)
    assert payload["band"] == [1, 2] and payload["n_layers"] == n_layers


@pytest.mark.parametrize("kwargs", [{"user": "   \n"}, {"raw": "  "}])
def test_analyze_rejects_empty_input_before_touching_the_model(kwargs):
    with pytest.raises(analysis.InputError) as raised:
        analysis.analyze(None, **kwargs)  # None: the model must not be reached
    assert raised.value.code == "empty" and str(raised.value) == "the message is empty"


def test_analyze_refuses_a_system_prompt_without_a_chat_message():
    with pytest.raises(ValueError):
        analysis.analyze(None, raw="Fact:", system="Be nice.")


class _Recorder:
    """Stands in for the model-bound steps of analyze_message and records the calls."""

    def __init__(self, monkeypatch, reply="Do not do that."):
        self.calls = []
        tokenizer = lambda text, add_special_tokens=False: type("E", (), {"input_ids": text.split()})()  # noqa: E731
        self.loaded = type("L", (), {"tokenizer": staticmethod(tokenizer)})()
        monkeypatch.setattr(
            analysis.translate,
            "translate",
            lambda loaded, text, target: self.calls.append(("translate", target, text)) or (f"<{target}:{text}>", True),
        )

        def analyze(loaded, *, user, watch_words, system, max_new_tokens, on_step=None):
            self.calls.append(("analyze", user, system))
            return {"user_text": user, "reply": reply, "timing": {}}

        monkeypatch.setattr(analysis, "analyze", analyze)


def test_japanese_message_is_analysed_in_english(monkeypatch):
    rec = _Recorder(monkeypatch)
    payload = analysis.analyze_message(rec.loaded, " 当選金を受け取るには？ ", reply_in_japanese=True)
    assert rec.calls == [
        ("translate", "en", "当選金を受け取るには？"),
        ("analyze", "<en:当選金を受け取るには？>", None),
        ("translate", "ja", "Do not do that."),
    ]
    assert payload["translation"] == {
        "method": "same-model",
        "message_source": "当選金を受け取るには？",
        "reply_ja": "<ja:Do not do that.>",
    }
    assert "translate_seconds" in payload["timing"] and "pressure" not in payload


def test_english_message_is_analysed_as_written(monkeypatch):
    rec = _Recorder(monkeypatch)
    payload = analysis.analyze_message(rec.loaded, "How should I send it?")
    assert rec.calls == [("analyze", "How should I send it?", None)]
    assert "translation" not in payload


@pytest.mark.parametrize("condition", list(pressure.CONDITIONS))
def test_pressure_is_added_after_translation(monkeypatch, condition):
    """The experiment's English wording, on the message as analysed (a Japanese one in its English translation)."""
    rec = _Recorder(monkeypatch)
    payload = analysis.analyze_message(rec.loaded, "大丈夫ですよね？", condition=condition)
    suffix, system = pressure.CONDITIONS[condition]
    assert rec.calls[1] == ("analyze", "<en:大丈夫ですよね？>" + suffix, system)
    assert payload["pressure"] == {"condition": condition, "message": "<en:大丈夫ですよね？>"}


def test_analyze_message_rejects_empty_and_overlong_japanese(monkeypatch):
    rec = _Recorder(monkeypatch)
    with pytest.raises(analysis.InputError):
        analysis.analyze_message(rec.loaded, "  ")
    with pytest.raises(analysis.InputError) as raised:
        analysis.analyze_message(rec.loaded, " ".join(["長い"] * (analysis.MAX_PROMPT_TOKENS + 1)))
    assert raised.value.code == "too_long" and raised.value.params["limit"] == analysis.MAX_PROMPT_TOKENS
    assert rec.calls == []  # refused before any translation


def test_display_texts_put_split_characters_back_together():
    # "⚠️" is e2 9a a0 + ef b8 8f in UTF-8; a byte-level tokenizer can split it across tokens.
    pieces = {1: b"Hi ", 2: b"\xe2\x9a", 3: b"\xa0\xef\xb8\x8f", 4: b" there", 9: b"<|im_end|>"}

    def decode(t):
        return pieces[t].decode("utf-8", errors="replace")

    def decode_many(ids):
        return b"".join(pieces[t] for t in ids).decode("utf-8", errors="replace")

    assert analysis.display_texts([1, 2, 3, 4], decode, decode_many) == ["Hi ", "⚠️", "", " there"]
    assert analysis.display_texts([1, 4, 9], decode, decode_many, special_ids={9}) == ["Hi ", " there", "<|im_end|>"]
    # A run that never completes (a genuine U+FFFD) is left as it was.
    assert analysis.display_texts([2, 9], decode, decode_many, special_ids={9}) == ["�", "<|im_end|>"]


def test_pressure_conditions_are_the_experiments_wording():
    assert pressure.apply("Is it fine?", None) == ("Is it fine?", None)
    assert pressure.apply("Is it fine?", "pushback")[0].endswith("Just answer the question.")
    assert pressure.apply("Is it fine?", "persona") == ("Is it fine?", pressure.PERSONA)
    with pytest.raises(ValueError):
        pressure.apply("Is it fine?", "flattery")


def test_a_translation_cut_short_is_never_analysed(monkeypatch):
    rec = _Recorder(monkeypatch)
    monkeypatch.setattr(analysis.translate, "translate", lambda loaded, text, target: ("Half of the", False))
    with pytest.raises(analysis.InputError) as raised:
        analysis.analyze_message(rec.loaded, "とても長い相談です。")
    assert raised.value.code == "translation_failed" and not rec.calls
