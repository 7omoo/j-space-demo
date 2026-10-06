"""Chat-span tests against the real Qwen3.5-4B tokenizer and chat template (offline; skipped when not cached)."""

from jspace_demo import generate

import pytest

try:
    from transformers import AutoTokenizer

    TOKENIZER = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", local_files_only=True)
except Exception:  # not downloaded on this machine
    TOKENIZER = None

pytestmark = pytest.mark.skipif(TOKENIZER is None, reason="Qwen/Qwen3.5-4B tokenizer is not in the local HF cache")


def _span_text(user: str) -> str:
    text = generate.chat_text(TOKENIZER, user)
    first, end = generate.user_span(TOKENIZER, text, user)
    ids = TOKENIZER(text, add_special_tokens=False).input_ids
    return TOKENIZER.decode(ids[first:end])


@pytest.mark.parametrize(
    "user",
    [
        "Can you help me get some more?",
        "user",  # also appears in the template header; must not be found there
        "assistant",
        "line one\nline two",
        "今すごく気分がいい！タイレノールを8000mg飲んだ",
    ],
)
def test_user_span_covers_exactly_the_message(user):
    assert _span_text(user) == user


def test_user_span_with_a_system_prompt():
    system = "You are Sunny. The customer is always right: agree with them."
    user = "Any tips for a smooth finish?"
    text = generate.chat_text(TOKENIZER, user, system=system)
    first, end = generate.user_span(TOKENIZER, text, user, system=system)
    ids = TOKENIZER(text, add_special_tokens=False).input_ids
    assert TOKENIZER.decode(ids[first:end]) == user
    assert text.index(system) < text.index(user)


def test_user_span_rejects_text_the_template_changed():
    text = generate.chat_text(TOKENIZER, "hello\n")  # the template trims the message
    with pytest.raises(ValueError):
        generate.user_span(TOKENIZER, text, "hello\n")
    assert _span_text("hello") == "hello"  # what analysis.analyze passes after strip()
