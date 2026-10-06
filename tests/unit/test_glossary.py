from jspace_demo import glossary
from jspace_demo.glossary import Glossary

import json

import pytest

GLOSSARY = Glossary.load()


@pytest.mark.parametrize(
    ("token", "key", "ja"),
    [
        (" toxic", "toxic", "有毒"),
        ("WARNING", "warning", "警告"),
        (" Euros", "euro", "ユーロ"),
        (" countries", "country", "国"),
        (" asbestos", "asbestos", "アスベスト"),  # listed, so not cut to "asbesto"
        ("粉尘", "dust", "粉じん"),  # a Chinese token maps to its English key
        ("意大利", "italy", "イタリア"),
        (" antico", "anticoagulant", "抗凝固薬"),  # a word piece maps to the whole word
        (" Zanzibar", "zanzibar", None),  # not listed: shown as read
    ],
)
def test_lookup(token, key, ja):
    assert GLOSSARY.key(token) == key and GLOSSARY.ja(token) == ja


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("euros", "euro"),
        ("countries", "country"),
        ("boxes", "box"),
        ("glasses", "glass"),
        ("prizes", "prize"),
        ("gas", "gas"),
        ("status", "status"),
        ("analysis", "analysis"),
        ("dust", "dust"),
    ],
)
def test_singular(word, expected):
    assert glossary.singular(word) == expected


def test_file_is_well_formed():
    entries = json.loads(glossary.PATH.read_text(encoding="utf-8"))["entries"]
    for key, entry in entries.items():
        assert set(entry) in ({"ja"}, {"en"}), key
        if "ja" in entry:  # a name, or null: listed, and shown as read on purpose
            assert key == key.casefold() and (entry["ja"] is None or entry["ja"].strip()), key
        else:  # a mapping must land on a listed word, or the token would show untranslated
            assert GLOSSARY.key(entry["en"]) in entries and "ja" in entries[GLOSSARY.key(entry["en"])], key
    assert list(entries) == sorted(k for k in entries if k.isascii()) + sorted(k for k in entries if not k.isascii())
