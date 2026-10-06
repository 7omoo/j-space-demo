"""Japanese names for the words a readout shows, from a reviewed dictionary (``data/glossary.json``).

A readout surfaces English words, word pieces and Chinese (sometimes Japanese) tokens. A word is shown in
Japanese only when the dictionary has it; anything else is shown as it was read. An entry whose name is null is
listed on purpose and shown as read (brand names, fragments with no clear meaning). CJK tokens map to an English
key first, so "粉尘" and "dust" count as the same word.
"""

from __future__ import annotations

from jspace_demo.paths import PACKAGE_DATA

import json
from functools import lru_cache
from pathlib import Path

PATH = PACKAGE_DATA / "glossary.json"


def singular(word: str) -> str:
    """A rough singular for merging plurals ("euros" -> "euro", "countries" -> "country"). Keys only, never shown."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith(("sses", "shes", "ches", "xes", "zzes")):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


class Glossary:
    def __init__(self, entries: dict[str, dict]):
        self.entries = entries

    @classmethod
    def load(cls, path: str | Path = PATH) -> Glossary:
        return _load(str(path))

    def english(self, text: str) -> str:
        """The English form of a token: the dictionary's key for a CJK token, otherwise the token as written."""
        token = text.strip()
        entry = self.entries.get(token) or self.entries.get(token.casefold())  # " Priz" maps like "priz"
        return entry["en"] if entry and "en" in entry else token

    def key(self, text: str) -> str:
        """The lower-case, singular English form that words are merged and looked up by."""
        word = self.english(text).casefold()
        return word if word in self.entries else singular(word)

    def ja(self, text: str) -> str | None:
        entry = self.entries.get(self.key(text))
        return entry.get("ja") if entry else None


@lru_cache(maxsize=4)
def _load(path: str) -> Glossary:
    return Glossary(json.loads(Path(path).read_text(encoding="utf-8"))["entries"])
