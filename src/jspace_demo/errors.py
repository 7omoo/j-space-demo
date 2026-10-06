"""Errors the user can fix, worded by the screens. Shared by the analysis, the job queue and the server; no torch."""

from __future__ import annotations


class InputError(ValueError):
    """A request the user can fix: an empty or over-long message. ``code`` and ``params`` let a screen word it."""

    MESSAGES = {
        "empty": "the message is empty",
        "too_long": "the message is too long ({n_tokens} tokens; the limit is {limit})",
        "translation_failed": "the message could not be translated into English",
    }

    def __init__(self, code: str, **params):
        super().__init__(self.MESSAGES[code].format(**params))
        self.code = code
        self.params = params
