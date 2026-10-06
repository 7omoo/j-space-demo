"""Where the demo keeps things on disk, relative to the repository checkout."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DATA = Path(__file__).with_name("data")  # case texts, reply translations, the glossary
WEB_DIR = ROOT / "web"  # the screens, served as they are
SITE_DATA = WEB_DIR / "data"  # precomputed cases per model (committed, read by both the local app and Pages)
CACHE_DIR = ROOT / "cache"  # full analyses and token masks (large, not committed)


def analysis_cache(model_key: str) -> Path:
    """Full payloads of the prepared cases for one model: the input of the export to SITE_DATA."""
    return CACHE_DIR / "analyses" / model_key


def history_dir() -> Path:
    """The user's own analyses. ``JSPACE_HISTORY_DIR`` overrides it; nothing here is committed or sent anywhere."""
    return Path(os.environ.get("JSPACE_HISTORY_DIR", ROOT / "data" / "history"))
