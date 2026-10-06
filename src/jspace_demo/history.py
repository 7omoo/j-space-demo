"""Analyses run on this machine, kept as files (``paths.history_dir()``): the history. No accounts; nothing is
sent anywhere.

Each analysis is two files: ``<id>.json`` (the input and the full analyses) and ``<id>.meta.json`` (the short
summary the history list reads, so listing does not parse every full analysis).
"""

from __future__ import annotations

from jspace_demo.paths import history_dir

import json
import os
import re
import secrets
import time
from datetime import datetime
from pathlib import Path

_ID = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{6}$")


def save(record: dict, summary: dict) -> str:
    analysis_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}"
    created = datetime.now().astimezone().isoformat(timespec="seconds")  # "2026-10-06T15:30:12+08:00"
    folder = history_dir()
    folder.mkdir(parents=True, exist_ok=True)
    _write(folder / f"{analysis_id}.json", {**record, "id": analysis_id, "created": created})
    _write(
        folder / f"{analysis_id}.meta.json",
        {**summary, "id": analysis_id, "created": created, "created_ns": time.time_ns()},
    )
    return analysis_id


def load(analysis_id: str) -> dict | None:
    path = _path(analysis_id)
    return json.loads(path.read_text(encoding="utf-8")) if path and path.exists() else None


def listing(limit: int = 200) -> list[dict]:
    """Newest first (by the save time in nanoseconds: two saves in the same second keep their order)."""
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in history_dir().glob("*.meta.json")]
    return sorted(rows, key=lambda r: r.get("created_ns", 0), reverse=True)[:limit]


def delete(analysis_id: str) -> bool:
    path = _path(analysis_id)
    if not path or not path.exists():
        return False
    path.unlink()
    path.with_name(f"{analysis_id}.meta.json").unlink(missing_ok=True)
    return True


def _path(analysis_id: str) -> Path | None:
    """The record's file, or ``None`` for anything that is not an id (no path tricks through the API)."""
    return history_dir() / f"{analysis_id}.json" if _ID.match(analysis_id) else None


def _write(path: Path, content: dict) -> None:
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(content, ensure_ascii=False), encoding="utf-8")
    os.replace(partial, path)
