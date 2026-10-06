"""Is another LLM workload running on this machine? A model shares the GPU and memory, so load one at a time.

    uv run python -m jspace_demo.preflight [required_free_gib]   # exit 0: clear to load a model

Checks LM Studio (its read-only ``lms ps``), any processes whose command line matches
``JSPACE_OTHER_LLM_PATTERNS`` (comma-separated), and the memory available (macOS ``vm_stat``). It only reads the
machine's state: it never stops or changes anything.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

LMS = Path.home() / ".lmstudio" / "bin" / "lms"


def _run(args: list[str], timeout: float = 20.0) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout)


def available_gib() -> float | None:
    """Memory the OS could hand out now (free, inactive, speculative and purgeable pages), or None if unknown."""
    try:
        out = _run(["vm_stat"]).stdout
    except OSError:
        return None
    page = re.search(r"page size of (\d+) bytes", out)
    if page is None:
        return None
    keys = ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable")
    pages = sum(int(m.group(1)) for key in keys if (m := re.search(rf"{key}:\s+(\d+)", out)))
    return pages * int(page.group(1)) / 2**30


def swap_used_mib() -> float:
    try:
        out = _run(["sysctl", "-n", "vm.swapusage"]).stdout
    except OSError:
        return 0.0
    match = re.search(r"used = ([\d.]+)M", out)
    return float(match.group(1)) if match else 0.0


def busy_reasons(required_free_gib: float = 0.0) -> list[str]:
    """Why a model should not be loaded right now; empty when it is clear."""
    reasons = []
    if LMS.exists():
        try:
            if "GENERATING" in _run([str(LMS), "ps"]).stdout:
                reasons.append("LM Studio is generating")
        except (subprocess.TimeoutExpired, OSError):
            reasons.append("LM Studio's state could not be read")
    patterns = os.environ.get("JSPACE_OTHER_LLM_PATTERNS", "")
    for pattern in filter(None, (p.strip() for p in patterns.split(","))):
        if _run(["pgrep", "-f", pattern]).returncode == 0:
            reasons.append(f"{pattern} is running")
    if required_free_gib:
        free = available_gib()
        if free is not None and free < required_free_gib:
            reasons.append(f"{free:.1f} GiB of memory available ({required_free_gib:.0f} GiB needed)")
    return reasons


def main() -> int:
    need = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
    found = busy_reasons(need)
    free = available_gib()
    print(f"available {'?' if free is None else f'{free:.1f}'} GiB, swap used {swap_used_mib():.0f} MiB")
    print("clear to load a model" if not found else "not clear: " + "; ".join(found))
    return 0 if not found else 3


if __name__ == "__main__":
    sys.exit(main())
