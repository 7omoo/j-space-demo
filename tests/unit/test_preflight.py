"""The pre-load check reads the machine's state and reports reasons; it must never act on what it finds."""

from jspace_demo import preflight

import subprocess


def _fake_run(outputs):
    def run(args, timeout=20.0):
        key = args[0].rsplit("/", 1)[-1]
        stdout, code = outputs.get(key, ("", 1))
        return subprocess.CompletedProcess(args, code, stdout=stdout, stderr="")

    return run


VM_STAT = (
    "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
    "Pages free: 65536.\nPages inactive: 65536.\nPages speculative: 0.\nPages purgeable: 0.\n"
)  # 2 GiB


def test_reports_lm_studio_generating_other_jobs_and_low_memory(monkeypatch, tmp_path):
    lms = tmp_path / "lms"
    lms.write_text("")
    monkeypatch.setattr(preflight, "LMS", lms)
    monkeypatch.setenv("JSPACE_OTHER_LLM_PATTERNS", "batch_inference.py")
    monkeypatch.setattr(
        preflight,
        "_run",
        _fake_run(
            {
                "lms": ("qwen3.8-27b  GENERATING  16.08 GB", 0),
                "pgrep": ("12345", 0),
                "vm_stat": (VM_STAT, 0),
            }
        ),
    )
    reasons = preflight.busy_reasons(required_free_gib=10)
    assert reasons == [
        "LM Studio is generating",
        "batch_inference.py is running",
        "2.0 GiB of memory available (10 GiB needed)",
    ]


def test_clear_when_nothing_else_runs(monkeypatch, tmp_path):
    monkeypatch.setattr(preflight, "LMS", tmp_path / "missing-lms")
    monkeypatch.setattr(preflight, "_run", _fake_run({"pgrep": ("", 1), "vm_stat": (VM_STAT, 0)}))
    assert preflight.busy_reasons(required_free_gib=1) == []
    assert round(preflight.available_gib(), 1) == 2.0
