"""Which device and dtype to run Qwen3.5-4B in: one candidate, checked against the CPU fp32 reference.

Run once per candidate, each in its own process so memory is released between candidates:

    uv run python experiments/select_dtype.py mps bf16

Prints one JSON line and exits 0 when the candidate passes the health check (jspace_demo.health). On
2026-10-06 all five candidates passed (mps bf16/fp16/fp32, cpu bf16/fp32); mps bf16 was the first in order.
"""

from jspace_demo import health, runtime  # first: must be imported before anything that imports torch
from jspace_demo.models import DEFAULT_MODEL

import json
import logging
import resource
import statistics
import sys
import time
import warnings

import torch

device, dtype = sys.argv[1], sys.argv[2]


class _Collect(logging.Handler):
    """Collects transformers log warnings (the reference-kernel fallback notices go through logging)."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


log_collector = _Collect()
logging.getLogger("transformers").addHandler(log_collector)

result: dict = {"device": device, "dtype": dtype}
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    try:
        start = time.time()
        model, tokenizer = runtime.load_model(DEFAULT_MODEL, dtype, device)
        runtime.synchronize(device)
        result["load_seconds"] = round(time.time() - start, 1)

        start = time.time()
        lens = runtime.load_lens(DEFAULT_MODEL)
        result["lens_load_seconds"] = round(time.time() - start, 1)
        result["lens_repr"] = repr(lens)

        hf_model = model._hf_model
        result["hf_class"] = type(hf_model).__name__
        result["n_params"] = sum(p.numel() for p in hf_model.parameters())
        result["layout_path"] = model.layout.path
        result["n_layers"] = model.n_layers
        result["d_model"] = model.d_model
        result["shape_checks"] = {
            "n_layers==32": model.n_layers == 32,
            "d_model==lens.d_model==2560": model.d_model == lens.d_model == 2560,
            "max(source_layers)<n_layers": max(lens.source_layers) < model.n_layers,
        }

        start = time.time()
        result.update(health.check(DEFAULT_MODEL, model, lens, tokenizer))
        result["health_check_seconds"] = round(time.time() - start, 1)

        # Bare forward pass on the 13-token boot prompt (no lens), median of 3 after one warm-up.
        ids = model.encode(health.BOOT_PROMPT)
        with torch.no_grad():
            model.forward(ids)
            runtime.synchronize(device)
            times = []
            for _ in range(3):
                start = time.time()
                model.forward(ids)
                runtime.synchronize(device)
                times.append(time.time() - start)
        result["forward_seconds_median"] = round(statistics.median(times), 3)
        if device == "mps":
            result["mps_driver_allocated_gib"] = round(torch.mps.driver_allocated_memory() / 2**30, 2)
    except Exception:  # a crashing candidate is a failed candidate; keep the tail of the error
        import traceback

        result["passed"] = False
        result["error"] = traceback.format_exc().strip().splitlines()[-6:]

result["max_rss_gib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30, 2)  # bytes on macOS
fallback = sorted({str(w.message) for w in caught if "fall back" in str(w.message).lower()})
result["mps_cpu_fallback_ops"] = fallback
result["other_warnings"] = sorted({str(w.message)[:200] for w in caught} - set(fallback))[:10]
result["transformers_warnings"] = sorted({m[:200] for m in log_collector.messages})[:10]
print(json.dumps(result, ensure_ascii=False))
sys.exit(0 if result.get("passed") else 1)
