"""Device and dtype selection, and loading a model together with its pre-fitted Jacobian lens."""

from __future__ import annotations

from jspace_demo import FALLBACK_READY
from jspace_demo.models import DEFAULT_MODEL, spec

import jlens
import torch
import transformers

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

# Chosen with experiments/select_dtype.py (2026-10-06): all five candidates (mps bf16/fp16/fp32, cpu bf16/fp32)
# passed the health check; (mps, bf16) was the first in order, at 8.2 GiB and 0.15 s per 13-token forward
# pass for the 4B model (CPU: 7-11 s).
DEFAULT_DTYPE = "bf16"


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model: str = DEFAULT_MODEL, dtype: str = DEFAULT_DTYPE, device: str | None = None):
    """Return ``(LensModel, tokenizer)``. ``AutoModelForCausalLM`` loads the text decoder only."""
    device = device or pick_device()
    if device == "mps" and not FALLBACK_READY:
        raise RuntimeError(
            "torch was imported before jspace_demo, so the MPS CPU fallback is off. "
            "Import jspace_demo first, or set PYTORCH_ENABLE_MPS_FALLBACK=1 in the shell."
        )
    name = spec(model).hf_name
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(name, dtype=DTYPES[dtype]).to(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(name)
    return jlens.from_hf(hf_model, tokenizer), tokenizer


def load_lens(model: str = DEFAULT_MODEL, device: str | None = None) -> jlens.JacobianLens:
    """Load the lens; with ``device``, also move its matrices there once.

    ``JacobianLens.transport`` calls ``J.to(residual.device)`` on every use, which copies the matrix each time
    while it sits on the CPU and is a no-op once it already lives on the device.
    """
    s = spec(model)
    lens = jlens.JacobianLens.from_pretrained(s.lens_repo, filename=s.lens_file, revision=s.lens_revision)
    if device is not None:
        lens.jacobians = {layer: J.to(device) for layer, J in lens.jacobians.items()}
    return lens


def synchronize(device: str) -> None:
    """Wait for queued GPU work, so a timing covers the work itself."""
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def free_cached_memory(device: str) -> None:
    """Return the allocator's cached blocks to the system, so memory held between requests does not creep up."""
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()


def held_memory_gib(device: str) -> float | None:
    """Memory the GPU allocator holds for this process, or ``None`` on the CPU."""
    if device == "mps":
        return round(torch.mps.driver_allocated_memory() / 2**30, 2)
    if device == "cuda":
        return round(torch.cuda.memory_reserved() / 2**30, 2)
    return None
