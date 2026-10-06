"""Fixtures built on upstream's tiny CPU decoder (vendor/jacobian-lens/tests/tiny.py): no download, runs in seconds."""

import jspace_demo  # noqa: F401  (must precede torch)

import importlib.util
from pathlib import Path

import pytest
import torch
from jlens import JacobianLens

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("upstream_tiny", ROOT / "vendor/jacobian-lens/tests/tiny.py")
upstream_tiny = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(upstream_tiny)


@pytest.fixture
def tiny():
    model = upstream_tiny.TinyDecoder(n_layers=4, d_model=8, vocab_size=32, seed=0).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def make_lens(model, *, identity: bool = False, seed: int = 1) -> JacobianLens:
    """A lens over every layer but the last. ``identity`` makes it equal to the logit lens."""
    gen = torch.Generator().manual_seed(seed)
    eye = torch.eye(model.d_model)
    jacobians = {
        layer: eye if identity else eye + 0.5 * torch.randn(model.d_model, model.d_model, generator=gen)
        for layer in range(model.n_layers - 1)
    }
    return JacobianLens(jacobians, n_prompts=1, d_model=model.d_model)


class FakeTokenizer:
    """Word-level tokenizer: known words (with or without a leading space) are one token, others split per char."""

    def __init__(self, vocab: dict[str, int]):
        self.vocab = vocab

    def __call__(self, text, add_special_tokens=False):
        ids = [self.vocab[text]] if text in self.vocab else [1000 + ord(c) for c in text]
        return type("Enc", (), {"input_ids": ids})()
