"""The models the demo knows, each with the Jacobian lens Neuronpedia published for it. Plain data: no torch."""

from __future__ import annotations

from dataclasses import dataclass

LENS_REPO = "neuronpedia/jacobian-lens"
LENS_REVISION = "qwen-n1000"


@dataclass(frozen=True)
class ModelSpec:
    key: str  # short name used in file paths and URLs
    hf_name: str  # HuggingFace model id
    label: str  # name shown on screen
    lens_file: str  # file in the lens repository
    licence: str  # the model's licence, for the credits
    memory_gib: float  # GPU memory the model and its lens hold once loaded (measured by `jspace-demo check`)
    lens_repo: str = LENS_REPO
    lens_revision: str = LENS_REVISION


MODELS: dict[str, ModelSpec] = {
    spec.key: spec
    for spec in (  # smallest first: the order of the model switcher
        ModelSpec(
            "qwen3.5-4b",
            "Qwen/Qwen3.5-4B",
            "Qwen3.5-4B",
            "qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt",
            "Apache-2.0",
            9.2,
        ),
        ModelSpec(
            "qwen3-8b",
            "Qwen/Qwen3-8B",
            "Qwen3-8B",
            "qwen3-8b/jlens/Salesforce-wikitext/Qwen3-8B_jacobian_lens.pt",
            "Apache-2.0",
            18.4,
        ),
        ModelSpec(
            "gemma-3-12b-it",
            "google/gemma-3-12b-it",
            "Gemma 3 12B",
            "gemma-3-12b-it/jlens/Salesforce-wikitext/gemma-3-12b-it_jacobian_lens.pt",
            "Gemma Terms of Use",
            26.7,
        ),
        ModelSpec(
            "qwen3-14b",
            "Qwen/Qwen3-14B",
            "Qwen3-14B",
            "qwen3-14b/jlens/Salesforce-wikitext/Qwen3-14B_jacobian_lens.pt",
            "Apache-2.0",
            31.9,
        ),
    )
}
DEFAULT_MODEL = "qwen3.5-4b"


def spec(key: str) -> ModelSpec:
    """The model called ``key``, or a ValueError that lists the known ones."""
    try:
        return MODELS[key]
    except KeyError:
        raise ValueError(f"unknown model {key!r}; known: {', '.join(MODELS)}") from None
