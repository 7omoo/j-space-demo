"""The shape of an analysis payload (analysis.build_payload plus analyze's extras), shared by the heavy tests."""

PAYLOAD_KEYS = (
    "tokens",
    "n_prompt",
    "user_text",
    "reply",
    "band",
    "n_layers",
    "vocab_size",
    "watch",
    "alerts",
    "concepts",
    "layers",
    "vocab",
    "mode",
    "model",
    "timing",
)


def assert_payload_shape(p: dict, *, n_layers: int | None = None) -> None:
    missing = [k for k in PAYLOAD_KEYS if k not in p]
    assert not missing, f"missing keys: {missing}"
    n_pos, layers = len(p["tokens"]), p["n_layers"]
    if n_layers is not None:
        assert layers == n_layers
    assert p["timing"]["n_tokens"] == n_pos
    assert 0 < p["n_prompt"] <= n_pos
    first, last = p["band"]
    assert 0 <= first <= last < layers
    assert len(p["concepts"]) == n_pos
    for word, found in p["watch"]["positions"].items():
        assert word in p["watch"]["words"]
        assert all(
            0 <= pos < p["n_prompt"] and 1 <= rank <= 100 and first <= layer <= last for pos, rank, layer in found
        )
    assert {t["k"] for t in p["tokens"]} <= {"user", "template", "special", "reply"}
    for tops in p["layers"].values():
        assert len(tops["ids"]) == n_pos and all(len(row) == layers for row in tops["ids"])
        assert all(r >= 1 for pos in tops["ranks"] for row in pos for r in row), "ranks must be 1-based"
        assert all(str(t) in p["vocab"] for pos in tops["ids"] for row in pos for t in row)
    for a in p["alerts"]:
        assert a["rank"] >= 1 and first <= a["layer"] <= last and 0 <= a["position"] < p["n_prompt"]
