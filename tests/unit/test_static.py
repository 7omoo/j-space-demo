"""Guards that need no model: page safety, upstream pinning, local-only serving, lens integrity, page logic."""

from jspace_demo import cli
from jspace_demo.models import DEFAULT_MODEL, spec
from jspace_demo.paths import ROOT, WEB_DIR

import glob
import hashlib
import os
import re
import shutil
import subprocess

import pytest

PINNED_UPSTREAM = "581d398613e5602a5af361e1c34d3a92ea82ba8e"
LENS_SHA256 = "1f9a8f8fd593f0ffec1a9640993257ca4560f8ae3e5602315643d5cc6818534e"


def test_page_never_parses_text_as_html():
    for path in WEB_DIR.rglob("*.js"):
        found = re.findall(r"innerHTML|outerHTML|insertAdjacentHTML|document\.write", path.read_text())
        assert not found, f"{path.name}: {found}"


def test_upstream_is_pinned_and_unedited():
    vendor = ROOT / "vendor" / "jacobian-lens"
    if not vendor.exists():
        pytest.skip("upstream not cloned (scripts/bootstrap.sh)")
    head = subprocess.run(["git", "-C", str(vendor), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    status = subprocess.run(
        ["git", "-C", str(vendor), "status", "--porcelain"], capture_output=True, text=True, check=True
    )
    assert head.stdout.strip() == PINNED_UPSTREAM
    assert status.stdout.strip() == ""


def test_server_listens_on_localhost_only(monkeypatch):
    import uvicorn

    seen = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: seen.update(kwargs))
    assert cli.main(["serve", "--no-model"]) == 0
    assert seen["host"] == "127.0.0.1" and seen["workers"] == 1
    with pytest.raises(SystemExit):  # there is no way to listen on another address
        cli.main(["serve", "--no-model", "--host", "0.0.0.0"])


def test_lens_file_matches_its_published_hash():
    from huggingface_hub import try_to_load_from_cache

    s = spec(DEFAULT_MODEL)
    path = try_to_load_from_cache(s.lens_repo, filename=s.lens_file, revision=s.lens_revision)
    if not isinstance(path, str):
        pytest.skip("lens not downloaded on this machine")
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            digest.update(block)
    assert digest.hexdigest() == LENS_SHA256


def _node() -> str | None:
    candidates = [
        shutil.which("node"),
        *sorted(glob.glob(os.path.expanduser("~/.nvm/versions/node/*/bin/node"))),
        "/opt/homebrew/bin/node",
    ]
    return next((c for c in candidates if c and os.access(c, os.X_OK)), None)


def test_page_logic_unit_tests_pass():
    node = _node()
    if node is None:
        pytest.skip("node is not installed")
    tests = sorted(str(p) for p in (ROOT / "tests" / "js").glob("*.test.mjs"))
    result = subprocess.run([node, "--test", *tests], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
