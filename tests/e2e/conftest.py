"""The static site, served under a sub-path as GitHub Pages serves it, for the browser tests."""

from jspace_demo.paths import WEB_DIR

import functools
import http.server
import threading

import pytest

SUBPATH = "j-space-demo"


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):  # the test output stays readable
        pass


@pytest.fixture(scope="session")
def site_url(tmp_path_factory):
    root = tmp_path_factory.mktemp("pages")
    (root / SUBPATH).symlink_to(WEB_DIR, target_is_directory=True)
    handler = functools.partial(_Quiet, directory=str(root))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/{SUBPATH}/"
    server.shutdown()


@pytest.fixture
def watched(page):
    """The page, with every console error, page error and request it makes recorded."""
    seen = {"errors": [], "requests": []}
    page.on("console", lambda msg: seen["errors"].append(msg.text) if msg.type == "error" else None)
    page.on("pageerror", lambda error: seen["errors"].append(str(error)))
    page.on("request", lambda request: seen["requests"].append(request.url))
    return page, seen
