"""The local app's own screen (try your own text: the form, the analysis while it runs, the analyses kept here),
served by the real server with the model replaced by recorded analyses (tests/fakes.py), so it runs without a model.

    uv run pytest -m e2e
"""

from jspace_demo import server

import socket
import threading
import time

import pytest
import uvicorn
from fakes import FakeLoaded, Recorder

pytestmark = pytest.mark.e2e

MESSAGE = "I'm sanding the old paint off the walls of my 1920s house this weekend."


@pytest.fixture(scope="module")
def app_url(tmp_path_factory):
    patch = pytest.MonkeyPatch()
    patch.setenv("JSPACE_HISTORY_DIR", str(tmp_path_factory.mktemp("history")))
    app = server.create_app("qwen3.5-4b", loader=lambda key: FakeLoaded(), analyse=Recorder())
    with socket.socket() as probe:
        probe.bind((server.HOST, 0))
        port = probe.getsockname()[1]
    running = uvicorn.Server(uvicorn.Config(app, host=server.HOST, port=port, log_level="warning"))
    thread = threading.Thread(target=running.run, daemon=True)
    thread.start()
    for _ in range(200):
        if running.started:
            break
        time.sleep(0.02)
    yield f"http://{server.HOST}:{port}/"
    running.should_exit = True
    thread.join(5)
    patch.undo()


def test_a_message_is_analysed_saved_and_found_again(watched, app_url):
    page, seen = watched
    page.goto(app_url)
    page.evaluate("localStorage.setItem('jspace.lang', 'en')")
    page.goto(app_url + "#/try")
    page.reload()
    page.fill("#message", MESSAGE)
    page.click("button[type=submit]")
    page.wait_for_selector(".result", timeout=15_000)  # the job ran and opened the saved analysis
    assert "#/mine/" in page.evaluate("location.hash")
    assert page.locator(".says-card .reply").inner_text().strip()  # drawn like a prepared case
    page.goto(app_url + "#/try")
    page.wait_for_selector(".mine-list .mine-row")
    assert MESSAGE[:20] in page.locator(".mine-list").inner_text()
    page.locator(".mine-item .icon-button").first.click()  # delete it
    page.wait_for_selector(".mine-list", state="detached")
    assert not seen["errors"], seen["errors"]


def test_the_try_screen_shows_the_model_and_the_local_pages(watched, app_url):
    page, seen = watched
    page.goto(app_url)
    page.evaluate("localStorage.setItem('jspace.lang', 'en')")
    page.goto(app_url + "#/try")
    page.reload()
    page.wait_for_selector("#message", state="visible")
    assert page.locator(".status").get_attribute("href") == "#/try"  # the model's state, in the header
    assert page.locator(".nav-link").all_inner_texts() == ["All cases", "About", "Try your own text"]
    assert not seen["errors"], seen["errors"]


@pytest.fixture(scope="module")
def off_url(app_url):
    """The same machine's app started without a model: it shares the history of the app above."""
    app = server.create_app(None)
    with socket.socket() as probe:
        probe.bind((server.HOST, 0))
        port = probe.getsockname()[1]
    running = uvicorn.Server(uvicorn.Config(app, host=server.HOST, port=port, log_level="warning"))
    thread = threading.Thread(target=running.run, daemon=True)
    thread.start()
    for _ in range(200):
        if running.started:
            break
        time.sleep(0.02)
    yield f"http://{server.HOST}:{port}/"
    running.should_exit = True
    thread.join(5)


def test_the_analyses_stay_listed_while_the_model_is_off(watched, app_url, off_url):
    page, seen = watched
    page.goto(app_url)
    page.evaluate("localStorage.setItem('jspace.lang', 'en')")
    page.goto(app_url + "#/try")
    page.reload()
    page.fill("#message", MESSAGE)
    page.click("button[type=submit]")
    page.wait_for_selector(".result", timeout=15_000)
    page.goto(off_url + "#/try")
    page.wait_for_selector(".mine-list .mine-row")
    assert MESSAGE[:20] in page.locator(".mine-list").inner_text()
    assert "uv run jspace-demo serve" in page.locator(".code").first.inner_text()  # start it with a model
    assert not seen["errors"], seen["errors"]


def test_only_spaces_are_refused_before_sending(watched, app_url):
    page, seen = watched
    page.goto(app_url + "#/try")
    page.wait_for_selector("#message")
    page.fill("#message", "   ")
    page.click("button[type=submit]")
    assert page.locator(".form-status").inner_text().strip()
    assert "#/try" in page.evaluate("location.hash") and not seen["errors"]
