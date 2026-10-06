"""Every screen of the static site in both languages at desktop and phone width (Chromium, via Playwright).

    uv run playwright install chromium   # once
    uv run pytest -m e2e

Checks what a visitor would notice first: the screen renders, nothing errors in the console, nothing scrolls
sideways at 375 px, the static site asks nothing of any other site (no API, no web fonts) in either theme, and every
case opens from the home table with its reply in full.
"""

from jspace_demo.paths import WEB_DIR

import json
from itertools import pairwise

import pytest

pytestmark = pytest.mark.e2e

MODELS_FILE = json.loads((WEB_DIR / "data" / "models.json").read_text(encoding="utf-8"))
DEFAULT_MODEL = MODELS_FILE["default"]
MODELS = MODELS_FILE["models"]
VIEWPORTS = {"desktop": {"width": 1280, "height": 800}, "phone": {"width": 375, "height": 667}}
SCREENS = {  # hash -> a selector that only that screen draws
    "#/": ".duel",
    "#/cases": ".case-grid",
    "#/case/warfarin-aspirin-persona": ".result-sycophancy",
    "#/case/lead-paint-risky": ".result-honest",
    "#/case/lead-paint-risky?model=qwen3-8b": ".result-weak",
    "#/case/lead-paint-persona": ".says-card .reply",
    "#/about": ".chain",
    "#/try": ".code",
    "#/case/boot-riddle": ".empty",
    "#/nowhere": ".empty",
}


def open_screen(page, site_url, hash_, lang, model=DEFAULT_MODEL):
    page.goto(site_url)
    page.evaluate(
        "([lang, model]) => { localStorage.setItem('jspace.lang', lang); localStorage.setItem('jspace.model', model); }",
        [lang, model],
    )
    page.goto(site_url + hash_)
    page.reload()


def published(model: str, name: str) -> dict:
    return json.loads((WEB_DIR / "data" / model / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize("theme", ["light", "dark"])
@pytest.mark.parametrize("viewport", VIEWPORTS)
@pytest.mark.parametrize("lang", ["ja", "en"])
@pytest.mark.parametrize("hash_", SCREENS)
def test_every_screen_renders_cleanly(watched, site_url, hash_, lang, viewport, theme):
    page, seen = watched
    page.set_viewport_size(VIEWPORTS[viewport])
    page.emulate_media(color_scheme=theme)  # the reader has not chosen: the system's theme
    open_screen(page, site_url, hash_, lang)
    page.wait_for_selector(SCREENS[hash_], state="visible", timeout=10_000)
    assert page.evaluate("document.documentElement.lang") == lang
    assert page.evaluate("document.documentElement.dataset.theme") == theme
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1"), "scrolls sideways"
    grid = page.locator(".grid-section .table-scroll")
    if grid.count():  # every column of the home table is in view, without scrolling the table either
        assert grid.evaluate("box => box.scrollWidth <= box.clientWidth + 1"), "the table scrolls sideways"
    assert page.locator("h1").count() >= 1 or SCREENS[hash_] == ".empty"
    assert not seen["errors"], seen["errors"]
    assert not [url for url in seen["requests"] if "/api/" in url]
    assert not [url for url in seen["requests"] if not url.startswith(site_url)], "asked another site"


def test_the_header_names_the_pages_of_the_static_site(watched, site_url):
    page, _ = watched
    open_screen(page, site_url, "#/", "en")
    page.wait_for_selector(".duel")
    assert page.locator(".nav-link").all_inner_texts() == ["All cases", "About"]


def test_the_home_page_sets_the_reply_against_the_readout(watched, site_url):
    page, _ = watched
    open_screen(page, site_url, "#/", "en")
    duel = page.locator(".duel")
    duel.wait_for(state="visible")
    assert duel.locator(".duel-says .duel-quote").inner_text().strip()
    assert duel.locator(".duel-mind .ruler-row").count() >= 1
    assert "What the LLM says" in duel.inner_text() and "What the LLM has in mind" in duel.inner_text()


def test_every_case_opens_from_the_table_with_its_reply_in_full(watched, site_url):
    page, seen = watched
    open_screen(page, site_url, "#/", "en")
    page.wait_for_selector(".case-grid")
    hrefs = page.locator(".case-grid .grid-cell").evaluate_all("links => links.map((a) => a.getAttribute('href'))")
    assert len(hrefs) == 28
    for href in hrefs:
        case_id = href.removeprefix("#/case/")
        views = published(DEFAULT_MODEL, f"cases/{case_id}.json")["views"]
        page.goto(site_url + href)
        reply = page.locator(".says-card .reply")
        reply.wait_for(state="visible")
        first_line = views["reply"].replace("**", "").strip().splitlines()[0][:30]
        assert first_line in reply.inner_text(), case_id  # the whole reply, not an excerpt
        assert page.locator(".result .outcome-large").count() == 1, case_id
    assert not seen["errors"], seen["errors"]


@pytest.mark.parametrize("model", [m["key"] for m in MODELS])
def test_the_table_counts_what_the_data_says_for_every_model(watched, site_url, model):
    page, _ = watched
    open_screen(page, site_url, "#/cases", "en", model)
    page.wait_for_selector(".case-grid")
    rows = published(model, "index.json")["cases"]
    pressured = [r for r in rows if r["kind"] == "pressure"]
    agreeing = sum(r["honesty"]["cell"] == "sycophancy" for r in pressured)
    assert f"sycophantic in {agreeing} of {len(pressured)} cases" in page.locator(".grid-summary").inner_text()
    assert page.locator(".case-grid .outcome-sycophancy").count() == agreeing + sum(
        r["variant"] == "risky" and r["honesty"]["cell"] == "sycophancy" for r in rows
    )


def test_a_model_button_switches_the_page_and_is_remembered(watched, site_url):
    page, seen = watched
    open_screen(page, site_url, "#/", "en")
    page.wait_for_selector(".case-grid")
    other = MODELS[1]
    page.get_by_role("button", name=other["label"]).click()
    page.wait_for_function(
        "label => document.querySelector('#grid-title')?.innerText.includes(label)", arg=other["label"]
    )
    assert page.evaluate("localStorage.getItem('jspace.model')") == other["key"]
    assert page.get_by_role("button", name=other["label"]).get_attribute("aria-pressed") == "true"
    assert not seen["errors"], seen["errors"]


def test_the_cases_link_brings_the_table_into_view(watched, site_url):
    page, _ = watched
    open_screen(page, site_url, "#/cases", "en")
    page.wait_for_selector(".case-grid")
    page.wait_for_function("window.scrollY > 0")  # the table sits below the example
    assert page.evaluate("document.activeElement.id") == "grid-title"


def test_the_expert_view_opens_and_follows_the_chosen_token_and_lens(watched, site_url):
    page, seen = watched
    open_screen(page, site_url, "#/case/lead-paint-persona", "en")
    page.wait_for_selector(".expert")
    assert page.locator(".strata").count() == 0  # drawn when first opened
    page.locator(".expert > summary").click()
    page.wait_for_selector(".strata .stratum")
    page.locator(".tok", has_text="mask").first.click()
    assert "mask" in page.locator(".strata-caption").inner_text()
    j_words = page.locator(".strata").inner_text()
    page.get_by_role("button", name="Logit lens").click()
    assert page.locator(".strata").inner_text() != j_words
    page.keyboard.press("ArrowRight")  # the strip keeps the keyboard
    assert page.locator(".tok-on").count() == 1 and not seen["errors"]


@pytest.mark.parametrize("lang", ["ja", "en"])
@pytest.mark.parametrize("model", [m["key"] for m in MODELS])
def test_the_boot_riddle_reads_every_model(watched, site_url, model, lang):
    page, seen = watched
    open_screen(page, site_url, "#/about", lang, model)
    page.wait_for_selector(".boot .chain")
    assert page.locator(".boot .chain-block").count() == 2  # at "boot" and at the last word, whichever the model
    assert page.locator(".boot .callout").inner_text().strip()
    page.locator(".boot .expert > summary").click()
    page.wait_for_selector(".boot .strata .stratum")
    assert not seen["errors"], seen["errors"]


def test_the_language_switch_redraws_the_page(watched, site_url):
    page, _ = watched
    open_screen(page, site_url, "#/about", "ja")
    page.wait_for_selector(".chain")
    assert "このデモについて" in page.locator("h1").inner_text()
    page.click(".lang-toggle")
    page.wait_for_function("document.documentElement.lang === 'en'")
    assert "About this demo" in page.locator("h1").inner_text()


def test_the_picker_wins_over_a_model_named_in_the_address(watched, site_url):
    page, seen = watched
    open_screen(page, site_url, f"#/case/lead-paint-risky?model={MODELS[1]['key']}", "en")
    page.wait_for_selector(".result")
    chosen = MODELS[2]["key"]
    page.select_option(".model-select", chosen)
    page.wait_for_function("key => location.hash.endsWith('?model=' + key)", arg=chosen)
    page.wait_for_selector(".result")
    assert page.locator(".model-select").input_value() == chosen
    assert page.evaluate("localStorage.getItem('jspace.model')") == chosen
    assert not seen["errors"], seen["errors"]


def test_an_earlier_link_to_a_tab_opens_the_case(watched, site_url):
    page, seen = watched
    open_screen(page, site_url, "#/case/warfarin-aspirin-persona/honesty", "en")
    page.wait_for_selector(".result-sycophancy")
    assert not seen["errors"], seen["errors"]


def test_the_skip_link_moves_the_keyboard_and_keeps_the_screen(watched, site_url):
    page, seen = watched
    open_screen(page, site_url, "#/about", "en")
    page.wait_for_selector(".chain")
    page.locator(".skip").focus()
    page.keyboard.press("Enter")
    assert page.evaluate("location.hash") == "#/about"
    assert page.evaluate("document.activeElement.id") == "view"
    assert page.locator(".chain").count() >= 1 and not seen["errors"]


def test_a_case_marks_the_phrase_it_was_reading_when_the_concern_came_up(watched, site_url):
    page, _ = watched
    open_screen(page, site_url, "#/case/lead-paint-persona", "ja")
    page.wait_for_selector(".message-card mark")
    assert page.locator(".message-card mark").inner_text().strip() == "I don't have a mask,"
    assert "「マスクは持っていない」" in page.locator(".trigger-where").inner_text()  # named in Japanese, no comma


def test_a_redraw_keeps_the_keyboard_where_it_was(watched, site_url):
    page, _ = watched
    open_screen(page, site_url, "#/", "en")
    page.wait_for_selector(".case-grid")
    page.locator(".lang-toggle").focus()
    page.keyboard.press("Enter")
    page.wait_for_function("document.documentElement.lang === 'ja'")
    assert page.evaluate("document.activeElement.dataset.focus") == "lang"
    other = MODELS[1]
    page.locator(f'[data-focus="model-{other["key"]}"]').focus()
    page.keyboard.press("Enter")
    page.wait_for_function(
        "label => document.querySelector('#grid-title')?.innerText.includes(label)", arg=other["label"]
    )
    assert page.evaluate("document.activeElement.dataset.focus") == f"model-{other['key']}"


def test_back_from_a_case_returns_to_the_same_place_in_the_table(watched, site_url):
    page, _ = watched
    open_screen(page, site_url, "#/", "en")
    page.wait_for_selector(".case-grid")
    cell = page.locator(".case-grid .grid-cell").last
    cell.scroll_into_view_if_needed()
    page.wait_for_timeout(300)  # the page saves the scroll a moment after it stops
    before = page.evaluate("window.scrollY")
    assert before > 0
    cell.click()
    page.wait_for_selector(".says-card")
    page.go_back()
    page.wait_for_selector(".case-grid")
    page.wait_for_function("y => Math.abs(window.scrollY - y) <= 2", arg=before)


def test_the_cases_link_on_the_cases_screen_goes_back_to_the_table(watched, site_url):
    page, _ = watched
    open_screen(page, site_url, "#/cases", "en")
    page.wait_for_selector(".case-grid")
    page.wait_for_function("window.scrollY > 0")
    page.evaluate("window.scrollTo(0, 0)")
    page.get_by_role("link", name="All cases").click()
    page.wait_for_function("window.scrollY > 0")
    assert page.evaluate("document.activeElement.id") == "grid-title"


@pytest.mark.parametrize("scheme", ["light", "dark"])
def test_the_theme_starts_from_the_system_and_keeps_the_readers_choice(watched, site_url, scheme):
    page, seen = watched
    page.emulate_media(color_scheme=scheme)
    open_screen(page, site_url, "#/", "en")
    page.wait_for_selector(".duel")
    assert page.evaluate("document.documentElement.dataset.theme") == scheme
    other = "light" if scheme == "dark" else "dark"
    page.locator(".theme-toggle").click()
    assert page.evaluate("document.documentElement.dataset.theme") == other
    assert page.evaluate("localStorage.getItem('jspace.theme')") == other
    assert page.evaluate("document.activeElement.dataset.focus") == "theme"  # the redrawn button keeps the keyboard
    page.reload()  # the choice wins over the system from then on
    page.wait_for_selector(".duel")
    assert page.evaluate("document.documentElement.dataset.theme") == other
    background = page.evaluate("getComputedStyle(document.body).backgroundColor")
    assert background == ("rgb(7, 8, 10)" if other == "dark" else "rgb(255, 255, 255)")
    assert not seen["errors"], seen["errors"]


def test_the_page_uses_its_own_fonts(watched, site_url):
    page, seen = watched
    open_screen(page, site_url, "#/", "ja")
    page.wait_for_selector(".duel")
    page.wait_for_function("document.fonts.status === 'loaded'")
    assert page.evaluate("document.fonts.check('16px Geist') && document.fonts.check('13px \"Geist Mono\"')")
    fonts = [url for url in seen["requests"] if url.endswith(".woff2")]
    assert fonts and all(url.startswith(site_url + "fonts/") for url in fonts)


@pytest.mark.parametrize("lang", ["ja", "en"])
@pytest.mark.parametrize("width", [600, 900, 1024])
def test_the_home_table_and_the_model_buttons_fit_between_phone_and_desktop(watched, site_url, width, lang):
    page, seen = watched
    page.set_viewport_size({"width": width, "height": 900})
    open_screen(page, site_url, "#/", lang)
    page.wait_for_selector(".case-grid")
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1"), "scrolls sideways"
    assert page.locator(".grid-section .table-scroll").evaluate("box => box.scrollWidth <= box.clientWidth + 1")
    names = page.locator(".case-grid th[scope=row]").evaluate_all(
        "cells => cells.map((cell) => cell.getBoundingClientRect().width)"
    )
    assert min(names) >= 110, names  # a message's name keeps a readable width, not a letter per line
    assert page.locator(".grid-cell").evaluate_all("cells => cells.every((c) => c.scrollWidth <= c.clientWidth + 1)")
    tops = page.locator(".model-button").evaluate_all(
        "buttons => buttons.map((button) => Math.round(button.getBoundingClientRect().top))"
    )
    per_row = [tops.count(top) for top in sorted(set(tops))]
    assert len(set(per_row)) == 1, per_row  # full rows of buttons: no row ends in an empty cell
    assert not seen["errors"], seen["errors"]


def test_the_theme_switch_follows_the_system_until_the_reader_chooses(watched, site_url):
    page, seen = watched
    page.emulate_media(color_scheme="dark")
    open_screen(page, site_url, "#/", "en")
    page.wait_for_selector(".duel")
    page.emulate_media(color_scheme="light")  # the system changes while the page is open
    page.wait_for_function("document.documentElement.dataset.theme === 'light'")
    assert page.locator(".theme-toggle").get_attribute("aria-label") == "Switch to dark"
    page.locator(".theme-toggle").click()  # the first click switches: the button named the theme it leads to
    assert page.evaluate("document.documentElement.dataset.theme") == "dark"
    page.emulate_media(color_scheme="dark")
    page.emulate_media(color_scheme="light")  # once chosen, the system no longer decides
    page.wait_for_timeout(200)
    assert page.evaluate("document.documentElement.dataset.theme") == "dark"
    assert not seen["errors"], seen["errors"]


@pytest.mark.parametrize("viewport", VIEWPORTS)
def test_the_ruler_says_each_words_rank_and_its_numbers_never_collide(watched, site_url, viewport):
    page, seen = watched
    page.set_viewport_size(VIEWPORTS[viewport])
    open_screen(page, site_url, "#/case/warfarin-aspirin-persona", "en")
    page.wait_for_selector(".mind-card .ruler-row")
    watched_words = published(DEFAULT_MODEL, "cases/warfarin-aspirin-persona.json")["views"]["watched"]
    rows = page.locator(".mind-card .ruler-row")
    assert rows.count() == len(watched_words)
    for i, word in enumerate(watched_words):  # read out in words, not only drawn
        assert f"rank {word['rank']} of" in rows.nth(i).locator(".sr-only").text_content()
    boxes = page.locator(".mind-card .ruler-tick").evaluate_all(
        "ticks => ticks.map((tick) => { const r = tick.getBoundingClientRect(); return [r.left, r.right, r.width]; })"
    )
    shown = [box for box in boxes if box[2] > 0.5]  # the numbers drawn (a bare or a hidden minor mark has none)
    assert len(shown) >= 4 and all(a[1] <= b[0] + 1 for a, b in pairwise(shown)), shown
    assert not seen["errors"], seen["errors"]


def test_the_marks_and_the_ruler_stay_visible_in_forced_colours(watched, site_url):
    page, seen = watched
    page.emulate_media(forced_colors="active")
    open_screen(page, site_url, "#/", "en")
    page.wait_for_selector(".case-grid")
    transparent = "rgba(0, 0, 0, 0)"
    mark = page.locator(".case-grid .mark-sycophancy").first
    assert mark.evaluate("m => getComputedStyle(m).backgroundColor") != transparent
    dot = page.locator(".duel-mind .ruler-dot").first
    assert dot.evaluate("d => getComputedStyle(d).backgroundColor") != transparent
    assert not seen["errors"], seen["errors"]
