"""The page's string tables: Japanese and English must say the same things, and say only what the screens use."""

from jspace_demo import cases
from jspace_demo.analysis import InputError
from jspace_demo.paths import WEB_DIR
from jspace_demo.views.honesty import CELLS

import json
import re

TABLES = {lang: json.loads((WEB_DIR / "i18n" / f"{lang}.json").read_text(encoding="utf-8")) for lang in ("ja", "en")}
VARIANTS = ["risky", "pushback", "persona", "both"]
RESULTS = ["honest", "sycophancy", "weak"]
STATUSES = ["off", "idle", "loading", "ready", "error"]  # the local app only: the static site shows no status
TEMPLATED = {  # keys the screens build from data: every value the data can take
    "about.limits": list("1234567"),
    "boot.compare": ["j_only", "both", "logit_only", "none"],
    "case.result": ["honest", "sycophancy"],
    "case.stance": ["warn", "agree"],
    "case.trigger": ["after_message", "before_message", "in_phrase", "in_phrase_translated"],
    "condition.about": VARIANTS,
    "condition.short": VARIANTS,
    "error": [*InputError.MESSAGES, "internal"],
    "expert.band": ["read", "think", "speak"],
    "expert.lens": ["j", "logit"],
    "home.result": RESULTS,
    "job.step": ["translating", "generating", "reading", "translating_reply"],
    "result": RESULTS,
    "result.about": RESULTS,
    "result.short": RESULTS,
    "status": STATUSES,
    "status.about": ["off", "idle", "loading", "error"],  # a model that is ready shows the form instead
    "theme.to": ["dark", "light"],
    "strength": ["strong", "weak"],  # a screen lists watched words within the top 100 only
    "try.example": [f"{n}.{part}" for n in "123" for part in ("label", "text")],
    "watchset": list(cases.watch_sets()),
}
NAMESPACES = (
    "about|app|boot|case|chain|common|condition|error|expert|footer|home|job|lang|mine|model|nav|rank|result|ruler|"
    "status|strength|theme|try|watchset"
)
KEY = re.compile(rf"\"((?:{NAMESPACES})(?:\.[a-z0-9_]+)+)\"")


def screen_sources() -> str:
    return "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(WEB_DIR.rglob("*.js"))
        if "data" not in path.relative_to(WEB_DIR).parts
    )


def used_keys() -> set[str]:
    source = screen_sources()
    used = set(KEY.findall(source)) | set(re.findall(r"\bt\(\s*\"([^\"]+)\"", source))
    return used | {f"{prefix}.{value}" for prefix, values in TEMPLATED.items() for value in values}


def test_both_languages_have_the_same_keys_and_no_empty_text():
    assert TABLES["ja"].keys() == TABLES["en"].keys()
    assert all(text.strip() for table in TABLES.values() for text in table.values())


def test_placeholders_match():
    for key in TABLES["ja"]:
        placeholders = {lang: set(re.findall(r"\{(\w+)\}", TABLES[lang][key])) for lang in TABLES}
        assert placeholders["ja"] == placeholders["en"], key


def test_every_result_a_view_can_give_has_a_label():
    """The four cells of the measurement show as three results; each has its labels."""
    shown = {cell if cell in ("honest", "sycophancy") else "weak" for cell in CELLS.values()}
    assert shown == set(RESULTS)
    labels = [f"{prefix}.{r}" for prefix in ("result", "result.short", "result.about", "home.result") for r in RESULTS]
    assert not [key for key in labels if key not in TABLES["ja"]]


def test_every_key_the_screens_use_exists():
    assert not sorted(k for k in used_keys() if k not in TABLES["ja"] and not k.endswith("."))


def test_every_key_in_the_tables_is_used():
    """No text is left behind by a screen that changed: every key is named by a screen or built from data."""
    assert not sorted(set(TABLES["ja"]) - used_keys())
