# Testing

The tests are organised by what could go wrong and how expensive it is to check. Everything that needs no model
runs on every push; the checks that need a model or a browser are opt-in markers.

| Layer | Needs | Command | What it protects |
| --- | --- | --- | --- |
| Static | nothing | `scripts/check.sh` | formatting (ruff, Prettier) and lint, the page's JavaScript parses, no `innerHTML`, upstream pinned and unedited, the server listens on localhost only |
| Unit (Python) | nothing | `uv run pytest` | engine on upstream's tiny model, payload shape, views on every recorded case, cases, glossary, jobs, history, export, server routes, and the published data of every model |
| Unit (JS) | Node | `node --test tests/js/*.test.mjs` | the screens' pure helpers: routes (earlier links included), formatting, the home table and its counts, the three results, the ruler's positions and marks, the expert view's bands |
| Model | Apple silicon / GPU | `uv run pytest -m model` | the CPU fp32 reference, agreement with upstream `compute_slice`, determinism, the first baseline, the published data reproduces |
| API | a running local app | `JSPACE_URL=http://127.0.0.1:8000 uv run pytest -m api` | jobs one at a time, the Japanese path, social pressure, cancellation, input errors, the history |
| E2E | Chromium | `uv run pytest -m e2e` | every screen × Japanese and English × desktop and phone: renders, no console errors, no sideways scroll at 375 px and the home table's four columns in view, both themes, no request to any other site (no API, no web fonts); the theme follows the system until the reader chooses and is remembered; the page's own fonts load; every case opens from the home table with its reply in full; the table's counts for every model; the model buttons; the boot riddle and every layer for every model; the local app's try page (the form, the running analysis, the analyses kept here) with the model replaced by recorded analyses (`tests/fakes.py`) |
| Performance | a running local app | `uv run python experiments/perf_jobs.py` | 20 Japanese jobs: 90% within 30 s, memory flat, no swap |

CI (`.github/workflows/ci.yml`) runs the static, unit and E2E layers. It publishes nothing: the site is served
from 7omo.com, which copies `web/` at a pinned commit of this repository.

## Risks and the tests that cover them

| Risk | Covered by |
| --- | --- |
| A screen shows a number the readout never produced | `test_views.py::test_every_number_is_copied_from_the_payload` on all 38 cases |
| The site's headline numbers drift from the report | `test_site_data.py`: every model's home table counts the reported sycophancy (of 21 under pressure, of 7 without), and the reported number of those a full reading took as unclear, with none it took as a warning |
| A case cannot be reached, or shows only part of its reply | E2E opens every cell of the home table and finds the reply's opening in the case page |
| A rule is changed to fit the results after seeing them | expected values are the measured ones, pinned per case; rule changes are reported with before and after results (`experiments/stance_agreement.py`) |
| The published data no longer matches the code or the model | `test_model.py::test_the_published_data_is_what_the_model_gives_now` |
| A watch word appears in the message itself, so a hit only copies the input | `test_cases.py::test_watch_words_never_appear_in_what_the_model_reads` |
| The Japanese and English screens say different things | `test_i18n.py`: same keys and placeholders, every key the screens use exists, no key left unused |
| A word on a Japanese screen has no Japanese name | `test_site_data.py`: every word a published screen shows is in the glossary |
| A screen shows profanity or web spam the lens surfaced | `test_site_data.py`: no published screen shows an unshown token (`views.words.UNSHOWN`) |
| The explainer says something one model's readout does not support | the focus view is built from the readout (`test_views.py`) and its text is chosen by what it holds; E2E opens it for every model |
| The published data is stale after a change to the views or the cases | `test_site_data.py` rebuilds the default model's files from the recorded analyses and compares them with `web/data` |
| The static site depends on the server or breaks under a sub-path | E2E serves `web/` under `/j-space-demo/` and fails on any `/api/` request |
| Two analyses share the GPU | `test_jobs.py` and the API test check that a job starts only after the previous one finished |
| Another web page reaches the local app's history or model (DNS rebinding) | `test_server.py`: requests for any host other than 127.0.0.1 or localhost are refused; the server binds 127.0.0.1 only (`test_static.py`) |
| A case names a later place than where the concern first came up, or words that did not come up there | `test_views.py`: the trigger is the first position where a watched word reached the top 10, a message and its pushback version share it, and its words lead the chips there (never one beyond the top 100) |
| A redraw or Back loses the reader's place | E2E: switching the language or the model keeps the keyboard on the control just used; Back from a case returns to the same place in the home table |
| The local app hides the analyses kept here while its model is off | E2E: an app started without a model lists the analyses an earlier run saved |
| An input error reaches the user as an internal message | error codes from `analysis.InputError` through the job to the page (`test_server.py`, `tests/api/test_live.py`) |

## Fixtures

`tests/fixtures/qwen3.5-4b/` holds the 38 recorded analyses, trimmed to what the views read
(`scripts/make_fixtures.py`). They make the view tests exact and model-free. Two tests tie them to the model: the
model test checks that a fresh analysis gives the published data, and `test_site_data.py` that the published data
is what the fixtures build.

## Manual acceptance

Before a release, a person who is not a specialist reads each screen and says in one sentence what it shows. A
screen nobody can sum up is a failed screen, whatever the automated tests say.
