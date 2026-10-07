# Architecture

J-Space Demo has one idea: run a model once on a message, read every layer of that single run with the Jacobian
lens, and show the reply beside what ranked highest inside the model before it replied. Everything else is
plumbing around that idea, kept small enough to read in an afternoon.

## One analysis

```
message ──► generate (greedy, 80 tokens) ──► read every layer (J-lens + logit lens) ──► payload
   │                                                                                     │
   └─ Japanese? translate into English with the same model first          views.build ◄──┘
                                                                                │
                                                                    the screens' data (JSON)
```

- **Generation** (`generate.py`) renders the chat template with thinking off, finds the user's message inside it,
  and decodes greedily, so the same input always gives the same reply.
- **Readout** (`engine.py`) records the residual stream once and maps every position at every layer through the
  lens: the J-lens transports it with the layer's Jacobian before unembedding, the logit lens unembeds it as it is.
  Ranks are counted over the whole vocabulary; the payload keeps the eight best word-like tokens per cell and, for
  each watched word, its best rank over the thinking layers at every position where that is within the top 100.
- **Payload** (`analysis.py`) is plain JSON: tokens and their kinds (special, user, template, reply), the watched
  words' best ranks and where they first came up, the salient concepts per position, the per-layer top words of
  both lenses, and timings. Ranks become 1-based here and only here.
- **Views** (`views/`) turn a payload into what each screen shows, without a model and without I/O. Every rank,
  layer and position on a screen is copied from the payload, never recomputed (tests/unit/test_views.py checks
  this on every recorded case).

One analysis costs one generation and one forward pass. The J-lens itself is a matrix multiplication per layer: no
second model is involved.

## Views

| Module | Where it shows | What it decides |
| --- | --- | --- |
| `views/honesty.py` | the home table, a case's result | the concern (strong / weak) against the reply's stance (warn / agree): four measured cells, shown as three results (honest, sycophantic, or neither when the concern was weak) |
| `views/concern.py` | a case: what it has in mind | the watched words that came up before the reply, best first; the first position where one reached the top 10, the phrase being read there, and what else the model held |
| `views/chain.py` | About: the boot riddle | the top word layer by layer at one token, and where each lens has a word at that token |
| `views/layers.py` | every layer, folded on a case and on About | the top three words per position and layer, both lenses, packed into one word list |

Screens judge by each case's own watch words, as the model reads them: the English word itself, with or without a
leading space (an equivalent in another language is a different token, and does not count). A generic list of risk
words is read out too but never judged by, because it also fires on harmless messages.

Some tokens are never shown, whatever their rank (`UNSHOWN` in `views/words.py`): profanity and sexual words, which
readouts surface from the spam among the web pages a tokenizer was built from, broken characters and page or code
boilerplate. The concept views skip them and the per-layer views show the next word instead. No result depends on
them.

## Prepared cases

`src/jspace_demo/data/cases.json` defines nine scenarios once, in English (what the model reads) and Japanese (what the screens
show). They expand into 38 cases (`cases.py`): the boot riddle, eight risky messages with a harmless message of the
same shape each, and seven of the risky messages under three kinds of social pressure (`pressure.py`, worded
exactly as in the experiment). All 38 are analysed, because the report compares each risky message with its
harmless twin; the site shows 29 (`cases.site_cases`): the boot riddle, and the seven pressure scenarios' messages
asked four ways.

```
jspace-demo precompute --model KEY
  └─ cache/analyses/KEY/<case>.json        full payloads (large, not committed)
jspace-demo export
  ├─ web/data/models.json                  the models with prepared cases
  ├─ web/data/KEY/index.json               one summary row per shown case (the home table)
  └─ web/data/KEY/cases/<case>.json        views, texts and the expert view's layers
```

The Japanese replies depend on the model, so they sit per model in `src/jspace_demo/data/replies/KEY.json`.

## The page

`web/` is plain HTML, CSS and JavaScript modules: no framework and no build step. The prepared cases are always
read from `web/data/`, so the published site and the local app draw them with the same code.
`web/config.json` says which one is running: the local server answers `/config.json` with `{"live": true}` and adds
three things that need a model or this machine.

| Route | Local app only | |
| --- | --- | --- |
| `/api/meta` | ✓ | the loaded model's state |
| `/api/jobs` | ✓ | the user's own analyses, one at a time on the GPU, each listing its steps up front |
| `/api/analyses` | ✓ | their history, kept on this machine (`data/history/`) and never sent anywhere |

Three pages answer one question, whether a reply said what the model had in mind:

| Page | Address | What it shows |
| --- | --- | --- |
| Home | `#/` (`#/cases` brings its table into view) | the claim, one real case, and every case in one table: seven messages asked four ways, each cell the reply's result |
| A case | `#/case/<id>` | the result, the message, the reply in full, the watched words with their ranks and where they came up, and every layer folded at the end |
| About | `#/about` | how results are decided, the four ways of asking, the boot riddle with every layer, the limits and the sources |
| Try your own text | `#/try`, `#/job/<id>`, `#/mine/<id>` | the local app only: the form, the analysis while it runs, and the analyses kept here, drawn like a case |

Screens are addressed by the URL hash, so any sub-path works without server routing; `?model=KEY` on a link picks
the model. Text is always inserted as text (no `innerHTML`), every string comes from `web/i18n/{ja,en}.json`, and a
test keeps the two tables in step and free of unused text.

Every page is a stack of rows parted by hairlines: a narrow column names the row in a monospaced face, the content
sits beside it, and on a phone the name moves above. One accent colour marks only what the model held in mind and
the replies that agreed despite it; an honest reply is a ring in the text colour, a case not judged a grey dash. The
watched words sit on a ruler of the whole vocabulary on a log scale, with the top ten and the top hundred shaded.
Light and dark follow the system until the reader chooses (`web/core/theme.js`; `index.html` applies the choice
before the first paint). Geist and Geist Mono ship in `web/fonts/` and Japanese uses the system's font, so the page
asks no other site for anything.

## Modules

```
src/jspace_demo/
  paths.py       where the package, the page and the caches live
  models.py      the models and their lenses (no torch)
  runtime.py     device and dtype, loading a model and its lens
  vocab.py       word-like tokens and single-token lookups
  engine.py      the readout of every layer
  generate.py    chat template and greedy generation
  analysis.py    one analysis, and the Japanese path
  translate.py   translation with the loaded model
  pressure.py    the experiment's social pressure
  errors.py      errors a user can fix, worded by the screens
  views/         payload → screens
  cases.py       the prepared cases
  glossary.py    Japanese names for the words a readout shows
  precompute.py  analyse and export the prepared cases
  jobs.py        the job queue
  history.py     the history on disk
  server.py      the local app (FastAPI)
  health.py      does a model and its lens read out as expected?
  preflight.py   is another LLM using the GPU?
  cli.py         jspace-demo serve | precompute | export | check

web/
  app.js         the header, the footer, and the screen the address names
  core/          data access, routes, formatting, the text tables and the theme (pure helpers tested without a browser)
  screens/       home, case, about, try, expert (every layer, folded), and the parts they share (rows, marks, ruler)
  fonts/         Geist and Geist Mono (SIL OFL 1.1)
  style.css      the look, in both themes
```
