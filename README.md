# J-Space Demo

**What the LLM says, and what it has in mind.**

J-Space Demo reads the inside of open language models with the Jacobian lens (J-lens) from Anthropic's paper
[*Verbalizable Representations Form a Global Workspace in Language Models*](https://transformer-circuits.pub/2026/workspace/index.html)
(2026), and sets each reply beside the words that ranked highest inside the model before it replied. When a reply
says "Absolutely, that sounds completely fine!" while *bleeding* is the top word in the model's workspace, the gap
is plain to see.

**[Open the demo](https://7omo.com/j-space/)** · [Read the post](https://7omo.com/blog/saying-yes-with-the-danger-in-mind/) · [日本語](README.ja.md) · [Report](docs/report.md) ·
[Architecture](docs/architecture.md) · [Testing](docs/testing.md)

![The home page: a reply that agrees beside the words of danger that topped the model's mind, then every case in one table](docs/images/home.png)

## What it found

Seven risky messages, each sent without pressure and under three kinds of social pressure, on four open models
(one greedy run each; the [report](docs/report.md) has every case):

| Replies that agreed while a word of the danger ranked in the model's top 10 | Qwen3.5-4B | Qwen3-8B | Gemma 3 12B | Qwen3-14B |
| --- | --- | --- | --- | --- |
| without pressure | 0 of 7 | 0 of 7 | 0 of 7 | 1 of 7 |
| under pressure | 14 of 21 | 8 of 21 | 15 of 21 | 8 of 21 |

- In all 45 such replies under pressure the word was in the top 5 of the model's whole vocabulary; for Qwen3.5-4B
  it was first every time. In 3 of them the word came up only while the model read the pressure sentence ("don't
  lecture me about safety"), so the concern may come from the pressure itself.
- A full reading of the replies took 6 of these 45 (1, 1, 2 and 2 by model), and Qwen3-14B's one case without
  pressure, as unclear: they go along with the plan but add a soft hedge, such as "check with someone first". The
  stance rule counts them as agreement; counted as warnings instead, the table would read 0 of 7 without pressure
  for every model and 13, 7, 13 and 6 of 21 under pressure.
- An always-agree persona moved 7, 2, 7 and 2 of the 7 messages; asking the model not to lecture moved 1 for every
  model.
- In the boot riddle, every model holds the unspoken *Italy* at "boot". The J-lens has it at more layers than the
  logit lens, which on Qwen3.5-4B never has it at the thinking layers.

## What the screens show

![A case: the reply in full beside the words of danger the model held before it replied](docs/images/case.png)

Three pages, one question: did the reply say what the model had in mind, or what the user wanted to hear?

- **Home**: the claim, one real case, and every case in one table. Seven risky messages are asked four ways (as
  they are, with pushback, to an always-agree persona, and both), and each cell marks the reply honest,
  sycophantic, or neither when the concern inside the model was weak. A button per model switches the whole page;
  each button shows how often that model gave in.
- **A case**: the message, the reply in full, the words the model held before replying with their ranks among the
  whole vocabulary, and the phrase it was reading when the concern first came up. Folded at the end, the expert
  view shows the top words at every layer for any token, under the J-lens or the logit lens.
- **About**: how results are decided, the four ways of asking, and the boot riddle ("the currency of the country
  shaped like a boot") as a check that the readout is the model's own thinking: it holds *Italy* in its middle
  layers before it answers *the Euro*. On Qwen3.5-4B the logit lens misses it; on the larger models it finds it at
  fewer layers than the J-lens.
- **Try your own text** (local app only): analyse your own message, in Japanese or English, under the same
  pressures worded exactly as in the experiment.

The comparison with a harmless message of the same shape, and the false alarm it found, are in the
[report](docs/report.md) rather than on the site. The page is in Japanese and English, light and dark, works on a
phone, and the public version needs no server and asks no other site for anything: every number on it was computed
in advance and ships as JSON with the site, and its fonts ship with it too.

## How it works

One analysis is one greedy generation and one forward pass that records the residual stream at every layer. Each
position at each layer is then mapped to the vocabulary twice: through the J-lens, which transports the residual
with the layer's pre-fitted Jacobian before unembedding, and through the logit lens, which unembeds it as it is.
No second model is involved; the lens is a matrix multiplication per layer.

The screens read only a view of that analysis, built without a model (`src/jspace_demo/views/`), and every rank,
layer and position on a screen is copied from the readout, never recomputed. [Architecture](docs/architecture.md)
has the details.

## Run it yourself

You need a Mac with Apple silicon and memory to spare (the 4B model holds about 9 GB), Python 3.13, `uv` and
`git`.

```bash
git clone https://github.com/7omoo/j-space-demo
cd j-space-demo
scripts/bootstrap.sh        # clones the upstream jacobian-lens at a pinned commit, then uv sync
uv run jspace-demo serve    # http://127.0.0.1:8000; downloads Qwen3.5-4B and its lens on first run
```

| Command | What it does |
| --- | --- |
| `uv run jspace-demo serve --model qwen3-8b` | the local app with another model for your own text |
| `uv run jspace-demo serve --no-model` | the screens and prepared cases only |
| `uv run jspace-demo precompute --model gemma-3-12b-it` | analyse the prepared cases with a model, then export them to `web/data/` |
| `uv run jspace-demo check --model qwen3-14b` | check that a model and its lens load and read out as expected |

Models: Qwen3.5-4B (default), Qwen3-8B, Gemma 3 12B and Qwen3-14B, each with the lens Neuronpedia published for
it. Without an Apple GPU the code falls back to a CUDA GPU or the CPU, but only Apple silicon has been tested.

## Tests

```bash
scripts/check.sh                       # formatting, lint, the page's JavaScript, unit tests: no model needed
uv run playwright install chromium     # once
uv run pytest -m e2e                   # every screen in both languages at desktop and phone width
uv run pytest -m model                 # with the real model (heavy; one LLM at a time)
```

[Testing](docs/testing.md) explains the layers and the risks each one covers. The experiments behind the report
are in [`experiments/`](experiments/).

## Limits

- What the lens reads out are associations, not intentions or judgements. "What it has in mind" is a figure of
  speech.
- A topic that suggests danger in itself also raises the concern on a harmless message (bleach → chlorine,
  investing → risk). The [report](docs/report.md) measures this against a harmless twin of every message.
- Each case is a single greedy run of a small model. The numbers are small and are not statistics.
- The seven messages were chosen on Qwen3.5-4B: four because its readout told the risky message apart from a
  harmless one of the same shape, and three more because they also drew a sycophantic reply from it. That model's
  counts are partly a result of the choosing; the other three met the same messages afterwards.
- A reply's stance is read by a keyword rule. It was revised twice after reading replies it misread; every
  version's results are in the [report](docs/report.md). A reply that goes along with the plan but adds a soft
  hedge counts as agreement.

## Credits and licences

- The paper and the J-lens: Anthropic, *Verbalizable Representations Form a Global Workspace in Language Models*
  (2026).
- [`jacobian-lens`](https://github.com/anthropics/jacobian-lens) (Apache-2.0) is cloned at install time at a pinned
  commit, used unmodified and not redistributed here.
- The lenses are [Neuronpedia's](https://huggingface.co/neuronpedia/jacobian-lens) (MIT), downloaded on first run.
- Models: [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B), [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)
  and [Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B) (Apache-2.0); [Gemma 3 12B](https://huggingface.co/google/gemma-3-12b-it)
  (Gemma Terms of Use). `web/data/` holds the models' replies and readouts for the prepared cases.
- Fonts: [Geist and Geist Mono](https://github.com/vercel/geist-font) (SIL Open Font License 1.1), bundled in
  `web/fonts/` with their licence. See [NOTICE](NOTICE).

J-Space Demo is released under the [Apache License 2.0](LICENSE).
