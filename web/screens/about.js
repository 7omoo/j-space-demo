// About: what the demo reads and how the results are decided, a check that the readout is the model's own
// in-between thinking (the boot riddle), the limits, and where everything comes from. Also the not-found screen.

import { h } from "../core/dom.js";
import { wordKey, wordName } from "../core/format.js";
import { lang, t } from "../core/i18n.js";
import { expertDetails } from "./expert.js";
import { conditionList, dots, emptyState, external, link, modelPicker, outcomeLabel, pageHead, row } from "./parts.js";
import { REPOSITORY, runLocally } from "./try.js";

const LIMITS = ["1", "2", "3", "4", "5", "6"];
const FOCUS_WORD = "boot"; // the riddle's clue (the case's focus token in data/cases.json)

// A row of paragraphs (strings) and other content.
function section(title, ...content) {
  return row({ label: title }, ...content.map((p) => (typeof p === "string" ? h("p", {}, p) : p)));
}

function judging() {
  return section(
    t("about.judge.title"),
    h(
      "dl",
      { class: "plain-defs" },
      h("dt", {}, t("about.judge.concern")),
      h("dd", {}, t("about.judge.concern.text")),
      h("dt", {}, t("about.judge.stance")),
      h("dd", {}, t("about.judge.stance.text")),
    ),
    h(
      "ul",
      { class: "result-cards" },
      ["sycophancy", "honest", "weak"].map((result) =>
        h("li", { class: "result-card" }, outcomeLabel(result), h("p", {}, t(`result.about.${result}`))),
      ),
    ),
    t("about.judge.words"),
  );
}

function chainBlock(title, note, chain, concern) {
  const l = lang();
  return h(
    "section",
    { class: "chain-block" },
    h("h3", {}, title),
    h("p", { class: "sub" }, note),
    h(
      "ol",
      { class: "chain" },
      chain.steps.map((step) =>
        h(
          "li",
          // by meaning: the riddle's answer counts in any language here (意大利 is Italy)
          { class: concern.has(wordKey(step.en ?? step.text)) ? "step step-hit" : "step" },
          h("span", { class: "step-word" }, wordName(step, l)),
          l === "ja" && step.ja ? h("span", { class: "sub", lang: "en" }, step.en) : null,
          h(
            "span",
            { class: "step-layers" },
            step.from === step.to
              ? t("chain.layer", { from: step.from })
              : t("chain.layers", { from: step.from, to: step.to }),
          ),
        ),
      ),
    ),
  );
}

// How the two lenses compare on Italy at the focus token ("boot"), in the words the readout supports for this model.
// When the logit lens never has Italy among its best words there, its best rank anywhere in the prompt says how far.
function lensCompare(focus, alerts, vocabSize) {
  const italy = focus?.words.find((w) => w.word === "Italy");
  if (!italy) return null;
  const { j, logit } = italy;
  const key = j && logit ? "both" : j ? "j_only" : logit ? "logit_only" : "none";
  const params = { jfrom: j?.from, jto: j?.to, jrank: j?.rank, lfrom: logit?.from, lto: logit?.to, lrank: logit?.rank };
  const best = alerts.find((a) => a.word === "Italy")?.logit_rank;
  const locale = lang() === "ja" ? "ja-JP" : "en-GB";
  return h(
    "div",
    { class: "callout" },
    h("p", {}, t(`boot.compare.${key}`, params)),
    key === "j_only" && best
      ? h(
          "p",
          {},
          t("boot.compare.logit_best", { lrank: best.toLocaleString(locale), vocab: vocabSize.toLocaleString(locale) }),
        )
      : null,
  );
}

// The comparison counts the English word; a model that holds Italy as another language's word is told so.
function foreignItaly(chain) {
  return chain.steps.find((step) => step.en === "Italy" && step.text !== "Italy")?.text ?? null;
}

async function bootRiddle(app) {
  const { views: v, texts, layers } = await app.data.case("boot-riddle");
  const model = app.data.modelInfo();
  const l = lang();
  const concern = new Set(v.concern_words.map(wordKey));
  const [atBoot, atEnd] = v.chains;
  // The texts speak of "boot"; a model whose tokenizer splits it gets its chains labelled by their own tokens.
  const atFocusWord = v.focus?.token.trim() === FOCUS_WORD;
  return row(
    { label: t("boot.title"), aside: modelPicker(app.data, app.rerender), id: "boot", cls: "boot" },
    h("p", {}, t("boot.lead", { model: model.label })),
    h(
      "div",
      { class: "question" },
      h("p", { class: "label" }, t("boot.question")),
      h("p", { class: "question-text" }, l === "ja" ? texts.message_ja : v.message),
      l === "ja" ? h("p", { class: "sub", lang: "en" }, v.message) : null,
      h("p", { class: "answer" }, t("boot.answer"), " ", h("b", {}, l === "ja" ? texts.reply_ja : v.reply)),
    ),
    chainBlock(
      atFocusWord ? t("boot.at_boot") : t("boot.at_token", { token: atBoot.token.trim() }),
      t("boot.at_boot.note"),
      atBoot,
      concern,
    ),
    atEnd ? chainBlock(t("boot.at_end"), t("boot.at_end.note"), atEnd, concern) : null,
    atFocusWord ? lensCompare(v.focus, v.alerts, model.vocab_size) : null,
    atFocusWord && foreignItaly(atBoot)
      ? h("p", { class: "note" }, t("boot.compare.english", { word: foreignItaly(atBoot) }))
      : null,
    h("p", { class: "boot-conclusion" }, t("boot.conclusion")),
    h("p", { class: "note" }, t("boot.layers", { n: model.n_layers, first: model.band[0], last: model.band[1] })),
    expertDetails({ views: v, layers, model: model.label, position: v.focus?.position }),
  );
}

function licences(models) {
  const line = (href, text, licence) => h("tr", {}, h("td", {}, external(href, text)), h("td", {}, licence));
  return h(
    "div",
    { class: "table-scroll" },
    h(
      "table",
      { class: "plain-table" },
      h(
        "thead",
        {},
        h(
          "tr",
          {},
          h("th", { scope: "col" }, t("about.licence.what")),
          h("th", { scope: "col" }, t("about.licence.licence")),
        ),
      ),
      h(
        "tbody",
        {},
        models.map((m) => line(`https://huggingface.co/${m.hf_name}`, m.label, m.licence)),
        line("https://huggingface.co/neuronpedia/jacobian-lens", t("about.licence.lenses"), "MIT"),
        line("https://github.com/anthropics/jacobian-lens", t("about.licence.upstream"), "Apache-2.0"),
        line("https://github.com/vercel/geist-font", t("about.licence.fonts"), "SIL OFL 1.1"),
        line(REPOSITORY, t("about.licence.this"), "Apache-2.0"),
      ),
    ),
  );
}

export async function about(route, app) {
  const { data } = app;
  return h(
    "div",
    { class: "page about" },
    pageHead({
      title: t("about.title"),
      lead: t("about.lead"),
      meta: [t("about.meta.method"), t("about.meta.source")],
    }),
    section(t("about.what.title"), t("about.what.1"), t("about.what.2")),
    judging(),
    section(t("about.conditions.title"), t("about.conditions.lead"), conditionList()),
    dots(),
    await bootRiddle(app),
    dots(),
    section(t("about.use.title"), t("about.use.1")),
    section(
      t("about.limits.title"),
      h(
        "ul",
        { class: "plain-list" },
        LIMITS.map((n) => h("li", {}, t(`about.limits.${n}`))),
      ),
    ),
    section(t("about.texts.title"), t("about.texts.1")),
    section(
      t("about.report.title"),
      t("about.report.1"),
      h("p", {}, external(`${REPOSITORY}/blob/main/docs/report.md`, t("about.report.link"))),
    ),
    dots(),
    section(t("about.licence.title"), licences(data.models)),
    section(
      t("about.sources.title"),
      h(
        "ul",
        { class: "plain-list" },
        h("li", {}, external("https://transformer-circuits.pub/2026/workspace/index.html", t("about.sources.paper"))),
        h("li", {}, external(REPOSITORY, t("about.sources.code"))),
      ),
    ),
    section(
      t("about.local.title"),
      t("about.local.1"),
      runLocally(),
      t("about.local.2"),
      h("pre", { class: "code" }, "uv run jspace-demo precompute --model qwen3-8b"),
    ),
  );
}

export async function notFound() {
  return h("div", { class: "page" }, emptyState(t("common.not_found"), link("#/", { class: "button" }, t("nav.home"))));
}
