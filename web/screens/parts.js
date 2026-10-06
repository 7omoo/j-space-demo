// Pieces several screens share: the page's rows, a result's mark, the ruler of the watched words, word chips, the
// pressure tag and its definitions, and the model picker.

import { outcome, VARIANTS } from "../core/cases.js";
import { h, icon } from "../core/dom.js";
import { rankPosition, rulerTicks, STRONG, strength, tickText, vocabText, WEAK, wordName } from "../core/format.js";
import { lang, t } from "../core/i18n.js";

export function link(href, props, ...children) {
  return h("a", { href, ...props }, ...children);
}

// A link that leaves the site: it opens in a new tab and carries an arrow.
export function external(href, text) {
  return h(
    "a",
    { href, target: "_blank", rel: "noopener" },
    text,
    h("span", { class: "ext", "aria-hidden": "true" }, "↗"),
  );
}

// A row of a page: its name in the narrow column (a heading, a line under it, a control such as the model picker),
// its content beside it. On a phone the name sits above the content. ``label`` may be text or nodes; ids let a group
// or a link name the row's heading and its line, and a ``focusable`` heading can take the keyboard when the page opens
// at it. A row is not a landmark of its own: the sections around rows name themselves.
export function row(
  { label, sub = null, aside = null, cls = null, level = "h2", labelId = null, subId = null, focusable = false },
  ...children
) {
  return h(
    "section",
    { class: cls ? `row ${cls}` : "row" },
    h(
      "div",
      { class: "row-label" },
      label ? h(level, { class: "row-title", id: labelId, tabindex: focusable ? "-1" : null }, label) : null,
      sub ? h("p", { class: "row-sub", id: subId }, sub) : null,
      aside,
    ),
    h("div", { class: "row-body" }, ...children),
  );
}

// A page's first row: the page's name (h1), a line above it and its lead, beside a few facts or controls in the
// narrow column (text becomes a line of its own).
export function pageHead({ title, lead = null, meta = [], before = null, cls = null }) {
  return h(
    "header",
    { class: cls ? `row page-head ${cls}` : "row page-head" },
    h(
      "div",
      { class: "row-label row-meta" },
      meta.map((item) => (typeof item === "string" ? h("p", {}, item) : item)),
    ),
    h(
      "div",
      { class: "row-body" },
      before,
      h("h1", { tabindex: "-1" }, title),
      lead ? h("p", { class: "lead" }, lead) : null,
    ),
  );
}

// The band of dots that parts a page's groups of rows.
export function dots() {
  return h("div", { class: "dots", "aria-hidden": "true" });
}

export function emptyState(text, ...actions) {
  return h("div", { class: "empty" }, h("p", {}, text), ...actions);
}

// A result's mark: a filled square (sycophantic), a ring (honest), a dash (weak concern, not judged).
export function outcomeMark(result) {
  return h("span", { class: `mark mark-${result}`, "aria-hidden": "true" });
}

// A reply's result: its mark and its name. ``short`` for the home table's cells, ``large`` for a case's result.
export function outcomeLabel(cell, { short = false, large = false } = {}) {
  const result = outcome(cell);
  return h(
    "span",
    { class: `outcome outcome-${result}${large ? " outcome-large" : ""}` },
    outcomeMark(result),
    h("span", { class: "outcome-text" }, t(short ? `result.short.${result}` : `result.${result}`)),
  );
}

export function quoted(text) {
  return t("common.quote", { text });
}

// The same quotation marks around nodes (a message with its pressure sentence set apart inside the quote).
export function quotedNodes(...children) {
  const [open, close] = t("common.quote", { text: "\u0000" }).split("\u0000");
  return [open, ...children, close];
}

export function quotedWord(word) {
  return t("common.quoted_word", { word });
}

// A word the model held: its Japanese name with the English small beside it, marked when it is a concern word.
export function wordChip(item, { concern = false } = {}) {
  const l = lang();
  const name = wordName(item, l);
  const english = item.en ?? item.word ?? item.text;
  return h(
    "span",
    { class: concern ? "chip chip-concern" : "chip" },
    concern ? icon("alert", 12) : null,
    h("span", { class: "chip-name" }, name),
    l === "ja" && english && english !== name ? h("span", { class: "chip-sub", lang: "en" }, english) : null,
  );
}

export function rankText(rank, vocabSize) {
  return vocabSize ? t("rank.of", { rank, vocab: vocabText(vocabSize, lang()) }) : t("rank", { rank });
}

// The watched words, each on a ruler of the whole vocabulary (log scale) with the top ten (strong) and the top hundred
// (weak) shaded and a dot at the word's best rank. The strength is also said in words, for a reader who cannot see it.
// Without the vocabulary's size there is no ruler to draw: the words and their ranks remain.
export function rankRuler(words, vocabSize) {
  const l = lang();
  const scaled = vocabSize > 1;
  const strongTo = rankPosition(STRONG, vocabSize);
  const weakTo = rankPosition(WEAK, vocabSize);
  return h(
    "div",
    { class: "ruler" },
    h(
      "ul",
      { class: "ruler-rows" },
      words.map((w) => {
        const level = strength(w.rank);
        return h(
          "li",
          { class: `ruler-row ruler-${level}` },
          h(
            "span",
            { class: "ruler-word" },
            h("b", {}, wordName(w, l)),
            l === "ja" && w.ja ? h("span", { class: "ruler-en", lang: "en" }, w.word) : null,
          ),
          scaled
            ? h(
                "span",
                { class: "ruler-track", "aria-hidden": "true" },
                h("span", { class: "ruler-zone ruler-zone-weak", style: `--from: ${strongTo}%; --to: ${weakTo}%` }),
                h("span", { class: "ruler-zone ruler-zone-strong", style: `--to: ${strongTo}%` }),
                h("span", { class: "ruler-dot", style: `--x: ${rankPosition(w.rank, vocabSize)}%` }),
              )
            : h("span", { "aria-hidden": "true" }),
          h("span", { class: "ruler-rank", "aria-hidden": "true" }, t("ruler.rank", { rank: w.rank })),
          h(
            "span",
            { class: "sr-only" },
            t("strength.with_rank", { level: t(`strength.${level}`), rank: rankText(w.rank, vocabSize) }),
          ),
        );
      }),
    ),
    scaled
      ? h(
          "div",
          { class: "ruler-scale", "aria-hidden": "true" },
          h(
            "span",
            { class: "ruler-axis" },
            rulerTicks(vocabSize).map((tick) =>
              h(
                "span",
                {
                  class: [
                    "ruler-tick",
                    tick.labelled ? null : "ruler-tick-bare",
                    tick.minor ? "ruler-tick-minor" : null,
                  ]
                    .filter(Boolean)
                    .join(" "),
                  style: `--x: ${tick.x}%`,
                },
                tick.labelled ? tickText(tick.value, l) : "",
              ),
            ),
          ),
        )
      : null,
    scaled
      ? h(
          "p",
          { class: "ruler-legend" },
          h("span", { class: "ruler-key ruler-key-strong" }, t("ruler.strong")),
          h("span", { class: "ruler-key ruler-key-weak" }, t("ruler.weak")),
          h("span", { class: "ruler-key-note" }, t("ruler.scale", { vocab: vocabText(vocabSize, l) })),
        )
      : null,
  );
}

// How a message was asked, by its short name: "risky" (as it is) or one of the pressures; nothing for none given.
export function conditionTag(variant) {
  return variant ? h("span", { class: "tag tag-pressure" }, t(`condition.short.${variant}`)) : null;
}

// The four ways a message was asked, defined: the home page and the About page.
export function conditionList() {
  return h(
    "dl",
    { class: "conditions-list" },
    VARIANTS.flatMap((variant) => [
      h("dt", {}, t(`condition.short.${variant}`)),
      h("dd", {}, t(`condition.about.${variant}`)),
    ]),
  );
}

// What "what it has in mind" means, said wherever a screen sets the reply against the readout.
export function mindCaveat() {
  return h("p", { class: "note mind-caveat" }, icon("info", 14), t("common.mind_caveat"));
}

// Which model's prepared cases the page shows. Placed by the data it switches, so it is clear what changes.
export function modelPicker(data, onChange) {
  if (data.models.length < 2) return null;
  const select = h(
    "select",
    { class: "model-select", "aria-label": t("model.label"), "data-focus": "model-select" },
    data.models.map((m) => h("option", { value: m.key, selected: m.key === data.model }, m.label)),
  );
  select.addEventListener("change", () => {
    data.setModel(select.value);
    onChange();
  });
  return h("label", { class: "model-picker" }, h("span", { class: "model-picker-label" }, t("model.label")), select);
}
