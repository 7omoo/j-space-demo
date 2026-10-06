// One case on one page: the result, the message, the reply in full, what the model had in mind before it replied,
// and, folded at the end, every layer's readout. The same page shows a prepared case (#/case/<id>) and an analysis of
// the user's own text (#/mine/<id>).

import { outcome } from "../core/cases.js";
import { h, icon } from "../core/dom.js";
import { cleanReply, splitBySpan, STRONG, truncate, wordName } from "../core/format.js";
import { lang, pick, t } from "../core/i18n.js";
import { expertDetails } from "./expert.js";
import {
  conditionTag,
  emptyState,
  link,
  mindCaveat,
  modelPicker,
  outcomeLabel,
  pageHead,
  quoted,
  quotedWord,
  rankRuler,
  row,
  wordChip,
} from "./parts.js";

export async function caseScreen(route, app) {
  const built = await app.data.case(route.id);
  const model = app.data.modelInfo();
  return casePage({
    ...built,
    meta: {
      variant: built.case.variant,
      scenario: pick(built.case.scenario_title),
      title: pick(built.case.title),
      model: model.label,
      picker: modelPicker(app.data, app.rerender),
    },
    back: link("#/cases", { class: "back" }, icon("arrowLeft", 14), t("case.back")),
  });
}

export async function mineScreen(route, app) {
  const { analysis, views, texts, layers } = await app.data.analysis(route.id);
  return casePage({
    views,
    texts,
    layers,
    meta: {
      variant: analysis.input.condition,
      scenario: t(`watchset.${analysis.input.watch_set}`),
      title: truncate(texts.message_ja ?? analysis.input.message, 70),
      model: views.model,
      picker: null,
    },
    back: link("#/try", { class: "back" }, icon("arrowLeft", 14), t("case.back_mine")),
  });
}

function casePage({ meta, views: v, texts, layers, back }) {
  if (v.mode === "raw") {
    // the explainer (the boot riddle) is told on the About page
    return h(
      "div",
      { class: "page case" },
      emptyState(t("case.explainer"), link("#/about", { class: "button" }, t("nav.about"))),
    );
  }
  return h(
    "div",
    { class: "page case" },
    pageHead({
      title: meta.title,
      meta: [back, meta.picker ?? t("case.model", { model: meta.model })],
      before: h(
        "p",
        { class: "case-meta" },
        conditionTag(meta.variant),
        h("span", { class: "sub" }, meta.scenario),
        v.translation?.message_source ? h("span", { class: "tag" }, t("case.translated")) : null,
      ),
      cls: "case-head",
    }),
    resultRow(v),
    messageRow(v, texts),
    replyRow(v, texts),
    mindRow(v, texts),
    row(
      { label: t("case.rule.title"), cls: "rule-row" },
      h("p", { class: "note" }, t("case.rule", { words: watchList(v) }), " ", link("#/about", {}, t("case.rule.more"))),
    ),
    layers
      ? row(
          { label: t("expert.label"), cls: "expert-row" },
          expertDetails({ views: v, layers, model: meta.model, position: v.trigger?.position }),
        )
      : null,
  );
}

// --------------------------------------------------------------------------- the result

function strongNames(v) {
  const l = lang();
  const words = (v.watched ?? []).filter((w) => w.rank <= STRONG).slice(0, 3);
  return words.map((w) => quotedWord(wordName(w, l))).join(l === "ja" ? "" : ", ");
}

function resultLine(v) {
  const hon = v.honesty;
  const result = outcome(hon.cell);
  if (result !== "weak") return t(`case.result.${result}`, { words: strongNames(v) });
  return hon.word
    ? t("case.result.weak", { word: quotedWord(wordName(hon.word, lang())), rank: hon.word.rank })
    : t("case.result.weak_none");
}

function resultRow(v) {
  if (!v.honesty)
    return row(
      { label: t("case.result.title"), cls: "result result-none" },
      h("p", { class: "result-line" }, t("case.result.no_reply")),
    );
  return row(
    { label: t("case.result.title"), cls: `result result-${outcome(v.honesty.cell)}` },
    outcomeLabel(v.honesty.cell, { large: true }),
    h("p", { class: "result-line" }, resultLine(v)),
  );
}

// --------------------------------------------------------------------------- the message

// The English the model read: the phrase it was reading when the concern came up marked, and a pressure sentence
// added to the message set apart.
function marked(v, pushback) {
  const span = v.trigger?.where === "in_message" ? v.trigger.phrase?.char_span : null;
  const [before, middle, after] = splitBySpan(v.message, span);
  const added = pushback && after.endsWith(pushback) ? pushback : null;
  return [
    before,
    middle ? h("mark", {}, middle) : null,
    added ? after.slice(0, -added.length) : after,
    added ? h("span", { class: "pushback" }, added) : null,
  ];
}

function messageRow(v, texts) {
  const l = lang();
  const translated = Boolean(v.translation?.message_source);
  const own = l === "ja" && texts.message_ja;
  return row(
    { label: t("case.message"), cls: "message-card" },
    own
      ? h(
          "p",
          { class: "message" },
          texts.message_ja,
          texts.pushback_ja ? h("span", { class: "pushback" }, " ", texts.pushback_ja) : null,
        )
      : h("p", { class: "message", lang: "en" }, marked(v, texts.pushback)),
    own
      ? h(
          "p",
          { class: "original", lang: "en" },
          h("span", { class: "label" }, t(translated ? "case.analysed" : "case.original")),
          " ",
          marked(v, texts.pushback),
        )
      : null,
    !own && translated
      ? h(
          "p",
          { class: "original", lang: "ja" },
          h("span", { class: "label" }, t("case.source")),
          " ",
          texts.message_ja,
        )
      : null,
    v.system
      ? h(
          "div",
          { class: "system-note" },
          h("p", { class: "label" }, t("case.system")),
          l === "ja" ? h("p", {}, t("condition.about.persona")) : null,
          h("p", { lang: "en", class: "system-text" }, v.system),
        )
      : null,
  );
}

// --------------------------------------------------------------------------- what it says

function replyRow(v, texts) {
  const l = lang();
  const replyJa = texts.reply_ja ?? v.translation?.reply_ja;
  const shown = l === "ja" && replyJa ? cleanReply(replyJa) : cleanReply(v.reply);
  const hon = v.honesty;
  return row(
    { label: t("case.says"), cls: "says-card" },
    h("p", { class: "reply" }, shown ? (v.reply_cut ? `${shown}…` : shown) : t("case.no_reply")),
    v.reply_cut ? h("p", { class: "note" }, t("case.reply_cut")) : null,
    l === "ja" && replyJa
      ? h(
          "p",
          { class: "original", lang: "en" },
          h("span", { class: "label" }, t("case.original")),
          " ",
          cleanReply(v.reply),
        )
      : null,
    hon
      ? h(
          "div",
          { class: "stance" },
          h("p", { class: `stance-line stance-${hon.stance}` }, t(`case.stance.${hon.stance}`)),
          h(
            "p",
            { class: "sub" },
            hon.cue ? t("case.stance.cue", { cue: quoted(hon.cue) }) : t("case.stance.no_cue"),
            " ",
            h("span", { lang: "en" }, quoted(truncate(hon.sentence, 180))),
          ),
        )
      : null,
  );
}

// --------------------------------------------------------------------------- what it has in mind

// Where the concern came up: the phrase being read (in Japanese when it was prepared), or before or after the message.
// A comma or colon at the phrase's end is left out of the quotation.
function triggerWhere(v, texts) {
  const trig = v.trigger;
  if (trig.where !== "in_message") return t(`case.trigger.${trig.where}`);
  const ja = lang() === "ja" ? texts.phrase_ja : null;
  const phrase = (ja || trig.phrase.text).replace(/[,;:、，]+$/, "");
  const key = !ja && v.translation?.message_source ? "case.trigger.in_phrase_translated" : "case.trigger.in_phrase";
  return t(key, { phrase: quoted(phrase) });
}

function mindRow(v, texts) {
  const l = lang();
  const watched = v.watched ?? [];
  const concern = new Set((v.trigger?.words ?? []).map((w) => w.toLowerCase()));
  const listed = new Set(watched.map((w) => w.word));
  const voiced = (v.honesty?.voiced ?? []).filter((word) => listed.has(word)); // of the words listed above
  return row(
    { label: t("case.mind"), sub: t("case.mind.sub"), cls: "mind-card" },
    watched.length ? rankRuler(watched, v.vocab_size) : h("p", { class: "sub" }, t("case.watched.none")),
    watched.length ? h("p", { class: "note" }, t("ruler.note")) : null,
    v.unread_words?.length
      ? h("p", { class: "note" }, t("case.watched.unread", { words: v.unread_words.join(", ") }))
      : null,
    voiced.length
      ? h(
          "p",
          { class: "note" },
          t("case.watched.voiced", {
            words: voiced
              .map((word) => wordName(v.alerts.find((a) => a.word === word) ?? { word }, l))
              .join(l === "ja" ? "、" : ", "),
          }),
        )
      : null,
    v.trigger
      ? h(
          "div",
          { class: "trigger" },
          h("p", { class: "trigger-where" }, icon("arrowUp", 15), triggerWhere(v, texts)),
          h(
            "p",
            { class: "chips" },
            v.trigger.concepts.map((c) =>
              wordChip(c, { concern: c.concern || concern.has((c.en ?? c.text).toLowerCase()) }),
            ),
          ),
        )
      : null,
    mindCaveat(),
  );
}

// The words this case watches, named as the screen names them; a word this model cannot read as one token is not
// watched, and the case says so above.
function watchList(v) {
  const l = lang();
  const unread = new Set(v.unread_words ?? []);
  return (v.concern_words ?? [])
    .filter((word) => !unread.has(word))
    .map((word) => wordName(v.alerts.find((a) => a.word === word) ?? { word }, l))
    .join(l === "ja" ? "・" : ", ");
}
