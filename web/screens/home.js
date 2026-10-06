// Home: the claim, one real case set against what the model had in mind, then every case in one table (also reached
// as #/cases). One question throughout: did the reply say what the model had in mind, or what the user wanted?

import { caseGrid, featuredCase, outcome, sycophancyCounts, VARIANTS } from "../core/cases.js";
import { h } from "../core/dom.js";
import { cleanReply, truncate } from "../core/format.js";
import { lang, pick, t } from "../core/i18n.js";
import { caseHref } from "../core/route.js";
import {
  conditionList,
  conditionTag,
  dots,
  link,
  mindCaveat,
  outcomeLabel,
  pageHead,
  quoted,
  quotedNodes,
  rankRuler,
  row,
} from "./parts.js";

// The claim, beside what the numbers rest on: how many models, which messages, one run each.
function hero(data, rows) {
  return pageHead({
    title: t("home.title"),
    lead: t("home.lead"),
    meta: [
      t("home.meta.models", { n: data.models.length }),
      t("home.meta.cases", { messages: caseGrid(rows).length, ways: VARIANTS.length }),
      t("home.meta.runs"),
    ],
    cls: "hero",
  });
}

// The models as buttons, each with how often its replies were sycophantic under pressure (said once, beside them,
// and in the group's name).
function modelSwitch(data, counts, onChange) {
  if (data.models.length < 2) return null;
  return row(
    {
      label: t("model.label"),
      sub: t("home.model.sub"),
      labelId: "model-switch-label",
      subId: "model-switch-sub",
      cls: "model-switch",
    },
    h(
      "div",
      { class: "model-buttons", role: "group", "aria-labelledby": "model-switch-label model-switch-sub" },
      data.models.map((m) =>
        h(
          "button",
          {
            type: "button",
            class: "model-button",
            "aria-pressed": m.key === data.model ? "true" : "false",
            "data-focus": `model-${m.key}`,
            onclick: () => {
              data.setModel(m.key);
              onChange();
            },
          },
          h("span", { class: "model-name" }, m.label),
          h(
            "span",
            { class: "model-count" },
            t("home.model.count", { n: counts[m.key].pressured.sycophancy, of: counts[m.key].pressured.n }),
          ),
        ),
      ),
    ),
  );
}

// The first paragraph of the reply, in the screen's language.
function opening(views, texts) {
  const ja = lang() === "ja" && (texts.reply_ja ?? views.translation?.reply_ja);
  return truncate(cleanReply(ja || views.reply).split("\n")[0], 200);
}

// One real case, row by row: the message, what the reply said, what the model had in mind, and the result.
function duel(featured, built, model) {
  const { views: v, texts } = built;
  const l = lang();
  const message = l === "ja" && texts.message_ja ? texts.message_ja : texts.message;
  const pushback = l === "ja" ? texts.pushback_ja : texts.pushback;
  const result = outcome(v.honesty.cell);
  return h(
    "section",
    { class: "duel", "aria-labelledby": "duel-title" },
    row(
      { label: t("home.example"), sub: model.label, labelId: "duel-title", cls: "duel-case" },
      h(
        "p",
        { class: "duel-head" },
        h("span", { class: "duel-scenario" }, pick(featured.scenario_title)),
        conditionTag(featured.variant),
      ),
      // the pressure sentence was part of the message the model read: inside the quote, set apart
      h(
        "p",
        { class: "duel-message" },
        quotedNodes(message, pushback ? h("span", { class: "pushback" }, " ", pushback) : null),
      ),
    ),
    row(
      { label: t("home.says"), sub: t("home.says.sub"), level: "h3", cls: "duel-says" },
      h("p", { class: "duel-quote" }, quoted(opening(v, texts))),
      h("p", { class: "duel-verdict" }, t(`case.stance.${v.honesty.stance}`)),
    ),
    row(
      { label: t("home.mind"), sub: t("home.mind.sub"), level: "h3", cls: "duel-mind" },
      rankRuler((v.watched ?? []).slice(0, 3), v.vocab_size),
      h("p", { class: "note" }, t("ruler.note")),
      mindCaveat(),
    ),
    row(
      { label: t("home.outcome"), level: "h3", cls: "duel-result" },
      h("p", { class: "duel-outcome" }, outcomeLabel(v.honesty.cell), h("span", {}, t(`home.result.${result}`))),
      link(caseHref(featured.id), { class: "text-link" }, t("home.example.more")),
    ),
  );
}

function caseTable(rows) {
  return h(
    "table",
    { class: "case-grid" },
    h(
      "thead",
      {},
      h(
        "tr",
        {},
        h("th", { scope: "col" }, t("home.grid.case")),
        VARIANTS.map((variant) => h("th", { scope: "col" }, t(`condition.short.${variant}`))),
      ),
    ),
    h(
      "tbody",
      {},
      caseGrid(rows).map((r) =>
        h(
          "tr",
          {},
          h("th", { scope: "row" }, pick(r.title)),
          r.cells.map((c, i) =>
            h(
              "td",
              {},
              c?.honesty
                ? link(
                    caseHref(c.id),
                    {
                      class: "grid-cell",
                      "aria-label": t("home.grid.cell", {
                        case: pick(r.title),
                        condition: t(`condition.short.${VARIANTS[i]}`),
                        result: t(`result.${outcome(c.honesty.cell)}`),
                      }),
                    },
                    outcomeLabel(c.honesty.cell, { short: true }),
                  )
                : h("span", { class: "sub" }, "—"),
            ),
          ),
        ),
      ),
    ),
  );
}

function legend() {
  return h(
    "ul",
    { class: "result-legend" },
    ["sycophancy", "honest", "weak"].map((result) =>
      h("li", {}, outcomeLabel(result, { short: true }), h("span", {}, t(`result.about.${result}`))),
    ),
  );
}

function gridSection(rows, model, counts) {
  const weak = counts.pressured.weak + counts.plain.weak;
  return h(
    "section",
    { id: "grid", class: "grid-section", "aria-labelledby": "grid-title" },
    row(
      {
        label: [h("span", {}, t("home.grid.title")), " ", h("span", { class: "row-sub" }, model.label)],
        labelId: "grid-title",
        focusable: true,
        cls: "grid-row",
      },
      h(
        "p",
        { class: "grid-summary" },
        t("home.grid.summary", {
          n: counts.pressured.n,
          syc: counts.pressured.sycophancy,
          plain_n: counts.plain.n,
          plain_syc: counts.plain.sycophancy,
        }),
        weak ? [" ", t("home.grid.weak", { n: weak })] : null,
      ),
      h("p", { class: "note" }, t("home.grid.open")),
      h("div", { class: "table-scroll" }, caseTable(rows)),
    ),
    row({ label: t("home.legend") }, legend(), h("p", { class: "note" }, t("home.data", { model: model.label }))),
    row({ label: t("home.conditions") }, conditionList()),
  );
}

export async function home(route, app) {
  const { data } = app;
  const lists = await Promise.all(data.models.map((m) => data.cases(m.key)));
  const counts = Object.fromEntries(data.models.map((m, i) => [m.key, sycophancyCounts(lists[i])]));
  const rows = await data.cases();
  const featured = featuredCase(rows);
  const built = featured ? await data.case(featured.id) : null;
  const model = data.modelInfo();
  return h(
    "div",
    { class: "page home" },
    hero(data, rows),
    modelSwitch(data, counts, app.rerender),
    built ? duel(featured, built, model) : null,
    dots(),
    gridSection(rows, model, counts[data.model]),
  );
}
