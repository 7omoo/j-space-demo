// The expert view: every layer at one token, under the J-lens or the logit lens. Pick a token in the strip (or move
// with the arrow keys); the strata list the top three words at each layer, grouped into the layers that read, think
// and speak. Words the case watches are marked. It sits folded at the end of a page and is drawn when first opened.

import { h, icon } from "../core/dom.js";
import { wordKey } from "../core/format.js";
import { lang, t } from "../core/i18n.js";

// How a token reads in the strip: spaces kept, line breaks and empty pieces made visible.
export function tokenLabel(text) {
  if (text === "") return "·";
  return text.replace(/\n/g, "↵");
}

export function band(layer, [first, last]) {
  if (layer < first) return "read";
  return layer <= last ? "think" : "speak";
}

// A folded section holding the expert view; the strata are built the first time it opens.
export function expertDetails({ views, layers, model, position }) {
  if (!layers) return null;
  const body = h("div", { class: "expert-body" });
  const details = h(
    "details",
    { class: "expert" },
    h(
      "summary",
      {},
      h("span", { class: "expert-title" }, t("expert.title")),
      h("span", { class: "sub" }, t("expert.note")),
    ),
    body,
  );
  details.addEventListener("toggle", () => {
    if (!details.open || body.hasChildNodes()) return;
    body.append(...expertView({ views, layers, model, position }));
    // the chosen token may sit far down the strip, below the system prompt: bring it into the strip's view
    const strip = body.querySelector(".token-strip");
    const chosen = strip.querySelector(".tok-on");
    strip.scrollTop = Math.max(0, chosen.offsetTop - strip.clientHeight / 3);
  });
  return details;
}

function expertView({ views, layers, model, position }) {
  const concern = new Set((views.concern_words ?? []).map(wordKey));
  const danger = views.mode !== "raw"; // the explainer watches the riddle's answer, not a risk
  const start = Number.isInteger(position) && position < layers.tokens.length ? position : layers.n_prompt - 1;
  const state = { lens: layers.top.j ? "j" : "logit", position: start };

  const buttons = layers.tokens.map((token, i) =>
    h(
      "button",
      {
        type: "button",
        class: `tok tok-${token.k}`,
        "data-pos": String(i),
        tabindex: "-1",
        "aria-label": t("expert.token", { n: i, text: token.t.trim() || tokenLabel(token.t) }),
      },
      tokenLabel(token.t),
    ),
  );
  const strip = h("div", { class: "token-strip", role: "toolbar", "aria-label": t("expert.tokens") }, buttons);
  const caption = h("p", { class: "strata-caption", "aria-live": "polite" });
  const strata = h("ol", { class: "strata", reversed: true });
  const lensButtons = ["j", "logit"]
    .filter((lens) => layers.top[lens])
    .map((lens) => {
      const button = h("button", { type: "button", class: "seg", "aria-pressed": "false" }, t(`expert.lens.${lens}`));
      button.addEventListener("click", () => {
        state.lens = lens;
        draw();
      });
      return [lens, button];
    });

  function select(next, focus = false) {
    state.position = Math.max(0, Math.min(layers.tokens.length - 1, next));
    draw();
    if (focus) buttons[state.position].focus();
  }

  strip.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-pos]");
    if (button) select(Number(button.dataset.pos));
  });
  strip.addEventListener("keydown", (event) => {
    const step = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[event.key];
    if (step) select(state.position + step, true);
    else if (event.key === "Home") select(0, true);
    else if (event.key === "End") select(layers.tokens.length - 1, true);
    else return;
    event.preventDefault();
  });

  function draw() {
    buttons.forEach((button, i) => {
      const on = i === state.position;
      button.classList.toggle("tok-on", on);
      button.setAttribute("tabindex", on ? "0" : "-1");
      button.setAttribute("aria-pressed", on ? "true" : "false");
    });
    for (const [lens, button] of lensButtons)
      button.setAttribute("aria-pressed", lens === state.lens ? "true" : "false");
    const token = layers.tokens[state.position];
    caption.textContent = t("expert.caption", {
      token: token.t.trim() || tokenLabel(token.t),
      lens: t(`expert.lens.${state.lens}`),
    });
    const grid = layers.top[state.lens][state.position];
    const rows = [];
    for (let layer = layers.n_layers - 1; layer >= 0; layer -= 1) {
      const zone = band(layer, layers.band);
      const firstOfZone = layer === layers.n_layers - 1 || band(layer + 1, layers.band) !== zone;
      rows.push(
        h(
          "li",
          { class: `stratum stratum-${zone}${firstOfZone ? " stratum-start" : ""}`, value: String(layer) },
          h("span", { class: "stratum-zone" }, firstOfZone ? t(`expert.band.${zone}`) : ""),
          h("span", { class: "stratum-layer" }, t("expert.layer", { n: layer })),
          h(
            "span",
            { class: "stratum-words" },
            grid[layer].map((index) => {
              const word = layers.words[index];
              const hit = concern.has(wordKey(word));
              return h(
                "span",
                { class: hit ? `sw sw-hit${danger ? "" : " sw-plain"}` : "sw", lang: "und" },
                hit && danger ? icon("alert", 12) : null,
                word.trim() || "·",
              );
            }),
          ),
        ),
      );
    }
    strata.replaceChildren(...rows);
  }

  draw();
  return [
    h("p", { class: "sub" }, t("expert.lead")),
    h(
      "div",
      { class: "expert-controls" },
      h(
        "div",
        { class: "expert-bar" },
        h("span", { class: "field-label" }, t("expert.lens")),
        h(
          "span",
          { class: "segmented", role: "group", "aria-label": t("expert.lens") },
          lensButtons.map(([, b]) => b),
        ),
      ),
      h("p", { class: "field-label" }, t("expert.pick")),
      strip,
    ),
    caption,
    strata,
    h("p", { class: "note" }, t("expert.band_note", { first: layers.band[0], last: layers.band[1], model })),
    lang() === "ja" ? h("p", { class: "note" }, t("expert.raw_words")) : null,
  ].filter(Boolean);
}
