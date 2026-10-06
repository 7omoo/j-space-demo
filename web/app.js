// The page: the header (brand, screens, model status, language, theme), the footer, and the screen the URL hash names.

import { createData, getJSON } from "./core/data.js";
import { clear, h, icon } from "./core/dom.js";
import { lang, loadI18n, setLang, t } from "./core/i18n.js";
import { parseRoute, withModel } from "./core/route.js";
import { followSystem, setTheme, theme } from "./core/theme.js";
import { about, notFound } from "./screens/about.js";
import { caseScreen, mineScreen } from "./screens/case.js";
import { home } from "./screens/home.js";
import { external } from "./screens/parts.js";
import { jobScreen, REPOSITORY, tryScreen } from "./screens/try.js";

const SCREENS = {
  home,
  cases: home, // the home page, scrolled to its table of cases
  case: caseScreen,
  mine: mineScreen,
  about,
  try: tryScreen,
  job: jobScreen,
  notfound: notFound,
};

// Header links: [route names that mark it current, href, label key, local app only]. The brand links home.
const NAV = [
  [["cases", "case"], "#/cases", "nav.cases", false],
  [["about"], "#/about", "nav.about", false],
  [["try", "job", "mine"], "#/try", "nav.try", true],
];

const app = { data: null, rerender };

let generation = 0;
let shownHash = null; // the address last drawn: drawing it again (language, model, a refresh) keeps the scroll

// After the model picker: an address that names a model (a ?model= link) would bring the old one back on the next
// render, so it is pointed at the chosen model first, without adding a history entry. It is the same screen drawn
// again, not a move to another one.
function rerender() {
  const named = parseRoute(location.hash).model;
  if (named && named !== app.data.model) {
    history.replaceState(history.state, "", withModel(location.hash || "#/", app.data.model));
    shownHash = location.hash;
  }
  render();
}

// Where a screen opens: back where the reader left it (Back and Forward), at the home page's table (#/cases), or at
// its top. The page keeps each history entry's scroll itself, because drawing a screen replaces the whole page.
history.scrollRestoration = "manual";
let scrollSave;
let drawing = false; // while a screen is drawn, the page's height changes under the reader: that is not their scroll
window.addEventListener(
  "scroll",
  () => {
    if (drawing) return;
    clearTimeout(scrollSave);
    scrollSave = setTimeout(() => history.replaceState({ ...history.state, scrollY: window.scrollY }, ""), 150);
  },
  { passive: true },
);

function arrive(route) {
  const view = document.getElementById("view");
  const saved = history.state?.scrollY;
  const table = route.name === "cases" ? document.getElementById("grid-title") : null;
  if (Number.isFinite(saved)) window.scrollTo(0, saved);
  else if (table) table.scrollIntoView();
  else window.scrollTo(0, 0);
  (table ?? view.querySelector("h1"))?.focus({ preventScroll: true });
}

// A control drawn again keeps the keyboard: the language switch, a model button, the model picker.
function focusKey() {
  return document.activeElement?.dataset?.focus ?? null;
}

function refocus(key) {
  if (key) document.querySelector(`[data-focus="${CSS.escape(key)}"]`)?.focus({ preventScroll: true });
}

function statusPill() {
  const state = app.data.meta.status;
  return h(
    "a",
    { href: "#/try", class: `status status-${state}`, title: t("status.title"), "aria-label": t(`status.${state}`) },
    h("span", { class: "status-dot", "aria-hidden": "true" }),
    h("span", { class: "status-text" }, t(`status.${state}`)), // on a phone the dot alone, for room
  );
}

// The model's state, the language switch and the theme switch. Drawn again after each screen, which may have
// refreshed the state.
function drawTools() {
  clear(document.getElementById("tools")).append(
    ...[app.data.live ? statusPill() : null, langToggle(), themeToggle()].filter(Boolean),
  );
}

function langToggle() {
  const other = lang() === "ja" ? "en" : "ja";
  return h(
    "button",
    {
      type: "button",
      class: "lang-toggle",
      lang: other,
      "data-focus": "lang",
      "aria-label": t("lang.switch.label"),
      onclick: () => {
        setLang(other);
        render();
      },
    },
    t("lang.switch"),
  );
}

// Light or dark: only the colours change, so the screen is not drawn again; the button is, to name the other theme.
function themeToggle() {
  const other = theme() === "dark" ? "light" : "dark";
  return h(
    "button",
    {
      type: "button",
      class: "theme-toggle",
      "data-focus": "theme",
      "aria-label": t(`theme.to.${other}`),
      title: t(`theme.to.${other}`),
      onclick: () => {
        setTheme(theme() === "dark" ? "light" : "dark"); // as it is now: the system may have changed it
        drawTools();
        refocus("theme");
      },
    },
    icon(other === "dark" ? "moon" : "sun", 17),
  );
}

// A header link to the screen already shown changes no address, so it takes the reader to where that screen opens.
function navLink([names, href, key], route) {
  return h(
    "a",
    {
      href,
      class: "nav-link",
      "aria-current": names.includes(route.name) ? "page" : null,
      onclick: (event) => {
        if (location.hash !== href) return;
        event.preventDefault();
        history.replaceState({ ...history.state, scrollY: null }, "");
        arrive(route);
      },
    },
    t(key),
  );
}

function chrome(route) {
  const nav = document.getElementById("nav");
  clear(nav).append(...NAV.filter(([, , , local]) => !local || app.data.live).map((item) => navLink(item, route)));
  drawTools();
  const brand = document.getElementById("brand");
  brand.textContent = t("app.title");
  if (route.name === "home") brand.setAttribute("aria-current", "page");
  else brand.removeAttribute("aria-current");
  document.getElementById("footer-note").textContent = t("footer.note");
  clear(document.getElementById("footer-links")).append(
    external(REPOSITORY, t("footer.code")),
    external(`${REPOSITORY}/blob/main/docs/report.md`, t("footer.report")),
  );
}

async function render() {
  const mine = ++generation;
  const focused = focusKey();
  clearTimeout(scrollSave); // a save still pending belongs to the screen being left
  drawing = true;
  const route = parseRoute(location.hash);
  if (route.model) app.data.setModel(route.model);
  chrome(route);
  const view = document.getElementById("view");
  const moved = location.hash !== shownHash;
  if (moved) clear(view).append(h("p", { class: "loading", role: "status" }, t("common.loading")));
  let node;
  try {
    node = await SCREENS[route.name](route, app);
  } catch (error) {
    node = h(
      "div",
      { class: "page" },
      h(
        "div",
        { class: "empty" },
        h("p", {}, error.status === 404 ? t("common.not_found") : t("common.error", { message: error.message })),
        h("a", { href: "#/", class: "button" }, t("nav.home")),
      ),
    );
  }
  if (mine !== generation) return; // a newer navigation won
  clear(view).append(node); // one step, so a screen drawn again keeps its scroll
  drawTools();
  const heading = view.querySelector("h1");
  document.title = heading ? `${heading.textContent} | ${t("app.title")}` : t("app.title");
  shownHash = location.hash;
  if (moved) arrive(route);
  else refocus(focused);
  drawing = false;
}

async function main() {
  followSystem(() => {
    if (!app.data) return; // the header is drawn once the data has loaded
    const focused = focusKey();
    drawTools();
    refocus(focused);
  });
  await loadI18n(getJSON);
  try {
    app.data = await createData();
  } catch {
    clear(document.getElementById("view")).append(h("p", { class: "empty" }, t("common.no_data")));
    return;
  }
  window.addEventListener("hashchange", render);
  // The skip link moves the keyboard to the screen. Followed as a link, "#view" would replace the route.
  document.querySelector(".skip").addEventListener("click", (event) => {
    event.preventDefault();
    document.getElementById("view").focus();
  });
  render();
}

main();
