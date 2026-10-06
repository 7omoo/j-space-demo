// Try your own text (the local app only): the form, the model's state when it is not ready yet, the analyses made on
// this machine, and the screen of a running analysis. The static site points to the local version instead.

import { h, icon } from "../core/dom.js";
import { stepStates, truncate, wordName } from "../core/format.js";
import { lang, t } from "../core/i18n.js";
import { mineHref } from "../core/route.js";
import { conditionTag, link, outcomeLabel, pageHead, row } from "./parts.js";

const SET_KEY = "jspace.watch_set";
const EXAMPLES = ["1", "2", "3"];
export const REPOSITORY = "https://github.com/7omoo/j-space-demo";

export function preferredSet(sets) {
  try {
    const saved = localStorage.getItem(SET_KEY);
    if (saved && sets.includes(saved)) return saved;
  } catch {
    // storage blocked: use the default
  }
  return sets.includes("general") ? "general" : sets[0];
}

export function rememberSet(name) {
  try {
    localStorage.setItem(SET_KEY, name);
  } catch {
    // not remembered; the analysis still runs
  }
}

// A job's error in the screen's words: an input the user can fix has its own code.
export function errorText(error) {
  if (!error) return "";
  if (typeof error === "string") return t("error.internal", { message: error });
  return error.code === "internal"
    ? t("error.internal", { message: error.message })
    : t(`error.${error.code}`, error.params);
}

export function runLocally() {
  return h(
    "pre",
    { class: "code" },
    `git clone ${REPOSITORY}\ncd j-space-demo\nscripts/bootstrap.sh\nuv run jspace-demo serve`,
  );
}

export function staticNotice() {
  return h(
    "div",
    { class: "page try" },
    pageHead({ title: t("try.title"), lead: t("try.static") }),
    row({ label: t("about.local.title") }, h("p", {}, t("try.static.how")), runLocally()),
  );
}

// The model is off, loading or failed: say so, and how to start the app with one. The analyses kept here stay
// listed. While the model loads, the screen looks again.
async function notReady(data, meta) {
  const loading = meta.status === "loading" || meta.status === "idle";
  const node = h(
    "div",
    { class: "page try" },
    pageHead({ title: t("try.title"), lead: t(`status.about.${meta.status}`) }),
    meta.status === "error" && meta.error
      ? row({ label: t("try.error") }, h("p", { class: "callout callout-warn" }, meta.error))
      : null,
    loading
      ? null
      : row(
          { label: t("try.start") },
          h("pre", { class: "code" }, "uv run jspace-demo serve"),
          h("p", { class: "note" }, t("try.start.note")),
          h("pre", { class: "code" }, "uv run jspace-demo serve --model qwen3-8b"),
        ),
    await analysesRow(data),
  );
  if (loading) {
    setTimeout(() => {
      if (node.isConnected) window.dispatchEvent(new HashChangeEvent("hashchange"));
    }, 2000);
  }
  return node;
}

function radios(name, values, chosen, label) {
  return values.map((value) =>
    h("label", { class: "radio" }, h("input", { type: "radio", name, value, checked: value === chosen }), label(value)),
  );
}

function form(data, meta) {
  const sets = Object.keys(meta.watch_sets);
  const message = h("textarea", {
    id: "message",
    name: "message",
    rows: 5,
    required: true,
    maxlength: 4000,
    placeholder: t("try.message.placeholder"),
  });
  const status = h("p", { class: "form-status", role: "status", "aria-live": "polite" });
  const submit = h("button", { type: "submit", class: "button primary" }, t("try.submit"), icon("arrowRight"));
  const node = h(
    "form",
    { class: "try-form" },
    h("label", { for: "message", class: "field-label" }, t("try.message")),
    message,
    h(
      "p",
      { class: "examples" },
      h("span", { class: "sub" }, t("try.examples")),
      EXAMPLES.map((n) =>
        h(
          "button",
          {
            type: "button",
            class: "chip-button",
            onclick: () => {
              message.value = t(`try.example.${n}.text`);
              message.focus();
            },
          },
          t(`try.example.${n}.label`),
        ),
      ),
    ),
    h(
      "fieldset",
      { class: "watch-sets" },
      h("legend", { class: "field-label" }, t("try.watch_set")),
      radios("watch_set", sets, preferredSet(sets), (name) =>
        h(
          "span",
          { class: "radio-text" },
          h("b", {}, t(`watchset.${name}`)),
          h("span", { class: "sub", lang: "en" }, `${meta.watch_sets[name].slice(0, 5).join(", ")}, …`),
        ),
      ),
    ),
    h(
      "fieldset",
      { class: "conditions" },
      h("legend", { class: "field-label" }, t("try.condition")),
      radios("condition", ["", ...(meta.conditions ?? [])], "", (value) =>
        // "": asked as it is; the pressures come from the server
        h(
          "span",
          { class: "radio-text" },
          h("b", {}, t(`condition.short.${value || "risky"}`)),
          h("span", { class: "sub" }, t(`condition.about.${value || "risky"}`)),
        ),
      ),
    ),
    h("div", { class: "form-actions" }, submit, status),
  );
  node.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!message.value.trim()) {
      status.textContent = t("error.empty"); // only spaces: the browser's own check lets it through
      message.focus();
      return;
    }
    const fields = new FormData(node);
    rememberSet(fields.get("watch_set"));
    submit.disabled = true;
    status.textContent = t("try.sending");
    try {
      const job = await data.submitJob({
        message: message.value,
        watch_set: fields.get("watch_set"),
        condition: fields.get("condition") || null,
        reply_lang: lang(),
      });
      location.hash = `#/job/${job.id}`;
    } catch (error) {
      const notReadyNow = error.status === 503 && error.detail?.status;
      status.textContent = notReadyNow
        ? t(`status.about.${error.detail.status}`)
        : t("try.failed", { message: error.message });
      submit.disabled = false;
    }
  });
  return node;
}

function formatDate(iso) {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  return date.toLocaleString(lang() === "ja" ? "ja-JP" : "en-GB", { dateStyle: "medium", timeStyle: "short" });
}

function bestLine(row) {
  if (!row.best) return t("mine.no_words");
  return t("mine.best", { word: wordName(row.best, lang()), rank: row.best.rank });
}

// The analyses kept on this machine, newest first, each with a button that deletes it.
async function analysesRow(data) {
  const rows = await data.analyses();
  const empty = () => h("p", { class: "sub" }, t("mine.empty"));
  const body = rows.length ? h("ul", { class: "mine-list" }) : empty();
  for (const row of rows) {
    const remove = h("button", { type: "button", class: "icon-button", "aria-label": t("mine.delete") }, icon("trash"));
    const item = h(
      "li",
      { class: "mine-item" },
      link(
        mineHref(row.id),
        { class: "mine-row" },
        h(
          "span",
          { class: "mine-meta" },
          row.cell ? outcomeLabel(row.cell) : null,
          conditionTag(row.condition),
          h("span", { class: "sub" }, t(`watchset.${row.watch_set}`)),
          row.translated ? h("span", { class: "tag" }, t("mine.translated")) : null,
          h("time", { datetime: row.created }, formatDate(row.created)),
        ),
        h("span", { class: "mine-title" }, truncate(row.title, 90)),
        h("span", { class: "sub" }, bestLine(row)),
      ),
      remove,
    );
    remove.addEventListener("click", async () => {
      remove.disabled = true;
      try {
        await data.deleteAnalysis(row.id);
      } catch (error) {
        remove.disabled = false;
        remove.title = error.message;
        return;
      }
      const next = item.nextElementSibling ?? item.previousElementSibling;
      item.remove();
      if (next)
        next.querySelector(".mine-row").focus(); // keep the keyboard in the list
      else body.replaceWith(empty());
    });
    body.append(item);
  }
  return row({ label: t("mine.title"), sub: t("mine.note"), cls: "mine" }, body);
}

export async function tryScreen(route, app) {
  const { data } = app;
  if (!data.live) return staticNotice();
  const meta = await data.refreshMeta();
  if (meta.status !== "ready") return notReady(data, meta);
  const model = meta.model?.label ?? "";
  return h(
    "div",
    { class: "page try" },
    pageHead({ title: t("try.title"), lead: t("try.lead", { model }), meta: [t("case.model", { model })] }),
    row({ label: t("try.form") }, form(data, meta), h("p", { class: "note" }, t("try.note"))),
    await analysesRow(data),
  );
}

export async function jobScreen(route, app) {
  const { data } = app;
  const list = h("ol", { class: "steps" });
  const status = h("p", { class: "job-status", role: "status", "aria-live": "polite" }, t("job.waiting"));
  const actions = h("p", { class: "actions" });
  const cancel = h("button", { type: "button", class: "button", hidden: true }, t("job.cancel"));
  cancel.addEventListener("click", async () => {
    cancel.disabled = true;
    await data.cancelJob(route.id).catch(() => null);
  });
  actions.append(cancel);
  const node = h(
    "div",
    { class: "page job" },
    pageHead({ title: t("job.title"), lead: t("job.lead") }),
    row({ label: t("job.progress") }, list, status, actions),
  );
  const draw = (job) => {
    const plan = job.plan?.length ? job.plan : ["generating", "reading"];
    list.replaceChildren(
      ...stepStates(plan, job.step, job.status).map(({ name, state }) =>
        h(
          "li",
          {
            class: `step-item step-${state}`,
            "aria-current": state === "active" ? "step" : null,
          },
          icon(state === "done" ? "check" : "arrowRight", 14),
          t(`job.step.${name}`),
        ),
      ),
    );
  };
  const finish = (text) => {
    status.textContent = text;
    actions.replaceChildren(link("#/try", { class: "button" }, t("job.retry")));
  };
  const started = Date.now();
  const tick = async () => {
    if (!node.isConnected && Date.now() - started > 2000) return; // the user moved on
    try {
      const job = await data.job(route.id);
      draw(job);
      if (job.status === "done") {
        location.replace(mineHref(job.result)); // the finished job's page is replaced, so Back skips it
        return;
      }
      if (job.status === "error") return finish(errorText(job.error));
      if (job.status === "cancelled") return finish(t("job.cancelled"));
      cancel.hidden = job.status !== "queued";
      status.textContent =
        job.status === "queued"
          ? t("job.queued", { ahead: job.ahead })
          : t("job.running", { seconds: Math.max(0, Math.round(Date.now() / 1000 - job.started)) });
    } catch (error) {
      if (error.status === 404) return finish(t("job.lost"));
      status.textContent = error.message;
    }
    setTimeout(tick, 1000);
  };
  setTimeout(tick, 0);
  return node;
}
