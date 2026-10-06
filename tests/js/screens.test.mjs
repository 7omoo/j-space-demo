// The screens' pure helpers. Run with: node --test tests/js/*.test.mjs   (tests/unit/test_static.py runs it too)
import assert from "node:assert/strict";
import { test } from "node:test";

import { caseGrid, featuredCase, outcome, sycophancyCounts, VARIANTS } from "../../web/core/cases.js";
import {
  cleanReply,
  interpolate,
  rankPosition,
  rulerTicks,
  splitBySpan,
  stepStates,
  strength,
  tickText,
  truncate,
  vocabText,
  wordKey,
  wordName,
} from "../../web/core/format.js";
import { caseHref, mineHref, parseRoute, withModel } from "../../web/core/route.js";
import { band, tokenLabel } from "../../web/screens/expert.js";

test("strength uses the same bands as views.py", () => {
  assert.deepEqual([1, 10, 11, 100, 101, null].map(strength), ["strong", "strong", "weak", "weak", "faint", "none"]);
});

test("the ruler places a rank by its logarithm, from rank 1 to the whole vocabulary", () => {
  const vocab = 248320;
  assert.equal(rankPosition(1, vocab), 0);
  assert.equal(rankPosition(vocab, vocab), 100);
  assert.equal(rankPosition(10 * vocab, vocab), 100); // never past the end
  assert.ok(Math.abs(rankPosition(10, vocab) - 18.54) < 0.01); // the top ten: under a fifth of the ruler
  assert.ok(Math.abs(rankPosition(100, vocab) - 37.07) < 0.01);
  assert.equal(rankPosition(null, vocab), 0);
});

test("the ruler's marks: powers of ten, and the end; one too near the end loses its number", () => {
  const ticks = rulerTicks(248320);
  assert.deepEqual(
    ticks.map((tick) => tick.value),
    [1, 10, 100, 1000, 10000, 100000, 248320],
  );
  assert.deepEqual(
    ticks.map((tick) => tick.labelled),
    [true, true, true, true, true, false, true],
  );
  assert.deepEqual(
    ticks.map((tick) => tick.minor),
    [false, false, false, true, true, true, false],
  );
  assert.equal(ticks.at(-1).x, 100);
  assert.deepEqual(
    [1, 1000, 10000, 248320].map((value) => tickText(value, "ja")),
    ["1", "1,000", "1万", "24.8万"],
  );
  assert.deepEqual(
    [1000, 10000, 151936].map((value) => tickText(value, "en")),
    ["1,000", "10k", "152k"],
  );
});

test("interpolate fills known names and leaves the rest", () => {
  assert.equal(interpolate("{rank}位 / {vocab}語", { rank: 1, vocab: "24.8万" }), "1位 / 24.8万語");
  assert.equal(interpolate("{missing} stays", {}), "{missing} stays");
});

test("vocabulary sizes read naturally in both languages", () => {
  assert.equal(vocabText(248320, "ja"), "24.8万");
  assert.equal(vocabText(248320, "en"), "248k");
  assert.equal(vocabText(1500000, "ja"), "150万");
});

test("cleanReply turns the model's Markdown into plain text", () => {
  assert.equal(cleanReply("**Do not mix.**\n\n### 1. Why\n* one\n\n\n\nend"), "Do not mix.\n\n1. Why\n・one\n\nend");
  assert.equal(cleanReply(undefined), "");
});

test("wordName prefers the Japanese name on a Japanese screen", () => {
  assert.equal(wordName({ word: "toxic", ja: "有毒" }, "ja"), "有毒");
  assert.equal(wordName({ word: "toxic", ja: "有毒" }, "en"), "toxic");
  assert.equal(wordName({ text: "粉尘", en: "dust", ja: null }, "ja"), "dust");
});

test("splitBySpan marks the phrase being read", () => {
  assert.deepEqual(splitBySpan("abcdef", [2, 4]), ["ab", "cd", "ef"]);
  assert.deepEqual(splitBySpan("abc", null), ["abc", "", ""]);
});

test("truncate", () => {
  assert.equal(truncate("abcdef", 4), "abc…");
  assert.equal(truncate("abc", 4), "abc");
});

test("routes parse from the hash and back", () => {
  assert.deepEqual(parseRoute(""), { name: "home" });
  assert.deepEqual(parseRoute("#/"), { name: "home" });
  assert.deepEqual(parseRoute("#/cases"), { name: "cases" });
  assert.deepEqual(parseRoute("#/about"), { name: "about" });
  assert.deepEqual(parseRoute("#/try"), { name: "try" });
  assert.deepEqual(parseRoute("#/case/lead-paint-persona"), { name: "case", id: "lead-paint-persona" });
  assert.deepEqual(parseRoute("#/mine/20261006-120000-abcdef"), { name: "mine", id: "20261006-120000-abcdef" });
  assert.deepEqual(parseRoute("#/job/abc"), { name: "job", id: "abc" });
  assert.deepEqual(parseRoute("#/case"), { name: "notfound" });
  assert.deepEqual(parseRoute("#/sycophancy"), { name: "notfound" }); // a page of the earlier site
  assert.deepEqual(parseRoute("#/case/%E2%80%A6"), { name: "case", id: "…" });
  assert.equal(caseHref("lead-paint-persona"), "#/case/lead-paint-persona");
  assert.equal(mineHref("x y"), "#/mine/x%20y");
});

test("an earlier link to a tab of a case still opens the case", () => {
  assert.deepEqual(parseRoute("#/case/lead-paint-risky/honesty"), { name: "case", id: "lead-paint-risky" });
  assert.deepEqual(parseRoute("#/cases/pressure"), { name: "cases" });
});

test("a shared link can name the model", () => {
  assert.deepEqual(parseRoute("#/case/ceo-wire-persona?model=qwen3-8b"), {
    name: "case",
    id: "ceo-wire-persona",
    model: "qwen3-8b",
  });
  assert.deepEqual(parseRoute("#/cases?model=gemma-3-12b-it"), { name: "cases", model: "gemma-3-12b-it" });
  assert.deepEqual(parseRoute("#/cases?other=1"), { name: "cases" });
  assert.equal(withModel("#/case/x?model=qwen3-8b", "gemma-3-12b-it"), "#/case/x?model=gemma-3-12b-it");
  assert.equal(withModel("#/about", "qwen3-14b"), "#/about?model=qwen3-14b");
});

test("a job's steps: done before the current one, all done at the end", () => {
  const plan = ["translating", "generating", "reading", "translating_reply"];
  const states = (step, status) => stepStates(plan, step, status).map((s) => s.state);
  assert.deepEqual(states(null, "queued"), ["pending", "pending", "pending", "pending"]);
  assert.deepEqual(states("reading", "running"), ["done", "done", "active", "pending"]);
  assert.deepEqual(states(null, "done"), ["done", "done", "done", "done"]);
  assert.deepEqual(stepStates(["generating"], "generating", "running"), [{ name: "generating", state: "active" }]);
});

const row = (scenario, variant, cell) => ({
  id: `${scenario}-${variant}`,
  scenario,
  scenario_title: { ja: scenario, en: scenario },
  kind: variant === "risky" ? "risk" : "pressure",
  variant,
  honesty: cell ? { cell } : null,
});
const ROWS = [
  { id: "boot-riddle", scenario: "boot-riddle", kind: "explainer", variant: "raw", honesty: null },
  row("lead-paint", "risky", "honest"),
  row("lead-paint", "pushback", "sycophancy"),
  row("lead-paint", "persona", "sycophancy"),
  row("lead-paint", "both", "ok"),
  row("warfarin-aspirin", "risky", "honest"),
  row("warfarin-aspirin", "persona", "sycophancy"),
  row("warfarin-aspirin", "both", "honest"),
];

test("the home page leads with an agreeing reply, preferring the declared order", () => {
  assert.equal(featuredCase(ROWS).id, "warfarin-aspirin-persona");
  assert.equal(featuredCase(ROWS.filter((r) => !r.id.startsWith("warfarin"))).id, "lead-paint-persona");
  const neverAgreed = ROWS.map((r) => (r.honesty ? { ...r, honesty: { cell: "honest" } } : r));
  assert.equal(featuredCase(neverAgreed).id, "lead-paint-pushback");
  const emptyReply = neverAgreed.map((r) => (r.id === "lead-paint-pushback" ? { ...r, honesty: null } : r));
  assert.notEqual(featuredCase(emptyReply).id, "lead-paint-pushback"); // an empty reply has no stance to feature
  assert.ok(featuredCase(emptyReply).honesty);
  assert.equal(featuredCase([]), null);
});

test("the four cells of the measurement show as three results", () => {
  assert.deepEqual(["honest", "sycophancy", "caution", "ok"].map(outcome), ["honest", "sycophancy", "weak", "weak"]);
});

test("the home table has one row per scenario, its cases in column order", () => {
  const grid = caseGrid(ROWS);
  assert.deepEqual(
    grid.map((g) => g.scenario),
    ["lead-paint", "warfarin-aspirin"],
  ); // the explainer is not a row
  assert.deepEqual(VARIANTS, ["risky", "pushback", "persona", "both"]);
  assert.deepEqual(
    grid[0].cells.map((c) => c.id),
    ["lead-paint-risky", "lead-paint-pushback", "lead-paint-persona", "lead-paint-both"],
  );
  assert.deepEqual(
    grid[1].cells.map((c) => c?.id ?? null),
    ["warfarin-aspirin-risky", null, "warfarin-aspirin-persona", "warfarin-aspirin-both"],
  );
});

test("sycophancy is counted under pressure and without it, apart", () => {
  assert.deepEqual(sycophancyCounts(ROWS), {
    pressured: { n: 5, sycophancy: 3, weak: 1 }, // lead-paint-both: agreed with a weak concern
    plain: { n: 2, sycophancy: 0, weak: 0 },
  });
  const noReply = ROWS.map((r) => (r.id === "lead-paint-pushback" ? { ...r, honesty: null } : r));
  assert.deepEqual(sycophancyCounts(noReply).pressured, { n: 4, sycophancy: 2, weak: 1 }); // nothing to judge
});

test("the expert view's token labels and bands", () => {
  assert.equal(tokenLabel(" boot"), " boot");
  assert.equal(tokenLabel("\n"), "↵");
  assert.equal(tokenLabel(""), "·");
  assert.deepEqual(
    [0, 7, 8, 25, 26, 31].map((layer) => band(layer, [8, 25])),
    ["read", "read", "think", "think", "speak", "speak"],
  );
});

test("watched words match as written, in any case, singular or plural, but not in another language", () => {
  assert.equal(wordKey(" Toxic"), wordKey("toxic"));
  assert.equal(wordKey("euros"), wordKey("Euro"));
  assert.notEqual(wordKey("意大利"), wordKey("Italy"));
});
