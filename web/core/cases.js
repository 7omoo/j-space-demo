// Pure helpers over the rows of a model's case list (data/<model>/index.json). Tested by tests/js/screens.test.mjs.

// How a message was asked, in the order of the home table's columns: as it is, then under the three pressures.
export const VARIANTS = ["risky", "pushback", "persona", "both"];

// The result a screen shows for a reply. The data keeps the measurement's four cells; when the concern was weak the
// model did not hold the danger strongly, so its reply is neither honest nor sycophantic: both weak cells read "weak".
export function outcome(cell) {
  return cell === "honest" || cell === "sycophancy" ? cell : "weak";
}

// The example the home page leads with: a reply that agrees while a watch word tops the readout. The first of
// these in the sycophancy cell; any other pressure case with a reply when a model never agreed. The order prefers
// replies whose agreement is plain on a full reading, not only by the stance rule ("double-check with your manager"
// is agreement to the rule, but a hedge to a reader).
const FEATURED = [
  "warfarin-aspirin-persona",
  "warfarin-aspirin-both",
  "crypto-loan-both",
  "lottery-scam-persona",
  "ceo-wire-both",
  "storm-hike-both",
  "lead-paint-persona",
  "lead-paint-both",
];

export function featuredCase(rows) {
  const pressure = rows.filter((r) => r.kind === "pressure");
  const agreeing = pressure.filter((r) => r.honesty?.cell === "sycophancy");
  const fallback = pressure.find((r) => r.honesty); // an empty reply has no stance to set against the readout
  return FEATURED.map((id) => agreeing.find((r) => r.id === id)).find(Boolean) ?? agreeing[0] ?? fallback ?? null;
}

// The home table: one row per scenario, in list order, its cases in VARIANTS order (null where one is missing).
export function caseGrid(rows) {
  const scenarios = new Map();
  for (const row of rows) {
    if (!VARIANTS.includes(row.variant)) continue; // the explainer is not a consultation
    if (!scenarios.has(row.scenario))
      scenarios.set(row.scenario, { scenario: row.scenario, title: row.scenario_title });
    scenarios.get(row.scenario)[row.variant] = row;
  }
  return [...scenarios.values()].map((s) => ({
    scenario: s.scenario,
    title: s.title,
    cells: VARIANTS.map((v) => s[v] ?? null),
  }));
}

// How many replies were sycophantic, and how many were not judged (a weak concern): under the three pressures, and
// asked as it is.
export function sycophancyCounts(rows) {
  const count = (found) => ({
    n: found.length,
    sycophancy: found.filter((r) => r.honesty.cell === "sycophancy").length,
    weak: found.filter((r) => outcome(r.honesty.cell) === "weak").length,
  });
  const judged = rows.filter((r) => r.honesty);
  return {
    pressured: count(judged.filter((r) => r.kind === "pressure")),
    plain: count(judged.filter((r) => r.variant === "risky")),
  };
}
