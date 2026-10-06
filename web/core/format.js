// Pure helpers for what the screens print (tested by tests/js/screens.test.mjs; no DOM here).

export const STRONG = 10; // the same bands as jspace_demo.views.words
export const WEAK = 100;

// The key a word is matched to the watched words by, as the views mark them: as written, in any case, singular or
// plural ("Toxic", "euros"). An equivalent in another language (意大利 for Italy) is a different word.
export function wordKey(word) {
  return String(word).trim().toLowerCase().replace(/s$/, "");
}

export function interpolate(text, params = {}) {
  return String(text).replace(/\{(\w+)\}/g, (match, name) => (name in params ? String(params[name]) : match));
}

export function strength(rank) {
  if (rank === null || rank === undefined) return "none";
  if (rank <= STRONG) return "strong";
  return rank <= WEAK ? "weak" : "faint";
}

// 248,320 -> "24.8万" (ja) / "248k" (en)
export function vocabText(n, lang) {
  if (lang === "ja") {
    const man = n / 10000;
    return `${man >= 100 ? Math.round(man) : Math.round(man * 10) / 10}万`;
  }
  return n >= 1000 ? `${Math.round(n / 1000)}k` : String(n);
}

// Where a rank sits on the ruler of the whole vocabulary, in percent: rank 1 at 0, the last word at 100, by its
// logarithm, so the top ten and the top hundred have room to be seen.
export function rankPosition(rank, vocabSize) {
  if (!(rank >= 1) || !(vocabSize > 1)) return 0;
  return Math.min(100, (Math.log10(rank) / Math.log10(vocabSize)) * 100);
}

// The ruler's marks: every power of ten below the vocabulary's size, then its end. A mark too near the end for its
// number to fit (within a tenth of the ruler) is drawn without it; a minor mark (1,000 and up, short of the end)
// loses its number where the ruler is narrow.
export function rulerTicks(vocabSize) {
  const ticks = [];
  for (let value = 1; value < vocabSize; value *= 10) {
    const x = rankPosition(value, vocabSize);
    ticks.push({ value, x, labelled: 100 - x >= 10, minor: value >= 1000 });
  }
  ticks.push({ value: vocabSize, x: 100, labelled: true, minor: false });
  return ticks;
}

// A mark's number: 1,000 as it is; ten thousand and up as the vocabulary's size reads (1万 / 10k).
export function tickText(value, lang) {
  return value >= 10000 ? vocabText(value, lang) : value.toLocaleString("en-GB");
}

// The model writes Markdown; the screens show plain text.
export function cleanReply(text) {
  return String(text ?? "")
    .replace(/\*\*|__/g, "")
    .replace(/^#{1,6}\s*/gm, "")
    .replace(/^\s*[*-]\s+/gm, "・")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

// A readout word as a screen names it: its Japanese name on a Japanese screen when the glossary has one.
export function wordName(item, lang) {
  if (!item) return "";
  const english = item.en ?? item.word ?? item.text ?? "";
  return lang === "ja" ? (item.ja ?? english) : english;
}

export function splitBySpan(text, span) {
  if (!span) return [text, "", ""];
  const [start, end] = span;
  return [text.slice(0, start), text.slice(start, end), text.slice(end)];
}

export function truncate(text, n) {
  const s = String(text ?? "");
  return s.length > n ? `${s.slice(0, n - 1)}…` : s;
}

// A job's planned steps with their state: the ones before the current step are done, all of them once the job is.
export function stepStates(plan, step, status) {
  const at = plan.indexOf(step);
  return plan.map((name, i) => ({
    name,
    state: status === "done" || (at >= 0 && i < at) ? "done" : i === at ? "active" : "pending",
  }));
}
