// Japanese and English text for the page (web/i18n/*.json; tests/unit/test_i18n.py keeps the two tables in step).

import { interpolate } from "./format.js";

const LANG_KEY = "jspace.lang";
const tables = {};
let current = "ja";

export async function loadI18n(getJSON) {
  const [ja, en] = await Promise.all([getJSON("i18n/ja.json"), getJSON("i18n/en.json")]);
  Object.assign(tables, { ja, en });
  setLang(initialLang(), { remember: false });
}

function initialLang() {
  try {
    const saved = localStorage.getItem(LANG_KEY);
    if (saved === "ja" || saved === "en") return saved;
  } catch {
    // storage blocked: fall back to the browser language
  }
  return (navigator.language || "").toLowerCase().startsWith("ja") ? "ja" : "en";
}

export function lang() {
  return current;
}

export function setLang(next, { remember = true } = {}) {
  current = next === "en" ? "en" : "ja";
  document.documentElement.lang = current;
  if (!remember) return;
  try {
    localStorage.setItem(LANG_KEY, current);
  } catch {
    // not remembered this time; the page still switches
  }
}

// A missing key shows as the key itself and is reported, so the browser tests (no console errors) catch it.
export function t(key, params = {}) {
  const text = tables[current]?.[key] ?? tables.en?.[key];
  if (text === undefined) console.error(`missing text: ${key}`);
  return interpolate(text ?? key, params);
}

// A {ja, en} pair in the current language, falling back to the other one.
export function pick(value) {
  if (value === null || value === undefined) return "";
  if (typeof value === "string") return value;
  return value[current] ?? value.en ?? value.ja ?? "";
}
