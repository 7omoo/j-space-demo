// Light or dark: the reader's choice, kept in localStorage, else the system's. index.html sets it before the first
// paint; this switches it, and follows the system while the reader has not chosen.

const THEME_KEY = "jspace.theme";

function stored() {
  try {
    const saved = localStorage.getItem(THEME_KEY);
    return saved === "dark" || saved === "light" ? saved : null;
  } catch {
    return null; // storage blocked: the system decides
  }
}

export function theme() {
  return document.documentElement.dataset.theme === "dark" ? "dark" : "light";
}

export function setTheme(next) {
  document.documentElement.dataset.theme = next;
  try {
    localStorage.setItem(THEME_KEY, next);
  } catch {
    // not remembered this time; the page still switches
  }
}

export function followSystem(onChange) {
  const query = matchMedia("(prefers-color-scheme: dark)");
  query.addEventListener("change", (event) => {
    if (stored()) return;
    document.documentElement.dataset.theme = event.matches ? "dark" : "light";
    onChange();
  });
}
