// Screens are addressed by the URL hash (#/case/lead-paint-persona). No server routing is needed, so the static site
// works under any sub-path (7omo.com/j-space/). A trailing ?model=<key> picks the model of a shared link.
// Pure: tested by tests/js/screens.test.mjs.

const SIMPLE = ["cases", "about", "try"]; // "cases" is the home page, scrolled to its table
const WITH_ID = ["case", "mine", "job"];

export function parseRoute(hash) {
  const [path, query = ""] = String(hash || "")
    .replace(/^#\/?/, "")
    .split("?");
  const parts = path
    .split("/")
    .filter(Boolean)
    .map((part) => {
      try {
        return decodeURIComponent(part);
      } catch {
        return part;
      }
    });
  const route = routeOf(parts);
  const model = new URLSearchParams(query).get("model");
  return model ? { ...route, model } : route;
}

// Anything after the id (an old tab name such as /honesty) is ignored, so earlier links still open the case.
function routeOf([head, id]) {
  if (head === undefined) return { name: "home" };
  if (SIMPLE.includes(head)) return { name: head };
  if (WITH_ID.includes(head)) return id ? { name: head, id } : { name: "notfound" };
  return { name: "notfound" };
}

// The same address with ?model=<key>, replacing any model it already names.
export function withModel(hash, key) {
  return `${String(hash).split("?")[0]}?model=${encodeURIComponent(key)}`;
}

export function caseHref(id) {
  return `#/case/${encodeURIComponent(id)}`;
}

export function mineHref(id) {
  return `#/mine/${encodeURIComponent(id)}`;
}
