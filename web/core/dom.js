// Small helpers that build the page from data. Text is always set as text: nothing a model wrote is parsed as HTML.

const SVG = "http://www.w3.org/2000/svg";

export function h(tag, props, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(props || {})) {
    if (value === null || value === undefined || value === false) continue;
    if (key.startsWith("on") && typeof value === "function") el.addEventListener(key.slice(2).toLowerCase(), value);
    else el.setAttribute(key, value === true ? "" : String(value));
  }
  return append(el, children);
}

export function append(el, children) {
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    el.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return el;
}

export function clear(el) {
  el.replaceChildren();
  return el;
}

const PATHS = {
  alert: ["M12 3.5 2.8 19.5h18.4L12 3.5z", "M12 10v4", "M12 17v.5"],
  check: ["M5 12.5l4.5 4.5L19 7.5"],
  info: ["M12 11v5", "M12 7.5v.5"],
  arrowRight: ["M5 12h14", "M13 6l6 6-6 6"],
  arrowLeft: ["M19 12H5", "M11 6l-6 6 6 6"],
  arrowUp: ["M12 19V5", "M6 11l6-6 6 6"],
  trash: ["M4 7h16", "M10 11v6", "M14 11v6", "M6 7l1 13h10l1-13", "M9 7V4h6v3"],
  sun: [
    "M12 2.5v2",
    "M12 19.5v2",
    "M2.5 12h2",
    "M19.5 12h2",
    "M5.3 5.3l1.4 1.4",
    "M17.3 17.3l1.4 1.4",
    "M5.3 18.7l1.4-1.4",
    "M17.3 6.7l1.4-1.4",
  ],
  moon: ["M20 14.5A8 8 0 1 1 9.5 4a6.5 6.5 0 0 0 10.5 10.5z"],
};
const CIRCLES = { info: 9, sun: 4 }; // the radius of an icon's circle, centred

export function icon(name, size = 16) {
  const svg = document.createElementNS(SVG, "svg");
  for (const [k, v] of Object.entries({
    width: size,
    height: size,
    viewBox: "0 0 24 24",
    "aria-hidden": "true",
    class: "icon",
    focusable: "false",
  }))
    svg.setAttribute(k, v);
  if (CIRCLES[name]) {
    const circle = document.createElementNS(SVG, "circle");
    for (const [k, v] of Object.entries({ cx: 12, cy: 12, r: CIRCLES[name] })) circle.setAttribute(k, v);
    svg.append(circle);
  }
  for (const d of PATHS[name] || []) {
    const path = document.createElementNS(SVG, "path");
    path.setAttribute("d", d);
    svg.append(path);
  }
  return svg;
}
