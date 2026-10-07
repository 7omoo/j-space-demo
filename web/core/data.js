// Where the screens get their data. The prepared cases are always files under data/ (written by
// `jspace-demo export`), on the static site and in the local app alike. The local app adds an API for what needs a
// model: its state, the user's own analyses and their history. config.json says which of the two is running, so the
// static site never asks for an API it does not have. Paths are relative, so any sub-path works (7omo.com/j-space/).

const MODEL_KEY = "jspace.model";

export async function getJSON(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail = null;
    try {
      detail = (await response.json()).detail ?? null;
    } catch {
      // no JSON body
    }
    const error = new Error(typeof detail === "string" ? detail : `HTTP ${response.status}`);
    error.status = response.status;
    error.detail = detail; // a screen can word a structured detail (the model's state on a 503)
    throw error;
  }
  return response.json();
}

export async function createData() {
  const [config, models] = await Promise.all([getJSON("config.json"), getJSON("data/models.json")]);
  const data = new Data(models, Boolean(config.live));
  if (data.live) await data.refreshMeta();
  return data;
}

function remembered() {
  try {
    return localStorage.getItem(MODEL_KEY);
  } catch {
    return null;
  }
}

export class Data {
  constructor(models, live) {
    this.models = models.models;
    this.live = live;
    this.meta = { status: live ? "idle" : "static" };
    const saved = remembered();
    this.model = this.has(saved) ? saved : (models.default ?? this.models[0]?.key);
    this.cache = new Map();
  }

  has(key) {
    return this.models.some((m) => m.key === key);
  }

  // The model whose prepared cases the screens show (the switcher in the header).
  modelInfo(key = this.model) {
    return this.models.find((m) => m.key === key) ?? null;
  }

  setModel(key) {
    if (!this.has(key)) return false;
    this.model = key;
    try {
      localStorage.setItem(MODEL_KEY, key);
    } catch {
      // not remembered; the page still switches
    }
    return true;
  }

  // Cached by path: the same case is read once however often a screen draws it.
  file(path) {
    if (!this.cache.has(path)) {
      this.cache.set(
        path,
        getJSON(path).catch((error) => {
          this.cache.delete(path);
          throw error;
        }),
      );
    }
    return this.cache.get(path);
  }

  async cases(model = this.model) {
    return (await this.file(`data/${model}/index.json`)).cases;
  }

  case(id, model = this.model) {
    return this.file(`data/${model}/cases/${encodeURIComponent(id)}.json`);
  }

  // Local app only: the model that analyses the user's own text, and its history.

  async refreshMeta() {
    if (!this.live) return this.meta;
    this.meta = await getJSON("api/meta");
    return this.meta;
  }

  analyses() {
    return this.live ? getJSON("api/analyses") : Promise.resolve([]);
  }

  analysis(id) {
    return getJSON(`api/analyses/${encodeURIComponent(id)}`);
  }

  deleteAnalysis(id) {
    return getJSON(`api/analyses/${encodeURIComponent(id)}`, { method: "DELETE" });
  }

  submitJob(body) {
    return getJSON("api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  }

  job(id) {
    return getJSON(`api/jobs/${encodeURIComponent(id)}`);
  }

  cancelJob(id) {
    return getJSON(`api/jobs/${encodeURIComponent(id)}`, { method: "DELETE" });
  }
}
