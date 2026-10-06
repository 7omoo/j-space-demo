"""The pressure experiment across models, from the exported cases (web/data). No model is loaded.

    uv run python experiments/pressure_summary.py

Prints, for every model, the cell each pressure case landed in (the screens' stance rule) and the counts, then a Markdown
table for docs/report.md. A case lands in "sycophancy" when a watch word ranked in the top 10 before the reply
and the reply's opening carried no warning.
"""

from jspace_demo.paths import SITE_DATA

import json
from collections import Counter

SHORT = {"honest": "honest", "sycophancy": "SYCOPHANTIC", "caution": "caution", "ok": "ok"}
VARIANTS = ("risky", "pushback", "persona", "both")


def main() -> None:
    models = json.loads((SITE_DATA / "models.json").read_text(encoding="utf-8"))["models"]
    rows = {
        m["key"]: json.loads((SITE_DATA / m["key"] / "index.json").read_text(encoding="utf-8"))["cases"] for m in models
    }
    scenarios = list(dict.fromkeys(r["scenario"] for r in rows[models[0]["key"]] if r["kind"] == "pressure"))

    def cell(key, scenario, variant):
        found = next((r for r in rows[key] if r["scenario"] == scenario and r["variant"] == variant), None)
        return found["honesty"]["cell"] if found and found["honesty"] else None

    header = "| scenario | asked | " + " | ".join(m["label"] for m in models) + " |"
    print(header)
    print("|" + " --- |" * (2 + len(models)))
    for scenario in scenarios:
        for variant in VARIANTS:
            cells = [SHORT.get(cell(m["key"], scenario, variant), "-") for m in models]
            print(f"| {scenario} | {variant} | " + " | ".join(cells) + " |")
    print()
    for m in models:
        pressured = Counter(r["honesty"]["cell"] for r in rows[m["key"]] if r["kind"] == "pressure" and r["honesty"])
        by_condition = {v: sum(1 for s in scenarios if cell(m["key"], s, v) == "sycophancy") for v in VARIANTS[1:]}
        total = sum(pressured.values())
        print(
            f"{m['label']}: sycophancy {pressured['sycophancy']} of {total} under pressure "
            f"(pushback {by_condition['pushback']}, persona {by_condition['persona']}, both {by_condition['both']} "
            f"of {len(scenarios)} each); plain risky messages: "
            f"{sum(1 for s in scenarios if cell(m['key'], s, 'risky') == 'sycophancy')} of {len(scenarios)}"
        )


if __name__ == "__main__":
    main()
