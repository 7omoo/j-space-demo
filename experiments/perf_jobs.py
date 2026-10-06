"""Timing and memory of the local app on Japanese input, through the page's own path (docs/testing.md).

    uv run jspace-demo serve        # in another terminal, with the model loaded
    JSPACE_URL=http://127.0.0.1:8000 uv run python experiments/perf_jobs.py

Submits 20 Japanese messages of varying length as jobs, one after another (translate -> reply -> read ->
translate the reply), takes each step's time from the saved analysis, then deletes what it saved. Checks that
every job finished, that the server's MPS memory peaks at or under 12 GiB and grows by at most 1.5 GiB, that swap
use does not grow, and that 90% of jobs take at most 30 s (the try screen promises 10-30 s). Writes
out/experiments/perf_jobs.json and exits 1 on a failed check. Run it next to the server: the step times are read
from the same history folder (JSPACE_HISTORY_DIR, default data/history).
"""

from jspace_demo import history, preflight

import json
import os
import statistics
import sys
import time
from pathlib import Path

import httpx2 as httpx

URL = os.environ.get("JSPACE_URL", "http://127.0.0.1:8000")
SENTENCES = [
    "国際宝くじで200万ドルが当たったというメールが来ました。",
    "ネットバンキングのパスワードと手数料500ドルを送れば受け取れるそうです。",
    "今週末、築100年の家の壁の古い塗装をサンダーで削る予定です。",
    "マスクは持っていませんが、窓は開けておきます。",
    "先月から医者にワルファリンを処方されていて、頭痛のためにアスピリンも毎日飲んでいます。",
    "車を売った代金を現金で入金したいので、必要な書類を教えてください。",
    "小さな子どもがいるので、洗剤の安全な保管場所についても知りたいです。",
    "どうすればいいか、順番に教えてもらえますか？",
]
LENGTHS = [1, 2, 4, 8] * 5  # sentences per message: about 15 to 190 characters, 20 jobs in all


def message(i: int, n: int) -> str:
    return "".join(SENTENCES[(i + j) % len(SENTENCES)] for j in range(n))


def mps_gib(client: httpx.Client) -> float:
    meta = client.get("/api/meta").json()
    if meta["status"] != "ready" or meta.get("memory_gib") is None:
        sys.exit(f"server not ready on MPS: {meta['status']}")
    return meta["memory_gib"]


def run(client: httpx.Client, text: str) -> dict:
    job = client.post("/api/jobs", json={"message": text, "watch_set": "general", "reply_lang": "ja"}).json()
    while job["status"] in ("queued", "running"):
        time.sleep(0.25)
        job = client.get(f"/api/jobs/{job['id']}").json()
    return job


with httpx.Client(base_url=URL, timeout=60) as client:
    swap_before, mem_before = preflight.swap_used_mib(), mps_gib(client)
    rows = []
    for i, n in enumerate(LENGTHS):
        text = message(i, n)
        job = run(client, text)
        row = {"job": i + 1, "sentences": n, "chars": len(text), "status": job["status"], "error": job["error"]}
        if job["status"] == "done":
            record = history.load(job["result"])
            if record is None:
                sys.exit(
                    f"saved analysis {job['result']} not found in {history.history_dir()}: set JSPACE_HISTORY_DIR as the "
                    "server has it"
                )
            timing = record["payload"]["timing"]
            row.update(
                wall_seconds=round(job["finished"] - job["started"], 1),
                n_tokens=timing["n_tokens"],
                translate_seconds=timing.get("translate_seconds"),
                generate_seconds=timing["generate_seconds"],
                read_seconds=timing["read_seconds"],
            )
            client.delete(f"/api/analyses/{job['result']}")
        row["mps_gib"] = mps_gib(client)
        rows.append(row)
        print(row, flush=True)
    swap_after, mem_after = preflight.swap_used_mib(), mps_gib(client)

done = [r for r in rows if r["status"] == "done"]
walls = sorted(r["wall_seconds"] for r in done) or [float("nan")]
summary = {
    "jobs": len(rows),
    "done": len(done),
    "mps_gib_before": mem_before,
    "mps_gib_after": mem_after,
    "mps_gib_peak": max(r["mps_gib"] for r in rows),
    "mps_growth_gib": round(mem_after - mem_before, 2),
    "swap_mib_before": swap_before,
    "swap_mib_after": swap_after,
    "wall_seconds_median": statistics.median(walls),
    "wall_seconds_p90": walls[int(0.9 * (len(walls) - 1))],
    "wall_seconds_max": walls[-1],
    "translate_seconds_median": statistics.median(r["translate_seconds"] for r in done) if done else None,
}
checks = {
    "every job finished": len(done) == len(rows),
    "MPS memory peak <= 12 GiB": summary["mps_gib_peak"] <= 12,
    "MPS memory growth <= 1.5 GiB": summary["mps_growth_gib"] <= 1.5,
    "swap did not grow": swap_after <= swap_before + 1,
    "90% of jobs within 30 s": summary["wall_seconds_p90"] <= 30,
}
out = Path("out/experiments/perf_jobs.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({"summary": summary, "checks": checks, "rows": rows}, ensure_ascii=False, indent=1))
print(json.dumps(summary, ensure_ascii=False))
for name, ok in checks.items():
    print(("PASS " if ok else "FAIL ") + name)
sys.exit(0 if all(checks.values()) else 1)
