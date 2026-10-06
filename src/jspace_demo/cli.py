"""The ``jspace-demo`` command.

    jspace-demo serve [--model KEY | --no-model] [--port 8000]   # on 127.0.0.1 only
    jspace-demo precompute --model KEY [--refresh] [--case ID ...]
    jspace-demo export [--model KEY ...]
    jspace-demo check [--model KEY]

Heavy imports (torch, transformers) happen inside the commands that need them, so ``--help`` stays quick.
"""

from __future__ import annotations

from jspace_demo.models import DEFAULT_MODEL, MODELS, spec

import argparse
import json
import sys
import time

READOUT_GIB = 2.0  # memory for the readout of every layer, on top of what the loaded model holds


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jspace-demo", description="What the LLM says, and what it has in mind.")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the local app")
    group = serve.add_mutually_exclusive_group()
    group.add_argument("--model", default=DEFAULT_MODEL, choices=MODELS, help="model for your own text")
    group.add_argument("--no-model", action="store_true", help="the screens and prepared cases only")
    serve.add_argument("--port", type=int, default=8000)

    pre = sub.add_parser("precompute", help="analyse the prepared cases with a model, then export them")
    pre.add_argument("--model", required=True, choices=MODELS)
    pre.add_argument("--refresh", action="store_true", help="analyse cached cases again")
    pre.add_argument("--case", action="append", dest="cases", metavar="ID", help="only this case (repeatable)")

    exp = sub.add_parser("export", help="write web/data from the analysed cases (no model needed)")
    exp.add_argument("--model", action="append", dest="models", choices=MODELS, help="only this model (repeatable)")

    chk = sub.add_parser("check", help="check that a model and its lens load and read out as expected")
    chk.add_argument("--model", default=DEFAULT_MODEL, choices=MODELS)

    args = parser.parse_args(argv)
    return {"serve": _serve, "precompute": _precompute, "export": _export, "check": _check}[args.command](args)


def _serve(args) -> int:
    from jspace_demo.server import HOST, create_app

    import logging

    import uvicorn

    if not args.no_model and not _ready_to_load(args.model):
        print("start with --no-model to see the screens and prepared cases only", file=sys.stderr)
        return 3
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:     %(name)s: %(message)s")
    app = create_app(None if args.no_model else args.model)
    uvicorn.run(app, host=HOST, port=args.port, workers=1)  # this machine only: the API has no accounts
    return 0


def _ready_to_load(model_key: str) -> bool:
    from jspace_demo import preflight

    reasons = preflight.busy_reasons(required_free_gib=spec(model_key).memory_gib + READOUT_GIB)
    if reasons:
        print("not loading the model: " + "; ".join(reasons), file=sys.stderr)
    return not reasons


def _precompute(args) -> int:
    from jspace_demo import analysis, cases, precompute, views

    unknown = sorted(set(args.cases or ()) - set(cases.all_cases()))
    if unknown:  # before loading a model for nothing
        print(f"unknown case {', '.join(unknown)}; see src/jspace_demo/data/cases.json", file=sys.stderr)
        return 2
    if not _ready_to_load(args.model):
        return 3
    start = time.time()
    loaded = analysis.load(args.model)
    print(f"{args.model} ready in {time.time() - start:.0f}s ({loaded.device}, {loaded.dtype})", flush=True)
    clock = {"last": time.time()}

    def report(case, payload, computed):
        alerts = views.concern_alerts(payload, case.watch)
        found = f"{alerts[0]['word']} rank {alerts[0]['rank']}" if alerts else "no watch word"
        took = f"{time.time() - clock['last']:5.1f}s" if computed else " cached"
        clock["last"] = time.time()
        print(f"  {case.id:<28} {took}  {found}", flush=True)

    precompute.compute(loaded, refresh=args.refresh, only=args.cases, on_case=report)
    if args.cases:
        print("analysed the chosen cases; run `jspace-demo export` once every case is analysed")
        return 0
    written = precompute.export([args.model])
    print(f"exported {written[args.model]} cases to web/data/{args.model}/")
    return 0


def _export(args) -> int:
    from jspace_demo import precompute

    for key, n in precompute.export(args.models).items():
        print(f"{key}: {n} cases")
    return 0


def _check(args) -> int:
    from jspace_demo import health, runtime

    if not _ready_to_load(args.model):
        return 3
    device = runtime.pick_device()
    model, tokenizer = runtime.load_model(args.model, runtime.DEFAULT_DTYPE, device)
    lens = runtime.load_lens(args.model, device=device)
    result = health.check(args.model, model, lens, tokenizer)
    result["memory_gib"] = runtime.held_memory_gib(device)
    print(json.dumps(result, ensure_ascii=False, indent=1))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
