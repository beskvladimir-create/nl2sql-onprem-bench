#!/usr/bin/env python3
"""CLI: run one model through the full ablation (base -> +link -> +correct -> +consistency).

Example:
    python scripts/run_bench.py --config configs/run.yaml --model qwen25-coder-32b
    python scripts/run_bench.py --config configs/run.yaml --model qwen25-coder-32b --limit 50
"""
from __future__ import annotations

import argparse
import json
import os

import yaml

from nl2sql_bench.model_client import ModelClient
from nl2sql_bench.runner import run

ABLATION = [
    ("base", dict(schema_link=False, self_correct=False, self_consistency=False)),
    ("correct", dict(schema_link=False, self_correct=True, self_consistency=False)),
    ("consistency", dict(schema_link=False, self_correct=True, self_consistency=True)),
    ("link", dict(schema_link=True, self_correct=False, self_consistency=False)),
    ("link_correct", dict(schema_link=True, self_correct=True, self_consistency=False)),
    ("link_correct_consistency", dict(schema_link=True, self_correct=True, self_consistency=True)),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True, help="key in config 'models'")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--stages", nargs="*", default=None,
                    help="subset of ablation stage names; default all")
    ap.add_argument("--concurrency", type=int, default=None,
                    help="examples processed in parallel; overrides config")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    m = cfg["models"][args.model]
    data_root = cfg["data_root"]
    out_dir = cfg.get("out_dir", "results")
    concurrency = args.concurrency or m.get("concurrency") or cfg.get("concurrency", 32)

    client = ModelClient(
        model=m["model"],
        base_url=m.get("base_url", "http://localhost:8000/v1"),
        api_key=m.get("api_key", "EMPTY"),
        max_tokens=m.get("max_tokens", 1024),
    )

    stages = [s for s in ABLATION if args.stages is None or s[0] in args.stages]
    for name, flags in stages:
        out = os.path.join(out_dir, f"{args.model}.{name}.jsonl")
        summary = run(data_root, client, out, limit=args.limit,
                      concurrency=concurrency, **flags)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
