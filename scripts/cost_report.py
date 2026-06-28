#!/usr/bin/env python3
"""Real cost per stage from wall-clock + a $/GPU-hour rate (not a token proxy).

Reads results/*.summary.json (which now carry wall_s and n) and, when the
matching .jsonl is present, the mean tokens/query. Prints $/1k-queries per stage.

    python scripts/cost_report.py --rate 1.8 results/qwen25-coder-32b.*.summary.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os


def mean_tokens(jsonl_path: str):
    if not os.path.exists(jsonl_path):
        return None, None
    n = pt = ct = 0
    for line in open(jsonl_path):
        u = json.loads(line).get("usage", {})
        pt += u.get("prompt_tokens", 0)
        ct += u.get("completion_tokens", 0)
        n += 1
    if not n:
        return None, None
    return pt / n, ct / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=float, required=True, help="$ per GPU-hour")
    ap.add_argument("summaries", nargs="+", help="*.summary.json (globs ok)")
    args = ap.parse_args()

    paths = []
    for p in args.summaries:
        paths.extend(sorted(glob.glob(p)))

    print(f"{'stage':<48}{'n':>6}{'wall_s':>9}{'q/s':>8}"
          f"{'tok/q':>8}{'$/1k-q':>9}")
    print("-" * 88)
    for p in paths:
        d = json.load(open(p))
        n = d.get("n") or 0
        wall = d.get("wall_s")
        stage = os.path.basename(p).replace(".summary.json", "")
        ptok, ctok = mean_tokens(p.replace(".summary.json", ".jsonl"))
        tokq = f"{(ptok + ctok):.0f}" if ptok is not None else "n/a"
        if wall and n:
            cost_1k = args.rate * (wall / 3600.0) / n * 1000.0
            qps = n / wall
            print(f"{stage:<48}{n:>6}{wall:>9.1f}{qps:>8.2f}{tokq:>8}{cost_1k:>9.3f}")
        else:
            print(f"{stage:<48}{n:>6}{'n/a':>9}{'n/a':>8}{tokq:>8}{'n/a':>9}"
                  "   (no wall_s; rerun with updated runner)")
    print(f"\n$/1k-queries = rate({args.rate}/GPU-hr) * wall_s/3600 / n * 1000")


if __name__ == "__main__":
    main()
