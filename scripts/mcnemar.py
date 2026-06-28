#!/usr/bin/env python3
"""Paired McNemar test between two result files on the same examples.

Use for every accuracy claim (e.g. base vs +self_correct, lexical vs embedding
linking): is the difference statistically significant or noise?

    python scripts/mcnemar.py results/qwen25-coder-32b.base.jsonl \
                              results/qwen25-coder-32b.correct.jsonl

Exact (binomial) two-sided p-value over the discordant pairs; no SciPy needed.
"""
from __future__ import annotations

import argparse
import json
import math


def load_correct(path: str) -> dict[int, bool]:
    out = {}
    for line in open(path):
        r = json.loads(line)
        out[r["qid"]] = bool(r["correct"])
    return out


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value: binomial(min(b,c); b+c, 0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("file_a")
    ap.add_argument("file_b")
    args = ap.parse_args()
    A, B = load_correct(args.file_a), load_correct(args.file_b)
    qids = sorted(set(A) & set(B))
    n = len(qids)
    acc_a = sum(A[q] for q in qids) / n
    acc_b = sum(B[q] for q in qids) / n
    b = sum(1 for q in qids if A[q] and not B[q])   # A right, B wrong
    c = sum(1 for q in qids if not A[q] and B[q])   # A wrong, B right
    p = mcnemar_exact_p(b, c)
    import os
    print(f"A = {os.path.basename(args.file_a)}  acc={acc_a*100:.2f}%")
    print(f"B = {os.path.basename(args.file_b)}  acc={acc_b*100:.2f}%")
    print(f"paired n={n}  delta={ (acc_b-acc_a)*100:+.2f} pp")
    print(f"discordant: A-only-right b={b}, B-only-right c={c}")
    print(f"McNemar exact two-sided p = {p:.4g}  "
          f"({'significant' if p < 0.05 else 'NOT significant'} at 0.05)")


if __name__ == "__main__":
    main()
