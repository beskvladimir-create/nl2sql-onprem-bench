#!/usr/bin/env python3
"""Model-free gold-table recall of a schema linker over BIRD dev.

Lets you pick k for the embedding linker so it holds >=95% gold-table recall,
and quantify how much the lexical linker prunes away the answer table.

    # lexical (no heavy deps; runs anywhere):
    python scripts/linker_recall.py --data data/bird --mode lexical
    # embedding, sweep k (needs sentence-transformers; run on the GPU pod):
    python scripts/linker_recall.py --data data/bird --mode emb --k 3 4 5 6 8
"""
from __future__ import annotations

import argparse
import statistics

from nl2sql_bench.data import load_bird_dev
from nl2sql_bench.schema import link_schema, link_schema_emb, table_recall


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/bird")
    ap.add_argument("--mode", choices=["lexical", "emb"], required=True)
    ap.add_argument("--k", type=int, nargs="*", default=[5],
                    help="top-k values to sweep (emb mode)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    exs = load_bird_dev(args.data)
    if args.limit:
        exs = exs[: args.limit]

    if args.mode == "lexical":
        recs, kept_sizes = [], []
        for ex in exs:
            kept = link_schema(ex.db_path, ex.question, ex.evidence)
            r = table_recall(kept, ex.gold_sql)
            if r is not None:
                recs.append(r)
                kept_sizes.append(len(kept))
        print(f"lexical: n={len(recs)}  mean gold-table recall={statistics.mean(recs)*100:.2f}%"
              f"  full-recall cases={sum(1 for r in recs if r==1.0)}/{len(recs)}"
              f"  mean kept tables={statistics.mean(kept_sizes):.2f}")
    else:
        for k in args.k:
            recs, kept_sizes = [], []
            for ex in exs:
                kept = link_schema_emb(ex.db_path, ex.question, ex.evidence, k=k)
                r = table_recall(kept, ex.gold_sql)
                if r is not None:
                    recs.append(r)
                    kept_sizes.append(len(kept))
            print(f"emb k={k}: n={len(recs)}  mean gold-table recall="
                  f"{statistics.mean(recs)*100:.2f}%"
                  f"  full-recall cases={sum(1 for r in recs if r==1.0)}/{len(recs)}"
                  f"  mean kept tables={statistics.mean(kept_sizes):.2f}")


if __name__ == "__main__":
    main()
