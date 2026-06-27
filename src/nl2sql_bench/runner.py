"""Run a recipe config over BIRD dev and write per-example results + a summary.

Examples are processed concurrently: each example is independent, so a thread
pool fans them out and the vLLM server batches the in-flight requests. Within an
example the recipe stays sequential (generate -> correct -> vote). Records are
written in the original dataset order regardless of completion order.
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict

from tqdm import tqdm

from .data import Example, load_bird_dev
from .evaluator import execution_accuracy
from .model_client import ModelClient
from .pipeline import Recipe


def _process(
    ex: Example,
    client: ModelClient,
    flags: dict,
    exec_timeout: float,
) -> dict:
    recipe = Recipe(client=client, exec_timeout=exec_timeout, **flags)
    try:
        pred = recipe.run(ex.db_path, ex.question, ex.evidence)
        ok, err = execution_accuracy(ex.db_path, pred, ex.gold_sql, exec_timeout)
    except Exception as e:  # one wedged/erroring example must not kill the stage
        return {
            "qid": ex.qid, "db_id": ex.db_id, "correct": False,
            "error": f"runtime: {type(e).__name__}: {e}",
            "pred_sql": "", "gold_sql": ex.gold_sql,
            "usage": asdict(recipe.usage),
        }
    return {
        "qid": ex.qid, "db_id": ex.db_id, "correct": ok, "error": err,
        "pred_sql": pred, "gold_sql": ex.gold_sql,
        "usage": asdict(recipe.usage),
    }


def run(
    data_root: str,
    client: ModelClient,
    out_path: str,
    *,
    schema_link: bool = False,
    self_correct: bool = False,
    self_consistency: bool = False,
    limit: int | None = None,
    exec_timeout: float = 30.0,
    concurrency: int = 32,
) -> dict:
    examples: list[Example] = load_bird_dev(data_root)
    if limit:
        examples = examples[:limit]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    flags = dict(
        schema_link=schema_link,
        self_correct=self_correct,
        self_consistency=self_consistency,
    )

    records: list[dict | None] = [None] * len(examples)
    workers = max(1, min(concurrency, len(examples)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {
            pool.submit(_process, ex, client, flags, exec_timeout): i
            for i, ex in enumerate(examples)
        }
        for fut in tqdm(as_completed(futs), total=len(futs), desc=os.path.basename(out_path)):
            i = futs[fut]
            records[i] = fut.result()

    n_correct = sum(int(r["correct"]) for r in records if r)
    with open(out_path, "w") as fout:
        for r in records:
            fout.write(json.dumps(r) + "\n")

    n = len(examples)
    summary = {
        "model": client.model,
        "n": n,
        "ex_accuracy": round(n_correct / n, 4) if n else 0.0,
        "config": {
            "schema_link": schema_link,
            "self_correct": self_correct,
            "self_consistency": self_consistency,
        },
        "concurrency": workers,
        "results_file": out_path,
    }
    with open(out_path.replace(".jsonl", ".summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary
