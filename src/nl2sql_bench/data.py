"""Load the BIRD dev split.

Expected layout under `data/bird/`:
    dev.json                 list of {question_id, db_id, question, evidence, SQL}
    dev_databases/<db_id>/<db_id>.sqlite

Download: https://bird-bench.github.io/  (dev set + databases).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass


@dataclass
class Example:
    qid: int
    db_id: str
    question: str
    evidence: str          # BIRD "external knowledge" hint
    gold_sql: str
    db_path: str           # absolute path to the sqlite file


def load_bird_dev(root: str) -> list[Example]:
    """Read dev.json and resolve each db's sqlite path. Fail loudly if missing."""
    dev_json = os.path.join(root, "dev.json")
    db_root = os.path.join(root, "dev_databases")
    if not os.path.isfile(dev_json):
        raise FileNotFoundError(
            f"{dev_json} not found. Download BIRD dev from https://bird-bench.github.io/"
        )
    with open(dev_json) as f:
        rows = json.load(f)

    out: list[Example] = []
    for r in rows:
        db_id = r["db_id"]
        db_path = os.path.join(db_root, db_id, f"{db_id}.sqlite")
        out.append(
            Example(
                qid=r.get("question_id", len(out)),
                db_id=db_id,
                question=r["question"].strip(),
                evidence=(r.get("evidence") or "").strip(),
                gold_sql=r["SQL"].strip(),
                db_path=db_path,
            )
        )
    missing = [e.db_id for e in out if not os.path.isfile(e.db_path)]
    if missing:
        raise FileNotFoundError(
            f"{len(set(missing))} db files missing under {db_root}, e.g. {sorted(set(missing))[:3]}"
        )
    return out
