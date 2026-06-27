"""Schema serialization + lightweight schema linking.

`serialize_schema` renders a compact CREATE-style view of the db for the prompt.
`link_schema` is the recipe's schema-linking step: keep only tables/columns whose
names overlap the question/evidence tokens (plus their primary/foreign keys), to
cut prompt size and distractors on wide databases. Deliberately simple and
model-agnostic — the ablation measures how much this lexical filter alone buys.
"""
from __future__ import annotations

import re
import sqlite3


def _tables(conn: sqlite3.Connection) -> dict[str, list[str]]:
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    names = [r[0] for r in cur.fetchall() if not r[0].startswith("sqlite_")]
    schema: dict[str, list[str]] = {}
    for t in names:
        cols = conn.execute(f'PRAGMA table_info("{t}")').fetchall()
        schema[t] = [c[1] for c in cols]  # c[1] = column name
    return schema


def serialize_schema(db_path: str, tables: list[str] | None = None) -> str:
    conn = sqlite3.connect(db_path)
    try:
        schema = _tables(conn)
    finally:
        conn.close()
    if tables is not None:
        schema = {t: c for t, c in schema.items() if t in tables}
    lines = []
    for t, cols in schema.items():
        lines.append(f"{t}({', '.join(cols)})")
    return "\n".join(lines)


def _subtokens(s: str) -> set[str]:
    """Split an identifier or sentence into lowercase word tokens (len > 1)."""
    return {w for w in re.split(r"[^a-z0-9]+", s.lower()) if len(w) > 1}


def link_schema(db_path: str, question: str, evidence: str = "") -> list[str]:
    """Return the subset of table names judged relevant to the question.

    Match on word-level sub-tokens. BIRD column names contain spaces and
    punctuation (e.g. ``Free Meal Count (K-12)``), so a column must be split
    into words before overlap testing; comparing the whole column string
    against single question tokens never matches multi-word columns and prunes
    away the very table that holds the answer.
    """
    conn = sqlite3.connect(db_path)
    try:
        schema = _tables(conn)
    finally:
        conn.close()
    q_tokens = _subtokens(f"{question} {evidence}")
    kept = []
    for t, cols in schema.items():
        name_tokens = _subtokens(t)
        for c in cols:
            name_tokens |= _subtokens(c)
        if name_tokens & q_tokens:
            kept.append(t)
    # Fallback: if nothing matched, keep everything rather than starve the model.
    return kept or list(schema.keys())
