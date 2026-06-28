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


# --- retrieval-based (embedding) schema linker -------------------------------
# Rehabilitated comparison vs the lexical linker: embed each table (name + its
# column names) and the question+evidence with a sentence-transformer, keep the
# top-k tables by cosine similarity. Heavy deps (sentence-transformers, torch)
# are imported lazily so the rest of the harness runs without them.

_EMB_MODEL = None
_EMB_CACHE: dict[str, tuple] = {}  # db_path -> (table_names, table_matrix)


def _get_embedder(model_name: str = "all-MiniLM-L6-v2"):
    global _EMB_MODEL
    if _EMB_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _EMB_MODEL = SentenceTransformer(model_name)
    return _EMB_MODEL


def _table_doc(name: str, cols: list[str]) -> str:
    return f"{name}: " + ", ".join(cols)


def _db_table_embeddings(db_path: str, model_name: str):
    """(table_names, normalized embedding matrix) for a db, cached per process."""
    if db_path in _EMB_CACHE:
        return _EMB_CACHE[db_path]
    import numpy as np
    conn = sqlite3.connect(db_path)
    try:
        schema = _tables(conn)
    finally:
        conn.close()
    names = list(schema.keys())
    docs = [_table_doc(t, schema[t]) for t in names]
    emb = _get_embedder(model_name).encode(docs, normalize_embeddings=True)
    emb = np.asarray(emb, dtype="float32")
    _EMB_CACHE[db_path] = (names, emb)
    return names, emb


def link_schema_emb(
    db_path: str, question: str, evidence: str = "", k: int = 5,
    model_name: str = "all-MiniLM-L6-v2",
) -> list[str]:
    """Top-k tables by cosine similarity between table docs and question+evidence."""
    import numpy as np
    names, emb = _db_table_embeddings(db_path, model_name)
    if len(names) <= k:
        return names
    q = _get_embedder(model_name).encode(
        [f"{question} {evidence}".strip()], normalize_embeddings=True
    )
    q = np.asarray(q, dtype="float32")[0]
    sims = emb @ q  # cosine (both already normalized)
    idx = np.argsort(-sims)[:k]
    return [names[i] for i in idx]


# --- gold-table recall (linker evaluation, model-free) -----------------------

_GOLD_TBL = re.compile(r"\b(?:from|join)\s+[`\"\[]?([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE)
_SQL_KW = {"select", "where", "group", "order", "by", "having", "on", "as",
           "and", "or", "limit", "distinct", "count", "sum", "avg", "min", "max"}


def gold_tables(sql: str) -> set[str]:
    """Best-effort set of table names referenced in a (gold) SQL query."""
    cands = {m.lower() for m in _GOLD_TBL.findall(sql)}
    return {t for t in cands if t not in _SQL_KW}


def table_recall(kept: list[str], gold_sql: str) -> float | None:
    """Fraction of gold tables retained by the linker (None if no gold tables found)."""
    gold = gold_tables(gold_sql)
    if not gold:
        return None
    keptl = {t.lower() for t in kept}
    return len(gold & keptl) / len(gold)
