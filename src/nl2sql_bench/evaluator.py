"""Execution Accuracy (EX), BIRD-style.

EX = 1 iff the result set of the predicted SQL equals the result set of the gold
SQL when both run against the same sqlite db. Order-insensitive set comparison,
matching the official BIRD evaluator semantics. Execution is time-bounded so a
runaway predicted query cannot hang the run.
"""
from __future__ import annotations

import sqlite3

from func_timeout import FunctionTimedOut, func_timeout


def _run(db_path: str, sql: str) -> list[tuple]:
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute(sql)
        return cur.fetchall()
    finally:
        conn.close()


def _exec(db_path: str, sql: str, timeout: float) -> list[tuple] | None:
    """Return rows, or None if the query errors or times out."""
    try:
        return func_timeout(timeout, _run, args=(db_path, sql))
    except FunctionTimedOut:
        return None
    except sqlite3.Error:
        return None


def result_set_match(pred_rows: list[tuple], gold_rows: list[tuple]) -> bool:
    # Order-insensitive multiset comparison (BIRD compares set equality of rows).
    return sorted(map(repr, pred_rows)) == sorted(map(repr, gold_rows))


def execution_accuracy(
    db_path: str, pred_sql: str, gold_sql: str, timeout: float = 30.0
) -> tuple[bool, str | None]:
    """Return (is_correct, error). error is set when the predicted SQL fails.

    The gold query is assumed valid; if it errors we surface that loudly so a bad
    data setup never silently counts as a wrong prediction.
    """
    gold_rows = _exec(db_path, gold_sql, timeout)
    if gold_rows is None:
        return False, "gold_sql failed to execute (check db setup)"
    pred_rows = _exec(db_path, pred_sql, timeout)
    if pred_rows is None:
        return False, "pred_sql failed to execute"
    return result_set_match(pred_rows, gold_rows), None
