"""The recipe. Each stage is independently toggleable so the ablation can isolate
its contribution:

    base               single greedy generation
    +schema_link       restrict prompt schema to linked tables
    +self_correct      execute; on error, feed the error back and revise (<=k)
    +self_consistency  sample m candidates, keep the majority by result set

Every stage returns the final SQL plus a usage tally (tokens, model calls,
exec calls) so cost can be attributed per component.
"""
from __future__ import annotations

import re
import sqlite3
from collections import Counter
from dataclasses import dataclass, field

from func_timeout import FunctionTimedOut, func_timeout

from .evaluator import _exec
from .model_client import ModelClient
from .schema import link_schema, link_schema_emb, serialize_schema

_SQL_BLOCK = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_sql(text: str) -> str:
    """Pull the SQL out of a model reply (fenced block if present, else best line)."""
    m = _SQL_BLOCK.search(text)
    cand = m.group(1) if m else text
    cand = cand.strip().rstrip(";").strip()
    # If the model prepended prose, keep from the first SELECT/WITH.
    low = cand.lower()
    for kw in ("with ", "select "):
        i = low.find(kw)
        if i != -1:
            return cand[i:].strip()
    return cand


def _prompt(schema: str, question: str, evidence: str) -> str:
    ev = f"\nExternal knowledge: {evidence}" if evidence else ""
    return (
        "You are an expert SQLite analyst. Given the schema, write ONE SQLite "
        "query that answers the question. Return only the query in a ```sql block.\n\n"
        f"Schema:\n{schema}\n\nQuestion: {question}{ev}"
    )


def _repair_prompt(schema: str, question: str, sql: str, error: str) -> str:
    return (
        "The query below failed. Fix it. Return only the corrected query in a "
        "```sql block.\n\n"
        f"Schema:\n{schema}\n\nQuestion: {question}\n\nQuery:\n{sql}\n\nError: {error}"
    )


@dataclass
class Usage:
    model_calls: int = 0
    exec_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0

    def add_call(self, c) -> None:
        self.model_calls += 1
        self.prompt_tokens += c.prompt_tokens
        self.completion_tokens += c.completion_tokens
        self.latency_s += c.latency_s


@dataclass
class Recipe:
    client: ModelClient
    schema_link: bool = False
    schema_link_emb: bool = False
    emb_top_k: int = 5
    emb_model: str = "all-MiniLM-L6-v2"
    self_correct: bool = False
    self_consistency: bool = False
    correct_iters: int = 2
    consistency_samples: int = 5
    sample_temperature: float = 0.6
    exec_timeout: float = 30.0
    usage: Usage = field(default_factory=Usage)
    linked_tables: list[str] | None = None  # tables kept by the active linker (for recall)

    def _schema_text(self, db_path: str, question: str, evidence: str) -> str:
        # embedding linker takes precedence over the lexical one when both set
        if self.schema_link_emb:
            tables = link_schema_emb(db_path, question, evidence,
                                     k=self.emb_top_k, model_name=self.emb_model)
            self.linked_tables = tables
            return serialize_schema(db_path, tables)
        if self.schema_link:
            tables = link_schema(db_path, question, evidence)
            self.linked_tables = tables
            return serialize_schema(db_path, tables)
        return serialize_schema(db_path)

    def _error_of(self, db_path: str, sql: str) -> str | None:
        """Return an error string if the SQL fails to execute, else None.

        Time-bounded: a runaway predicted query (e.g. an accidental cross join)
        must not hang the self-correct loop forever. A timeout is reported as an
        error so the model gets a repair signal AND the thread is freed.
        """
        self.usage.exec_calls += 1

        def _try() -> None:
            conn = sqlite3.connect(db_path)
            try:
                conn.execute(sql).fetchall()
            finally:
                conn.close()

        try:
            func_timeout(self.exec_timeout, _try)
            return None
        except FunctionTimedOut:
            return f"query exceeded {self.exec_timeout}s time limit"
        except sqlite3.Error as e:
            return str(e)
        except Exception as e:  # malformed SQL can raise non-sqlite errors too
            return str(e)

    def _generate(self, schema: str, question: str, evidence: str, temp: float) -> str:
        c = self.client.complete(_prompt(schema, question, evidence), temperature=temp)
        self.usage.add_call(c)
        return extract_sql(c.text)

    def _correct(self, schema: str, question: str, db_path: str, sql: str) -> str:
        for _ in range(self.correct_iters):
            err = self._error_of(db_path, sql)
            if err is None:
                return sql
            c = self.client.complete(_repair_prompt(schema, question, sql, err))
            self.usage.add_call(c)
            sql = extract_sql(c.text)
        return sql

    def run(self, db_path: str, question: str, evidence: str) -> str:
        schema = self._schema_text(db_path, question, evidence)

        if self.self_consistency:
            cands = [
                self._generate(schema, question, evidence, self.sample_temperature)
                for _ in range(self.consistency_samples)
            ]
            if self.self_correct:
                cands = [self._correct(schema, question, db_path, s) for s in cands]
            sql = self._vote(db_path, cands)
        else:
            sql = self._generate(schema, question, evidence, self.client.temperature)
            if self.self_correct:
                sql = self._correct(schema, question, db_path, sql)
        return sql

    def _vote(self, db_path: str, cands: list[str]) -> str:
        """Execution-guided majority vote: group candidates by result set, pick the
        SQL from the largest group of *successfully executing* queries."""
        buckets: dict[str, list[str]] = {}
        for s in cands:
            self.usage.exec_calls += 1
            rows = _exec(db_path, s, self.exec_timeout)
            if rows is None:
                continue
            key = repr(sorted(map(repr, rows)))
            buckets.setdefault(key, []).append(s)
        if not buckets:
            return cands[0]  # all failed; return something for the record
        best = max(buckets.values(), key=len)
        return best[0]
