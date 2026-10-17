"""Repair for a serving defect seen on one OpenRouter provider (Cloudflare, the only
endpoint for qwen/qwen-2.5-coder-32b-instruct in October 2026): spaces adjacent to
digits are dropped from completions ("> 500 AND" -> ">500AND", "LIMIT 1" -> "LIMIT1"),
which makes some queries unexecutable. `repair_ws` re-inserts a space between a digit
and an adjacent upper-case SQL keyword, outside quoted literals and identifiers. Apply it to the extracted SQL (or to a
completion: code fences are not treated as quotes). It only inserts whitespace, so it never changes a query that already parses.
"""
from __future__ import annotations

import re

_KW_AFTER = re.compile(
    r"(?<=\d)(?=(?:AND|OR|ORDER|GROUP|LIMIT|OFFSET|FROM|WHERE|THEN|ELSE|END|AS|HAVING|UNION"
    r"|JOIN|INNER|LEFT|ON|WHEN|IN|IS|NOT|LIKE|BETWEEN|DESC|ASC|CASE|EXCEPT|INTERSECT)\b)")
_KW_BEFORE = re.compile(r"\b(LIMIT|OFFSET|THEN|ELSE|WHEN|AND|OR|SELECT|BETWEEN|BY)(?=\d)")
_QUOTED = re.compile(r"('(?:[^']|'')*'|\"[^\"\n]*\"|(?<!`)`[^`\n]+`(?!`)|\[[^\]\n]*\])")


def repair_ws(text: str) -> str:
    parts = _QUOTED.split(text)
    for i in range(0, len(parts), 2):  # even indices are outside quotes
        parts[i] = _KW_BEFORE.sub(r"\1 ", _KW_AFTER.sub(" ", parts[i]))
    return "".join(parts)
