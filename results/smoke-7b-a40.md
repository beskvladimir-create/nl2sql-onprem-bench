# Smoke run — Qwen2.5-Coder-7B-Instruct on RunPod A40 48GB (BIRD dev, limit=50)

Purpose: validate harness end-to-end before the 32B headline run. Not a headline number.

| stage                      | EX (50 ex) |
|----------------------------|-----------|
| base                       | 0.18 |
| +self_correct              | 0.22 |
| +self_consistency          | 0.28 |
| schema_link                | 0.08 |
| schema_link +correct       | 0.14 |
| schema_link +correct +cons | 0.16 |

## Findings
- Harness, EX evaluator, self-correct and self-consistency all work end-to-end.
- Recipe without linking is monotonic: 18 → 22 → 28 (+10 pts on a weak 7B).
- BUG (fixed): `link_schema` compared whole multi-word BIRD column names against
  single question tokens, so columns like `Free Meal Count (K-12)` never matched
  and the answer table got pruned → schema-link stages collapsed to 0%. Fixed by
  word-level sub-token matching (schema.py `_subtokens`). Stages recovered to 8–16%.
- Honest signal: on BIRD dev's small schemas, lexical schema-linking is net-negative
  vs full schema (16% vs 28%). Pruning can drop a needed table; small schemas don't
  need pruning. Worth reporting; revisit linking for wide-schema dbs only.

## Next
- Parallelize runner (currently sequential, one example at a time) so vLLM batches.
- Headline: Qwen2.5-Coder-32B on A100 80GB, full dev (1534), all stages.
