# Official BIRD re-scoring of the released runs

`official_per_item.csv` re-scores every released prediction file in `results/*.jsonl`
(the original preprint runs, unchanged) on pristine copies of the BIRD dev databases
opened read-only (`file:...?mode=ro`), with a 30 s execution timeout. No model was
called; the predictions are scored exactly as they were produced.

Produced by `kaggle/sql_official_eval.py` (per-item JSON per file) and merged into this
CSV by `kaggle/sql_merge.py`. Read by `analysis/paper_numbers.py`.

| column | meaning |
|---|---|
| `file` | released prediction file, e.g. `qwen25-coder-32b.base.jsonl` |
| `qid` | BIRD dev question index (0..1533) |
| `released` | 1/0: the `correct` field as recorded in the released file (preprint harness: sorted-multiset comparison, read-write databases) |
| `multiset` | 1/0: re-scored with the preprint comparison (sorted multiset of row representations) on read-only pristine databases |
| `set` | 1/0: re-scored with the official BIRD comparison `set(pred_rows) == set(gold_rows)` on read-only pristine databases; this is the value used in the paper |

`multiset` and `set` are empty (NaN when read with pandas) when the gold query failed or
timed out in that scoring pass. Such questions are handled by the uniform gold-failure
rule of the paper: any question whose gold query failed in at least one scoring pass of
any reported run (question ids 518, 596, 646, 701, 1014) is scored wrong in every run.

40 files are covered: 38 with all 1534 items and the two 20-item `ollama-gemma3-12b`
smoke-test files. Which released runs are used in the paper and which are excluded is
listed in the top-level README.
