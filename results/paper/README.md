# Numbers of the revised paper

Everything in this folder (except `inputs/`) is written by `python3 analysis/paper_numbers.py`
(run from the repository root; deterministic, bootstrap seed 0).

| file | content |
|---|---|
| `paper_ex.csv` | EX (%) and number correct for every run used in the paper, official set comparison, after the uniform gold-failure rule |
| `paper_tests_holm.csv` | every paired comparison: the Holm family (30 exact McNemar tests: 20 main, 10 CodeLlama; `family` = `main` / `codellama`) and the descriptive comparisons (`family` = `descriptive`), with gained/lost items, p, Holm-adjusted p, Wald and database-clustered bootstrap 95% CIs |
| `paper_usage.csv` | per run: mean prompt/completion/total tokens, model calls, p50/p95 summed model-call time, failure categories, wall clock where meaningful |
| `paper_numbers.txt` | full printed log, including the whitespace-defect counts, CodeLlama retry statistics, self-correction mechanism, linker recall and contamination probe |
| `autonumbers.tex` | LaTeX macro block with the tables and values used in the manuscript |

`inputs/` holds two per-run fact tables for the released (preprint) runs that the script
reads: `wall_clock.csv` (logged wall clock and concurrency) and
`invalid_items_per_file.csv` (per-run item categories, including the heuristic count of
predictions affected by the old SQL extractor and the embedder-race failures). They were
computed from the released prediction and summary files with an earlier recomputation
script and are included as data.
