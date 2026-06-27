# Size frontier — Qwen2.5-Coder 7B / 14B / 32B on BIRD dev (full, n=1534)

All on-prem, vLLM, greedy (temp 0), concurrency 64, Execution Accuracy.
7B/14B on L40S 46GB (torch 2.6.0+cu124, vllm 0.8.5); 32B on A100 80GB
(torch 2.9.0+cu126, vllm 0.11.1). 4 cheap single-call stages (self-consistency
omitted — the 32B run showed it adds +0.13 pp for ~5x cost; see frontier-32b.md).

| stage          |   7B   |  14B   |  32B   |
|----------------|:------:|:------:|:------:|
| base           | 39.05% | 47.39% | 50.39% |
| +self_correct  | 42.50% | 48.70% | 51.63% |
| schema_link    | 39.31% | 45.37% | 48.83% |
| link +correct  | 40.87% | 47.46% | 50.65% |

## Findings
1. **Diminishing returns with size (base EX):** 7B 39.05 → 14B 47.39 (**+8.34 pp**)
   → 32B 50.39 (**+3.00 pp**). The 7B→14B jump is large; 14B→32B is much smaller.
   On BIRD, a 14B open model already captures most of what a 32B delivers on-prem.
2. **self_correct helps weaker models more:** +3.45 pp (7B), +1.31 pp (14B), +1.24 pp (32B).
   A cheap recipe component that partly compensates for smaller capacity.
3. **schema_link is dominated at every size:** it trails base on 14B (45.37 vs 47.39)
   and 32B (48.83 vs 50.39); on 7B it is roughly flat (39.31 vs 39.05) but the
   link+correct track still trails plain correct (40.87 vs 42.50). Lexical pruning
   drops needed tables on BIRD's narrow schemas. Recommendation holds across sizes:
   **base + self_correct, no schema-linking.**

Per-stage summaries + per-example predictions: `results/qwen25-coder-{7b,14b}.*.{summary.json,jsonl}`.
