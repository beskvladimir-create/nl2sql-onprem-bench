# Headline — Qwen2.5-Coder-32B-Instruct on BIRD dev (full, 1534)

Hardware: 1× A100-SXM4-80GB (RunPod), vLLM 0.11.1, torch 2.9.0+cu126 (driver 565.57 / CUDA 12.7).
Metric: Execution Accuracy (EX) on the full BIRD dev split, n=1534, greedy (temp 0), concurrency 64.
Run: 2026-06-26, ~66 min wall (model load ~4 min + ablation).

| stage                          | schema_link | self_correct | self_consistency | EX (1534) |
|--------------------------------|:-----------:|:------------:|:----------------:|:---------:|
| base                           |      –      |      –       |        –         |  50.39 %  |
| +self_correct                  |      –      |      ✓       |        –         |  51.63 %  |
| +self_consistency              |      –      |      ✓       |        ✓         |**51.76 %**|
| schema_link                    |      ✓      |      –       |        –         |  48.83 %  |
| schema_link +correct           |      ✓      |      ✓       |        –         |  50.65 %  |
| schema_link +correct +cons     |      ✓      |      ✓       |        ✓         |  50.98 %  |

## Findings
- **Headline 51.76 % EX** (base → self_correct → self_consistency) — honest, reproducible number
  for a 32B open model served on-prem, sitting in the gap between the cloud BIRD ceiling (~80 %)
  and weak open models (~50–71 %).
- **The no-linking recipe is monotonic:** 50.39 → 51.63 → 51.76. self_correct adds +1.24 pp,
  self_consistency a further +0.13 pp.
- **Lexical schema-linking is net-negative on BIRD's narrow schemas:** base 50.39 % vs link 48.83 %
  (−1.56 pp), and every linking-track stage trails its no-link counterpart
  (correct 51.63 vs 50.65; +cons 51.76 vs 50.98). Pruning drops needed tables on small schemas.
  This confirms the earlier 7B smoke finding — revisit linking only for wide-schema DBs.

Per-stage summaries: `results/qwen25-coder-32b.<stage>.summary.json`.
Per-example predictions (`*.jsonl`) remain on the pod; fetch before terminating if needed.
