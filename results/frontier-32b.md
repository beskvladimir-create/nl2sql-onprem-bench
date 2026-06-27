# Accuracy frontier — Qwen2.5-Coder-32B on BIRD dev (full, n=1534)

A100-80GB, vLLM 0.11.1, torch 2.9.0+cu126, greedy (temp 0), concurrency 64.
Per-query usage is recorded in each `*.jsonl` (`usage`: model_calls, prompt/completion
tokens, latency_s). Latency is the summed model-call time per query (overlapped under
concurrency, so wall-clock per query is lower).

| stage              | EX %  | Δ vs prev | calls/q | compl tok/q | total tok/q | lat p50 (s) | lat p95 (s) |
|--------------------|:-----:|:---------:|:-------:|:-----------:|:-----------:|:-----------:|:-----------:|
| base               | 50.39 |     –     |  1.00   |     54      |     421     |    3.74     |    7.27     |
| +self_correct      | 51.63 |  +1.24    |  1.10   |     64      |     472     |    3.90     |   10.64     |
| +self_consistency  | 51.76 |  +0.13    |  5.48   |    318      |    2358     |   17.64     |   50.02     |
| schema_link        | 48.83 |  −1.56    |  1.00   |     54      |     355     |    3.73     |    7.56     |
| link +correct      | 50.65 |     –     |  1.12   |     68      |     412     |    3.92     |   13.14     |
| link +corr +cons   | 50.98 |     –     |  5.61   |    335      |    2056     |   18.53     |   64.05     |

Total tokens per stage (1534 queries): base 646k · +correct 724k · +cons **3.62M** ·
link 545k · link+correct 632k · link+corr+cons **3.15M**.

## The frontier insight (the article's spine)
1. **self_correct is the sweet spot.** +1.24 pp EX for only ~12 % more tokens and ~0.1 call/query.
   A near-free win for any on-prem deployment.
2. **self_consistency is not worth it on BIRD.** It buys **+0.13 pp** over self_correct while
   costing **~5× tokens** (724k → 3.62M), **~5× model calls** (1.10 → 5.48/query) and
   **~4.5× tail latency** (p95 10.6 s → 50 s). Terrible ROI — the consistency vote barely
   moves a deterministic-at-temp-0 + corrected model.
3. **Lexical schema-linking is strictly dominated.** It lowers EX (−1.56 pp at base) *and*
   the token saving is small (646k → 545k, ~16 %). Pruning drops needed tables on BIRD's
   narrow schemas. Don't pay accuracy for a minor token cut.

**Practical recommendation:** deploy **base + self_correct only** → 51.63 % EX at ~470 tok/query
and ~4 s p50. Skip schema-linking and self-consistency on BIRD-class schemas.

## Cost note
Whole 6-stage ablation (9,204 query-evaluations) ran in ~66 min wall on one A100-80GB.
At a representative RunPod rate (~$1.8/GPU-hr) that is ~$2 total. Per-stage cost scales
with total tokens above; plug your own $/GPU-hr. (Exact per-stage wall time not logged;
token totals are the faithful relative-compute proxy.)
