# nl2sql-onprem-bench

## Description

Harness, per-item predictions and analysis scripts for the paper

> Vladimir Beskorovainyi. *How far do on-premises open LLMs get on Text-to-SQL? A cross-family
> size and technique frontier on BIRD.* Revised version (2026); an earlier version is
> [arXiv:2606.29733](https://arxiv.org/abs/2606.29733).

Release **v0.3.0** accompanies the revised paper. It corrects harness defects present in the
preprint and in v0.2.0, re-scores all released runs with the official BIRD comparison, adds
the re-runs used in the revised paper, and adds the scripts that regenerate every number,
table and figure. **All numbers in the preprint and in earlier versions of this README are
superseded**; see [Corrections in v0.3.0](#corrections-in-v030-relative-to-the-preprint-and-v020).

Archived on Zenodo, concept DOI [10.5281/zenodo.20952793](https://doi.org/10.5281/zenodo.20952793)
(always resolves to the latest archived release).

## Dataset information

- **Dataset:** BIRD development split, `dev_20240627` package: 1534 natural-language
  questions with gold SQL over 11 SQLite databases, with BIRD's external-knowledge evidence.
- **Source and terms:** third-party, distributed by its authors at
  <https://bird-bench.github.io/>; it is not included in this repository. Put it under
  `data/bird/` (`dev.json` and `dev_databases/<db>/<db>.sqlite`) to run the harness.
- **Released here:** per-item predictions of every run (`results/`), the per-item official
  re-scoring, run summaries with token counts, and every number of the paper
  (`results/paper/`). Recomputing the paper's numbers needs only these files, not BIRD.

## Methodology

What the benchmark is:

- **Data.** Full BIRD development split, `dev_20240627` package (n = 1534 questions over 11
  SQLite databases). BIRD is third-party and is not redistributed; obtain it from
  <https://bird-bench.github.io/>.
- **Prompt.** One zero-shot prompt: a minimal schema listing (every table with its column
  names; no types, keys or sample values; column names with spaces or parentheses unquoted),
  the question and the BIRD external-knowledge evidence. Greedy decoding (temperature 0)
  except for self-consistency sampling.
- **Recipe components** (ablated one at a time on top of a single call):
  - *lexical schema linking*: keep a whole table if any word-level sub-token of its name or of
    one of its column names occurs in the question or evidence; no key closure, no column
    pruning; full schema if nothing matches;
  - *embedding schema linking*: all-MiniLM-L6-v2, top k = 6 tables by cosine similarity
    (k is recorded in the run summaries; the code default is 5);
  - *execution-guided self-correction*: if the draft fails to execute or times out, the
    error is fed back for up to two repair rounds (the repair prompt omits the evidence);
  - *self-consistency*: five samples at temperature 0.6, each self-corrected, majority vote
    over execution results (Qwen2.5-Coder-32B self-hosted only).
- **Metric.** Execution accuracy (EX) with the official BIRD comparison
  `set(pred_rows) == set(gold_rows)`, on read-only connections (`file:...?mode=ro`) to
  pristine databases, 30 s timeout; a prediction that fails or times out is wrong. A scoring
  pass is the execution of gold and predicted queries from which a run's correctness is taken
  (the official re-scoring for the released runs, the harness evaluation for the re-runs);
  gold results were computed in every pass rather than cached, so a few slow gold queries
  timed out in some passes only. **Uniform gold-failure rule:** the 5 questions whose gold
  query failed or timed out in at least one scoring pass of any reported run (ids 518, 596,
  646, 701, 1014; 518 and 701 failed in every pass in which their gold query was executed,
  the others only in some, and each pass had 1 to 5 such items) are scored wrong in every
  run. None of them was scored correct in any run, so the rule changes no EX value; it keeps
  them out of every discordant pair. The maximum attainable EX is therefore 99.67%. Every
  configuration is scored on all 1534 items.
- **Statistics.** Paired exact McNemar tests on per-item correctness, Holm correction over
  all 30 within-model recipe comparisons (20 for Qwen2.5-Coder, Llama and OpenRouter, 10 for
  CodeLlama), and a database-clustered percentile bootstrap 95% CI (11 databases resampled,
  10,000 replicates, seed 0). Size, family, cross-serving and replicate comparisons are
  descriptive and not in the Holm family.

### Serving configurations

| Models | Serving | Hardware / software | Precision | Conc. | Max tokens |
|---|---|---|---|--:|--:|
| Qwen2.5-Coder-Instruct 7B, 14B | self-hosted vLLM | 1x L40S 48 GB; vLLM 0.8.5, torch 2.6 | 16-bit (checkpoint dtype, `dtype=auto`; not logged) | 64 | 1024 |
| Qwen2.5-Coder-Instruct 32B | self-hosted vLLM | 1x A100-SXM4 80 GB; vLLM 0.11.1, torch 2.9 | 16-bit (checkpoint dtype; not logged) | 64 | 1024 |
| CodeLlama-Instruct 7B, 13B (re-run) | self-hosted vLLM (Kaggle) | 2x Tesla T4 16 GB, tensor parallel 2; vLLM 0.31.0, eager mode, prefix caching, `max_model_len` 3072, `max_num_seqs` 64; client timeout 90 s with 2 retries; retry pass at concurrency 32, timeout 900 s | fp16 | 64 | 1024 |
| Llama-3.1-8B-Instruct, Llama-3.3-70B-Instruct | DeepInfra API | provider-managed | FP8 | 24 | 1024 |
| Qwen2.5-Coder-32B-Instruct (re-run) | OpenRouter API (single provider: Cloudflare) | provider-managed; whitespace repair of every completion | undisclosed | 16 | 4096 |

Results from the three serving blocks (self-hosted 16-bit, FP8 API, OpenRouter API) are not
directly comparable; comparisons across blocks are labelled cross-serving.

## Results

All values are computed by `analysis/paper_numbers.py` and stored in
[`results/paper/`](results/paper/). EX in %, BIRD dev, n = 1534, official set comparison on
read-only databases, uniform gold-failure rule. "Corr.": self-correction; "--": not run.

| Serving | Model | Base | + Corr. | Lexical linking | Lexical + Corr. | Embedding linking | Embedding + Corr. |
|---|---|--:|--:|--:|--:|--:|--:|
| Self-hosted, 16-bit | Qwen2.5-Coder-7B | 42.63 | 46.54 | 43.48 | 45.37 | -- | -- |
| Self-hosted, 16-bit | Qwen2.5-Coder-14B | 52.02 | 53.52 | 49.87 | 52.09 | -- | -- |
| Self-hosted, 16-bit | Qwen2.5-Coder-32B | 54.89 | 56.19 | 53.72 | 55.41 | -- | -- |
| Self-hosted, fp16 | CodeLlama-7B | 23.73 | 27.12 | 25.42 | 28.16 | 22.62 | -- |
| Self-hosted, fp16 | CodeLlama-13B | 26.27 | 30.64 | 26.34 | 29.92 | 25.68 | -- |
| API, FP8 (DeepInfra) | Llama-3.1-8B | 37.09 | 41.07 | 34.75 | -- | -- | -- |
| API, FP8 (DeepInfra) | Llama-3.3-70B | 54.69 | 55.67 | 50.46 | -- | -- | -- |
| API, OpenRouter (whitespace-repaired) | Qwen2.5-Coder-32B | 54.50 | 55.87 | 53.46 | -- | 54.11 | 55.35 |

Self-consistency (Qwen2.5-Coder-32B, self-hosted): 56.13 after self-correction and 55.54
after lexical linking and self-correction.

### Paired tests (Holm family, 30 comparisons)

Δ: difference in EX (percentage points); +/−: items gained/lost; p: exact McNemar;
p (Holm): adjusted over all 30 comparisons; CI: database-clustered bootstrap 95% interval.

| Model | Comparison | Δ | +/− | p | p (Holm) | Clustered 95% CI |
|---|---|--:|--:|--:|--:|--:|
| Qwen-7B | base -> +self-correction | +3.91 | 67/7 | 2.1e-13 | 5.9e-12 | [1.29, 8.17] |
| Qwen-7B | lexical linking -> +self-correction | +1.89 | 35/6 | 4.9e-6 | 1.1e-4 | [1.06, 2.75] |
| Qwen-7B | base -> lexical linking | +0.85 | 93/80 | 0.36 | 1.00 | [-2.92, 5.76] |
| Qwen-14B | base -> +self-correction | +1.50 | 26/3 | 1.5e-5 | 3.2e-4 | [0.94, 2.10] |
| Qwen-14B | lexical linking -> +self-correction | +2.22 | 35/1 | 1.1e-9 | 2.6e-8 | [1.50, 3.14] |
| Qwen-14B | base -> lexical linking | -2.15 | 55/88 | 0.0072 | 0.10 | [-5.61, 0.42] |
| Qwen-32B | base -> +self-correction | +1.30 | 27/7 | 8.2e-4 | 0.014 | [0.45, 2.06] |
| Qwen-32B | lexical linking -> +self-correction | +1.69 | 32/6 | 2.4e-5 | 4.9e-4 | [0.46, 3.24] |
| Qwen-32B | base -> lexical linking | -1.17 | 48/66 | 0.11 | 1.00 | [-3.19, 0.22] |
| Qwen-32B | +self-correction -> +self-consistency | -0.07 | 13/14 | 1.00 | 1.00 | [-0.47, 0.37] |
| Qwen-32B | lexical + self-correction -> +self-consistency | +0.13 | 18/16 | 0.86 | 1.00 | [-0.93, 1.37] |
| Llama-8B (FP8 API) | base -> +self-correction | +3.98 | 71/10 | 1.8e-12 | 4.7e-11 | [2.77, 5.25] |
| Llama-8B (FP8 API) | base -> lexical linking | -2.35 | 63/99 | 0.0058 | 0.087 | [-4.83, -0.63] |
| Llama-70B (FP8 API) | base -> +self-correction | +0.98 | 44/29 | 0.10 | 1.00 | [-0.20, 2.35] |
| Llama-70B (FP8 API) | base -> lexical linking | -4.24 | 37/102 | 3.2e-8 | 7.4e-7 | [-6.98, -1.98] |
| Qwen-32B (OpenRouter) | base -> +self-correction | +1.37 | 25/4 | 1.0e-4 | 0.0020 | [0.52, 2.10] |
| Qwen-32B (OpenRouter) | embedding linking -> +self-correction | +1.24 | 22/3 | 1.6e-4 | 0.0028 | [0.26, 2.19] |
| Qwen-32B (OpenRouter) | base -> lexical linking | -1.04 | 47/63 | 0.15 | 1.00 | [-3.10, 0.37] |
| Qwen-32B (OpenRouter) | base -> embedding linking | -0.39 | 33/39 | 0.56 | 1.00 | [-1.00, 0.25] |
| Qwen-32B (OpenRouter) | lexical -> embedding linking | +0.65 | 67/57 | 0.42 | 1.00 | [-1.14, 3.01] |
| CodeLlama-7B | base -> +self-correction | +3.39 | 55/3 | 2.3e-13 | 6.1e-12 | [2.36, 4.77] |
| CodeLlama-7B | lexical linking -> +self-correction | +2.74 | 43/1 | 5.1e-12 | 1.3e-10 | [1.79, 3.78] |
| CodeLlama-7B | base -> lexical linking | +1.69 | 90/64 | 0.044 | 0.57 | [-0.42, 3.36] |
| CodeLlama-7B | base -> embedding linking | -1.11 | 32/49 | 0.075 | 0.90 | [-2.02, -0.32] |
| CodeLlama-7B | lexical -> embedding linking | -2.80 | 59/102 | 8.8e-4 | 0.014 | [-4.47, -0.61] |
| CodeLlama-13B | base -> +self-correction | +4.37 | 69/2 | 2.2e-18 | 6.5e-17 | [2.65, 6.24] |
| CodeLlama-13B | lexical linking -> +self-correction | +3.59 | 56/1 | 8.0e-16 | 2.3e-14 | [2.32, 4.92] |
| CodeLlama-13B | base -> lexical linking | +0.07 | 63/62 | 1.00 | 1.00 | [-1.93, 2.11] |
| CodeLlama-13B | base -> embedding linking | -0.59 | 28/37 | 0.32 | 1.00 | [-2.41, 0.97] |
| CodeLlama-13B | lexical -> embedding linking | -0.65 | 64/74 | 0.44 | 1.00 | [-1.79, 0.87] |

Descriptive comparisons (size, family, cross-serving, replication, repair) are in
`results/paper/paper_tests_holm.csv` (`family == "descriptive"`). In brief:

- Qwen2.5-Coder base EX rises by +9.39 points from 7B to 14B (clustered CI 6.23 to 13.68) and
  by +2.87 points from 14B to 32B (p = 0.0038, clustered CI -0.15 to 5.63); the second step
  coincides with a change of GPU and vLLM version and is not separately attributable to size.
- After the whitespace repair, Qwen2.5-Coder-32B through OpenRouter and self-hosted differ by
  -0.39 points in the base configuration (2 gained / 8 lost, p = 0.11, clustered CI -0.75 to
  0.00). Without the repair (run v2) the gap was -3.32 points (2/53); of the 53 lost items,
  45 differed from the self-hosted prediction only in whitespace. Most of the apparent
  three-point serving gap was therefore this provider defect.
- The two unrepaired OpenRouter runs (v2, v3) agree within ±0.20 points per stage (p ≥ 0.38);
  the repair raised EX by 2.61 to 3.13 points per stage (v3 -> v4).

### Summary of findings

- Self-correction is the only component with a consistent gain. Within the self-correction
  run (repaired non-executing drafts that end correct; free of run-to-run noise) it adds
  +1.24 to +3.72 points over the 10 self-correction configurations of the Qwen2.5-Coder, Llama
  and OpenRouter runs, and +3.46 / +4.43 points for CodeLlama-7B / -13B from the base
  configuration (+2.67 to +4.43 over all four CodeLlama configurations); every clustered
  interval excludes zero. It costs 1.1 to 1.5 times the base tokens (1.82 / 1.93 times for
  CodeLlama, whose drafts fail more often). The base vs self-correction test is significant
  after Holm correction for every model except Llama-3.3-70B.
- Schema linking never improved EX significantly. Lexical linking reduced EX for
  Llama-3.3-70B (-4.24 points, significant after Holm). The embedding linker has higher
  gold-table recall (96.53% vs 93.95%) but did not differ detectably from no linking on the
  OpenRouter run; for CodeLlama-7B it was significantly worse than the lexical linker.
- Five-sample self-consistency (one model) changed EX by -0.07 and +0.13 points at about five
  times the tokens.
- A second greedy pass (the unrepaired items of a self-correction run) changed the correctness
  of at least 0.20–2.54% of items for Qwen2.5-Coder self-hosted, 0.07–0.26% for CodeLlama,
  0.33–0.39% for the OpenRouter run and 1.76–3.65% through the FP8 API. These are lower bounds,
  so differences below about one point should not be interpreted.
- No dollar costs are reported; tokens and model calls per question are in
  `results/paper/paper_usage.csv`.

## Corrections in v0.3.0 relative to the preprint and v0.2.0

The preprint (arXiv:2606.29733) and releases v0.1.0/v0.2.0 were affected by the following
defects. All are corrected in this release, and the README of v0.2.0 (its tables, the
"headline" numbers, the dollar-cost table and the findings) is withdrawn.

1. **Read-write databases.** The harness opened the BIRD databases read-write, so
   data-definition statements emitted by models (e.g. `ALTER TABLE ... ADD COLUMN`) executed
   and persisted; 68 final CodeLlama predictions contained such statements, and the
   CodeLlama-13B and -34B runs saw altered schemas. All scoring and re-runs now use read-only
   connections (`src/nl2sql_bench/evaluator.py`, `schema.py`).
2. **Comparison stricter than official BIRD.** The evaluator compared sorted multisets of row
   representations. All EX values are now computed with the official set comparison on
   pristine databases (`results/official_rescoring/`), which raises the EX of the 20 retained
   released runs by 3.6 to 5.5 points.
3. **Truncated FP8 API arm for Qwen2.5-Coder-32B; ~13-point claim withdrawn.** In the v1 FP8
   API run of Qwen2.5-Coder-32B, 508 of 1534 completions were truncated at 1024 tokens of
   chain-of-thought prose. This arm (`results/qwen25-coder-32b-api.*` and, despite their
   names, `results/qwen25-coder-32b.link_emb*`) is invalid and is replaced by the OpenRouter
   re-run (max 4096 tokens, no truncated completions). The earlier claim of a quantization
   penalty of about 13 points is **withdrawn**, as are the lexical-vs-embedding linking numbers
   that used this arm.
4. **SQL extractor.** The extractor could start inside prose. `extract_sql`
   (`src/nl2sql_bench/pipeline.py`) now takes the last fenced block, else the first line
   starting with `SELECT` or `WITH`, else the first `SELECT` or CTE header. By a heuristic
   count the old extractor affected 0 to 5 items per released run; the retained released runs
   are re-scored as produced, and all re-runs use the new extractor. Known remaining
   limitation: the new extractor can still start inside prose when a prose line begins with
   "With" or "Select", and it takes the last fenced block even if that block is not the
   answer; no material effect was found on the re-runs.
5. **Embedder race.** `codellama-34b.link_emb` lost 61 items to a race when the embedding
   model was loaded lazily from worker threads. The embedder is now loaded and used under a
   lock (`src/nl2sql_bench/schema.py`), and the re-run scripts load it once before the
   worker threads start.
6. **CodeLlama re-run; CodeLlama-34B excluded.** Because of defect 1, the preprint CodeLlama
   results are replaced by clean re-runs of the 7B and 13B models
   (`results/reruns/codellama_clean/`). CodeLlama-34B could not be re-run on the hardware
   available and is no longer reported; its released predictions are kept for transparency
   only.
7. **Uniform gold-failure rule.** Items with a failing or timed-out gold query are now scored
   wrong by one rule in every run (see Metric above).
8. **OpenRouter whitespace defect and repair.** The only OpenRouter provider for
   Qwen2.5-Coder-32B (Cloudflare) drops spaces adjacent to digits (`> 500 AND` arrives as
   `>500AND`, `LIMIT 1` as `LIMIT1`). In two unrepaired runs of all five stages (v2, v3), 81
   to 88 final predictions per stage had a digit glued to an upper-case SQL keyword, and none
   of them executed. The reported OpenRouter arm (v4) applies `repair_ws`
   (`src/nl2sql_bench/textfix.py`) to every completion before SQL extraction: outside quoted
   literals and identifiers, a space is inserted between a digit and an adjacent upper-case
   SQL keyword. It modified 97 to 120 completions per stage; no final v4 prediction contains a
   glued digit and keyword, and no OpenRouter call failed. In v3 the repair step was
   inadvertently a no-op; v2 and v3 are released as an unrepaired replication.
9. **CodeLlama client-timeout retry.** In the first CodeLlama pass (concurrency 64 on two T4
   GPUs, 90 s client timeout) 7 to 331 requests per run (832 in total, most in
   `codellama-13b-clean.correct`) timed out on the client. These items were re-issued with
   the same prompts, greedy decoding and harness at concurrency 32 with a 900 s timeout, and
   each retried record replaced the first-pass record (`analysis/merge_codellama_retry.py`).
   All other retried items completed. One item (qid 1086 in `codellama-13b-clean.correct`)
   still failed: its draft reached the 1024-token completion limit, and the resulting repair
   prompt (2049 input tokens plus 1024 requested output tokens) exceeded the 3072-token
   context limit (`max_model_len`) of the server, which rejected it with HTTP 400 in all three
   retry attempts. It is scored wrong.

As a consequence, two significance claims of the preprint no longer hold: after correction,
lexical linking does not significantly reduce EX for Qwen2.5-Coder-32B or Llama-3.1-8B.

### Status of the released (preprint) runs in `results/*.jsonl`

The released prediction and summary files are kept unchanged.

| files | status in the revised paper |
|---|---|
| `qwen25-coder-{7b,14b,32b}.{base,correct,link,link_correct}`, `qwen25-coder-32b.{consistency,link_correct_consistency}`, `llama-3.1-8b.{base,correct,link}`, `llama-3.3-70b.{base,correct,link}` (20 runs) | used, re-scored with the official comparison |
| `qwen25-coder-32b-api.{base,link}`, `qwen25-coder-32b.{link_emb,link_emb_correct}` | invalid (truncated FP8 API arm), not used |
| `codellama-{7b,13b,34b}.*` | not used (database alteration); 7B and 13B replaced by `results/reruns/codellama_clean/` |
| `ollama-gemma3-12b.*` | 20-item smoke test, not an evaluation |

The `correct` field and `ex_accuracy` in these files are the preprint values (multiset
comparison, read-write databases); use `results/official_rescoring/official_per_item.csv`
or `results/paper/paper_ex.csv` instead. The released Qwen2.5-Coder self-hosted runs were
produced with an earlier harness revision (repair not triggered by timeouts, old extractor,
no wall-clock logging); they are re-scored, not regenerated.

## Code information

Repository layout:

```
src/nl2sql_bench/
  data.py          load BIRD dev (questions, gold SQL, SQLite paths)
  schema.py        schema serialization, lexical and embedding schema linking (read-only, embedder lock)
  model_client.py  OpenAI-compatible client (vLLM or API)
  pipeline.py      prompt templates, extract_sql, self-correction, self-consistency
  evaluator.py     execution accuracy: official set comparison on read-only connections
  textfix.py       repair_ws: whitespace repair for the OpenRouter provider defect
  runner.py        runs one configuration, writes <run>.jsonl and <run>.summary.json
scripts/run_bench.py   CLI entry point
configs/run.yaml       model endpoints and run configuration
tests/                 unit tests of the harness fixes and the whitespace repair
results/*.jsonl, *.summary.json   released preprint runs (unchanged)
results/official_rescoring/       per-item official re-scoring of the released runs
results/reruns/                   v0.3.0 re-runs: openrouter_v2, openrouter_v3, openrouter_v4 (reported), codellama_clean
results/paper/                    every number of the revised paper (CSV, log, LaTeX macros) + inputs/
results/archive/                  historical notes from the preprint (superseded)
analysis/                         paper_numbers.py, make_figures.py, merge_codellama_retry.py
figures/                          Figure1-4.png of the revised paper
kaggle/                           Kaggle scripts that produced the re-runs and the re-scoring
```

## Requirements

- Python 3.10 or later.
- Harness: `openai>=1.30`, `pyyaml>=6.0`, `tqdm>=4.66`, `func-timeout>=4.3`
  (`pip install -e .`); tests: `pytest>=8.0` (`.[dev]`).
- Embedding schema linking: `sentence-transformers>=2.7`, `numpy>=1.24` (`.[emb]`).
- Recomputing the paper: `numpy`, `pandas`, `scipy`, `matplotlib`, `pillow` (`.[analysis]`);
  no GPU, model or BIRD download needed.
- Running models: an OpenAI-compatible endpoint (for example vLLM) or an API key; the serving
  configurations used in the paper are listed above.

## Usage

How to reproduce:

### 1. Recompute every number, table and figure from the released predictions

No GPU, model or BIRD download is needed.

```bash
pip install numpy pandas scipy matplotlib pillow     # or: pip install -e ".[analysis]"
python3 analysis/merge_codellama_retry.py   # optional: regenerates the merged CodeLlama files (byte-identical)
python3 analysis/paper_numbers.py           # writes results/paper/* and figures/Figure1-4.png
```

`paper_numbers.py` is deterministic (bootstrap seed 0). Options: `--no-figures` skips the
figures; `--update-tex PATH` replaces the block between `% BEGIN AUTONUMBERS` and
`% END AUTONUMBERS` in the manuscript at `PATH` (not needed otherwise).

### 2. Run the harness

```bash
pip install -e ".[dev,emb]"
PYTHONPATH=src python3 -m pytest -q tests
# put BIRD dev under data/bird/ (dev.json + dev_databases/<db>/<db>.sqlite)
# serve a model with an OpenAI-compatible endpoint, e.g.
#   vllm serve Qwen/Qwen2.5-Coder-32B-Instruct --port 8000
python scripts/run_bench.py --config configs/run.yaml --model qwen25-coder-32b --stages base correct
```

New runs are written to `runs/` (`out_dir` in `configs/run.yaml`; git-ignored). Use `--emb-top-k 6` for the embedding-linking stages
as in the paper. Regenerating runs from scratch is subject to serving nondeterminism
(greedy decoding under batched inference is not bit-reproducible), so re-runs will differ
from the released predictions by about the run-to-run variation reported in the paper.

### 3. Kaggle re-runs

The OpenRouter and CodeLlama re-runs and the official re-scoring were run in Kaggle
notebooks with the scripts in [`kaggle/`](kaggle/README.md), which also lists which script
produced which folder. The OpenRouter key is read from a Kaggle secret
(`OPENROUTER_API_KEY`).

## Citations

Paper:

```bibtex
@misc{beskorovainyi2026nl2sqlonprem,
  title  = {How far do on-premises open {LLMs} get on {Text-to-SQL}? A cross-family size and technique frontier on {BIRD}},
  author = {Beskorovainyi, Vladimir},
  year   = {2026},
  note   = {Revised version; an earlier version is arXiv:2606.29733},
  url    = {https://arxiv.org/abs/2606.29733}
}
```

Code and data (see also `CITATION.cff`):

```bibtex
@software{beskorovainyi2026nl2sqlonprembench,
  title     = {nl2sql-onprem-bench},
  author    = {Beskorovainyi, Vladimir},
  year      = {2026},
  version   = {0.3.0},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20952793},
  url       = {https://github.com/beskvladimir-create/nl2sql-onprem-bench}
}
```

Author: Vladimir Beskorovainyi, Besk Tech (ORCID 0009-0004-7005-6242). The DOI above is the
concept DOI and resolves to the latest archived release.

## License and contribution

MIT (see `LICENSE`). The license covers the harness, scripts and result files in this
repository. The BIRD dataset is not included; it is distributed by its authors under their
own terms (<https://bird-bench.github.io/>). No client data, prompts or systems are included.

Contributions are welcome as GitHub issues or pull requests.
