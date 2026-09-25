# nl2sql-onprem-bench

Honest, reproducible benchmark of **on-prem open-weight LLMs** for Text-to-SQL.

Question: how close can a privately deployable open model get to cloud SOTA on
[BIRD](https://bird-bench.github.io/), under a fixed GPU budget and with zero
data egress, and how much of the gap does a lightweight, model-agnostic recipe
(schema linking → self-correction → self-consistency) close, at what cost?

We report not just Execution Accuracy (EX) but the **accuracy frontier**:
EX vs cost/query, latency (p50/p95) and VRAM. No cherry-picking, failures included.
Every claim is tested with the paired exact McNemar test.

**Paper:** *How Far Do On-Prem Open LLMs Get on Text-to-SQL? A Cross-Family
Size×Technique Frontier on BIRD*, [arXiv:2606.29733](https://arxiv.org/abs/2606.29733)
(29 Jun 2026). Archived code/results: **Zenodo DOI
[10.5281/zenodo.20952793](https://doi.org/10.5281/zenodo.20952793)**
(concept DOI, always resolves to the latest archived release).

## Status
Results available for **three model families across two generations**, all on full
BIRD dev (n=1534) under a matched protocol:

- **Qwen2.5-Coder** 7B / 14B / 32B (on-prem, vLLM, fp16)
- **CodeLlama-Instruct** 7B / 13B / 34B (on-prem, vLLM, fp16)
- **Llama-3.x** 8B / 70B (matched FP8 API serving, relative claims only)

Per-example predictions and per-stage summaries for every run are in `results/`.
A `gemma3-12b` run is a 20-example smoke check only, not a full evaluation.
Harness is GPU-ready, point `configs/run.yaml` at a vLLM (or Ollama) endpoint.

## Results (BIRD dev, n=1534, Execution Accuracy, greedy)

### Size × technique frontier, on-prem fp16 (vLLM)

| family | size | base | +self_correct | schema_link (lexical) | link+correct |
|---|---|:--:|:--:|:--:|:--:|
| Qwen2.5-Coder | 7B | 39.05 | 42.50 | 39.31 | 40.87 |
| Qwen2.5-Coder | 14B | 47.39 | 48.70 | 45.37 | 47.46 |
| Qwen2.5-Coder | 32B | 50.39 | **51.63** | 48.83 | 50.65 |
| CodeLlama-Instruct | 7B | 20.93 | 23.99 | 22.23 | 23.73 |
| CodeLlama-Instruct | 13B | 21.64 | 23.01 | 20.60 | 22.03 |
| CodeLlama-Instruct | 34B | 24.51 | 27.12 | 23.79 | 26.66 |

Headline (Qwen 32B, full ablation incl. self-consistency): **51.76% EX**
(base 50.39 → +self_correct 51.63 → +self_consistency 51.76).

### Third family, modern generation (matched FP8 API serving)

Reported on its own serving, not mixed with the fp16 numbers above (absolute EX is
~13 pp lower under FP8; Qwen-32B base on this serving = 37.55 for reference).

| model (FP8 API) | base | +self_correct | schema_link (lexical) |
|---|:--:|:--:|:--:|
| Llama-3.1-8B-Instruct | 32.92 | 36.57 | 31.16 |
| Llama-3.3-70B-Instruct | 49.22 | 50.26 | 45.57 |

### Lexical vs embedding schema linking

Embedding linker: all-MiniLM-L6-v2, top-6 tables by cosine, tuned to ≥95% gold-table
recall (**96.5%** achieved vs 94.0% lexical).

| model | base | link (lexical) | link_emb (retrieval) |
|---|:--:|:--:|:--:|
| CodeLlama-34B (on-prem fp16) | 24.51 | 23.79 | 23.34 |
| Qwen2.5-Coder-32B (FP8 API) | 37.55 | 36.57 | 38.20 |

### Significance (paired exact McNemar)

| comparison | Δ pp | p |
|---|:--:|:--:|
| CodeLlama-7B: base → +self_correct | +3.06 | 2e-11 |
| CodeLlama-34B: base → +self_correct | +2.61 | 4.2e-09 |
| Qwen-32B: base → +self_correct | +1.24 | 0.0013 |
| Llama-3.1-8B: base → +self_correct | +3.65 | 6e-11 |
| Llama-3.3-70B: base → +self_correct | +1.04 | 0.08 (n.s.) |
| Qwen-32B: base → schema_link (lexical) | −1.56 | 0.035 |
| Llama-3.1-8B: base → schema_link (lexical) | −1.76 | 0.03 |
| Llama-3.3-70B: base → schema_link (lexical) | −3.65 | 1.4e-06 |
| CodeLlama-34B: base → schema_link (lexical) | −0.72 | 0.29 (n.s.) |
| CodeLlama-34B: lexical → embedding link | −0.45 | 0.55 (n.s.) |
| Qwen-32B: +self_correct → +self_consistency | +0.13 | 0.86 (n.s.) |

### Real cost (wall-clock, not token proxy), $/1k-queries @ $1.8/GPU-hr

| stage (CodeLlama-34B) | wall_s | $/1k-q |
|---|:--:|:--:|
| base | 376 | 0.123 |
| correct | 658 | 0.215 |
| link (lexical) | 372 | 0.121 |
| link_emb | 412 | 0.134 |
| link + correct | 692 | 0.225 |

## Findings

Details in `results/journal-upgrade.md`, `results/size-frontier.md`,
`results/headline-32b-a100.md`.

1. **Generation matters more than raw size, and the recipe is family-robust.**
   Qwen2.5-Coder dominates the older CodeLlama at matched size (7B: 39.1 vs 20.9),
   but a modern non-Qwen model (Llama-3.3-70B, 49.2 on a matched serving) is
   competitive, so CodeLlama's weakness reflects its 2023 generation, not
   "non-Qwen = weak". Within a family, returns diminish: 7B→14B is +8.34 pp,
   14B→32B only +3.00 pp.
2. **Self-correction is a robust, near-free win**, significant on all three families
   where there is room to improve, and larger on weaker models (+3.45 pp on Qwen-7B
   vs +1.24 pp on Qwen-32B).
3. **Schema linking does not help, and a stronger linker does not rescue it.**
   Lexical linking significantly *hurts* on Qwen-32B and on both Llama sizes; an
   embedding retriever at 96.5% gold-table recall is statistically indistinguishable
   from no linking (Qwen-32B FP8: base/lexical/embedding all within noise, p ≥ 0.09)
   and from the lexical linker (CodeLlama-34B p = 0.55). This rules out the
   "weak lexical strawman" objection across three families.
4. **Self-consistency is poor value:** +0.13 pp for ~5× tokens (p = 0.86), with p95
   latency going 10.6 s → 50.0 s.

### Scope of the linking result

The linking finding is **scoped, not universal**. On BIRD's relatively narrow schemas
the prompt is rarely the bottleneck, so pruning buys little context and risks dropping
a needed table. On genuinely wide schemas (enterprise warehouses with hundreds of
tables, where the full schema does not fit the context at all) linking is a
feasibility requirement rather than an accuracy technique, and we expect it to help.
See the paper's Discussion and Limitations (iii). Measuring that regime is future work.

## Layout
```
src/nl2sql_bench/
  data.py        load BIRD dev (questions + gold SQL + sqlite dbs)
  schema.py      schema serialization + lexical and embedding schema linking
  model_client.py  vLLM (OpenAI-compatible) + cloud reference client
  pipeline.py    the recipe: generate -> self-correct -> self-consistency
  evaluator.py   Execution Accuracy (set comparison against gold)
  runner.py      orchestrates a run, writes results.jsonl
scripts/run_bench.py   CLI entrypoint
configs/               models + run config
results/               per-example predictions (.jsonl) + per-stage summaries
```

## Quickstart (once GPU is up)
```bash
uv venv && uv pip install -e .
# 1. serve a model on the GPU box:
#    vllm serve Qwen/Qwen2.5-Coder-32B-Instruct --port 8000
# 2. put BIRD dev under data/bird/ (dev.json + dev_databases/)
python scripts/run_bench.py --config configs/run.yaml --model qwen25-coder-32b
```

## Method
Ablation per recipe component on each model:
`base -> +schema_link -> +self_correct -> +self_consistency`.
Zero-shot, single greedy call, no in-context examples, no fine-tuning.
Primary metric EX (official set comparison); also cost/query, latency, VRAM.
All pairwise differences tested with the paired exact McNemar test on per-example
correctness.

## Data (BIRD)
The BIRD dataset is **not** included in this repository (it is large and is
distributed by its authors). Download the BIRD **dev** split from
https://bird-bench.github.io/ and place it as:
```
data/bird/dev.json
data/bird/dev_databases/<db_name>/<db_name>.sqlite
```
Then run the harness as in Quickstart. Per-example predictions for our runs are
included under `results/*.jsonl`; per-stage summaries under `results/*.summary.json`.

## Citation
Paper:
```bibtex
@misc{beskorovainyi2026nl2sqlonprem,
  title  = {How Far Do On-Prem Open LLMs Get on Text-to-SQL?
            A Cross-Family Size x Technique Frontier on BIRD},
  author = {Vladimir Beskorovainyi},
  year   = {2026},
  eprint = {2606.29733},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url    = {https://arxiv.org/abs/2606.29733}
}
```
Code and results: see `CITATION.cff`, archived at Zenodo DOI
[10.5281/zenodo.20952793](https://doi.org/10.5281/zenodo.20952793).
That is the concept DOI and it always points at the newest release. The v0.1.0
snapshot (10.5281/zenodo.20952794) carries the Qwen2.5-Coder family only; the
CodeLlama and Llama-3.x results landed in v0.2.0.

## IP / scope
New code, public BIRD data only. No client data, prompts, or systems.
Author: Vladimir Beskorovainyi (ORCID 0009-0004-7005-6242 / besk.tech).
License: MIT (see `LICENSE`).
