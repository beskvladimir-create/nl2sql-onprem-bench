# nl2sql-onprem-bench

Honest, reproducible benchmark of **on-prem open-weight LLMs** for Text-to-SQL.

Question: how close can a privately deployable open model get to cloud SOTA on
[BIRD](https://bird-bench.github.io/), under a fixed GPU budget and with zero
data egress — and how much of the gap does a lightweight, model-agnostic recipe
(schema linking → self-correction → self-consistency) close, at what cost?

We report not just Execution Accuracy (EX) but the **accuracy frontier**:
EX vs cost/query, latency (p50/p95) and VRAM. No cherry-picking, failures included.

## Status
Results available. Full BIRD dev (n=1534) runs for Qwen2.5-Coder 7B / 14B / 32B
(plus a gemma3-12b cross-family point) under a matched protocol; see `results/`.
Harness is GPU-ready — point `configs/run.yaml` at a vLLM (or Ollama) endpoint.

## Results (BIRD dev, n=1534, Execution Accuracy, greedy)

Size x technique frontier (on-prem, vLLM):

| stage          |   7B   |  14B   |  32B   |
|----------------|:------:|:------:|:------:|
| base           | 39.05% | 47.39% | 50.39% |
| + self_correct | 42.50% | 48.70% | 51.63% |
| schema_link    | 39.31% | 45.37% | 48.83% |
| link + correct | 40.87% | 47.46% | 50.65% |

Headline (32B, full ablation incl. self-consistency): **51.76% EX**
(base 50.39 -> +self_correct 51.63 -> +self_consistency 51.76).

Findings (details in `results/size-frontier.md`, `results/headline-32b-a100.md`):
- **Diminishing returns with size:** 7B->14B is +8.34 pp; 14B->32B only +3.00 pp.
- **self_correct helps weaker models more** (+3.45 pp on 7B vs +1.24 pp on 32B).
- **Lexical schema-linking is net-negative** on BIRD's narrow schemas (trails base at every size).
- **self-consistency is not worth it** (+0.13 pp for ~5x cost).

## Layout
```
src/nl2sql_bench/
  data.py        load BIRD dev (questions + gold SQL + sqlite dbs)
  schema.py      schema serialization + lightweight schema linking
  model_client.py  vLLM (OpenAI-compatible) + cloud reference client
  pipeline.py    the recipe: generate -> self-correct -> self-consistency
  evaluator.py   Execution Accuracy (set comparison against gold)
  runner.py      orchestrates a run, writes results.jsonl
scripts/run_bench.py   CLI entrypoint
configs/               models + run config
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
Primary metric EX (official set comparison); also cost/query, latency, VRAM.

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
If you use this benchmark or code, please cite it (see `CITATION.cff`). A
preprint describing the study is forthcoming on arXiv; this README will be
updated with the arXiv identifier and DOI on release.

## IP / scope
New code, public BIRD data only. No client data, prompts, or systems.
Author: Vladimir Beskorovainyi (ORCID 0009-0004-7005-6242 / besk.tech).
License: MIT (see `LICENSE`).
