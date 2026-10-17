# Kaggle scripts used for the v0.3.0 re-runs and re-scoring

These are the scripts as they were run in Kaggle notebook sessions (October 2026), kept
verbatim for provenance. Each script

- clones this repository from GitHub (the then-current v0.2.0 state) into `/kaggle/working`,
- overwrites `src/nl2sql_bench/evaluator.py`, `pipeline.py` and `schema.py` with the
  corrected versions embedded in the script as base64 (read-only databases, official set
  comparison, new `extract_sql`, embedder lock). The embedded files are identical to the
  files in this release, except for a corrected docstring in `schema.py`,
- downloads the BIRD dev package from the BIRD authors' URL (BIRD is not redistributed here).

The OpenRouter key is read at run time from a Kaggle secret by name
(`UserSecretsClient().get_secret("OPENROUTER_API_KEY")`); no key is stored in the scripts.
When run as an interactive session, the run scripts (except `codellama_retry.py`) limit
each stage to 8 items; the reported runs were full runs (n = 1534).

| script | produced | released as |
|---|---|---|
| `sql_official_eval.py` | per-item official (set) and multiset re-scoring of every released `results/*.jsonl` on read-only databases, no model calls | input to `sql_merge.py` |
| `sql_merge.py` | merges the per-item JSON files of `sql_official_eval.py` into one CSV | `results/official_rescoring/official_per_item.csv` |
| `sql_rerun.py` | OpenRouter v2: Qwen2.5-Coder-32B-Instruct, max_tokens 4096, five stages, no whitespace repair (tag `qwen25-coder-32b-or`) | `results/reruns/openrouter_v2/` |
| `sql_rerun_ws.py` | OpenRouter v3: same configuration with a whitespace-repair step that turned out to be a no-op, plus sequential official re-scoring (`*.rescored.json`, `gold_failed.json`) (tag `qwen25-coder-32b-orws`) | `results/reruns/openrouter_v3/` |
| `sql_rerun_ws4.py` | OpenRouter v4: same configuration with the working whitespace repair applied to every completion (tag `qwen25-coder-32b-orws4`); **the reported OpenRouter arm** | `results/reruns/openrouter_v4/` |
| `codellama_rerun.py` | CodeLlama-Instruct 7B and 13B, vLLM, fp16, 2x Tesla T4 (tensor parallel 2), five stages, concurrency 64, client timeout 90 s (tags `codellama-{7b,13b}-clean`) | `results/reruns/codellama_clean/raw/` |
| `codellama_retry.py` | retry pass: re-issues only the items that timed out on the client in the first pass, concurrency 32, client timeout 900 s | `results/reruns/codellama_clean/retry/` |

The merged CodeLlama files (`results/reruns/codellama_clean/<tag>.jsonl`, `retry_log.csv`)
are produced locally by `analysis/merge_codellama_retry.py`.

Notes:

- The module docstrings of `codellama_rerun.py` and `codellama_retry.py` partly repeat the
  header of `sql_rerun.py` (OpenRouter, max_tokens 4096); that text does not describe the
  CodeLlama runs. The code is authoritative: local vLLM server, max_tokens 1024.
- The whitespace repair in `sql_rerun_ws4.py` is the same function as
  `src/nl2sql_bench/textfix.py:repair_ws`, defined inline because the scripts patch only
  the three harness files listed above.
