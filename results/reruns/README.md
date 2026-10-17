# Re-runs for release v0.3.0

All re-runs use the corrected harness (read-only databases, official set comparison,
new SQL extractor, embedder loaded once before the worker threads), on the full BIRD
dev split (n = 1534), greedy decoding. The Kaggle script that produced each folder is
listed in `kaggle/README.md`.

| folder | run tag | what it is | used in the paper as |
|---|---|---|---|
| `openrouter_v4/` | `qwen25-coder-32b-orws4` | Qwen2.5-Coder-32B-Instruct via OpenRouter (provider Cloudflare), max_tokens 4096, concurrency 16, whitespace repair applied to every completion | **the reported OpenRouter arm** (all tables) |
| `openrouter_v3/` | `qwen25-coder-32b-orws` | same configuration; the repair step was inadvertently inactive (its quote pattern treated the code fence as a quoted span). `*.rescored.json`: sequential official re-scoring with cached gold results; `gold_failed.json`: gold failures in that pass | unrepaired run; replication of v2 and the v3 -> v4 repair effect |
| `openrouter_v2/` | `qwen25-coder-32b-or` | first OpenRouter run, no repair (summary retained for the base stage only) | unrepaired run; cross-serving gap before the repair and v2 -> v3 replication |
| `codellama_clean/` | `codellama-{7b,13b}-clean` | CodeLlama-Instruct 7B and 13B, fp16, vLLM on 2x Tesla T4 (Kaggle), five stages | CodeLlama results (replace the preprint CodeLlama runs) |

`codellama_clean/` layout:

- `raw/`: first pass (concurrency 64, client timeout 90 s, 2 client retries). Requests that
  timed out on the client are recorded as `runtime:` errors. Also the vLLM server logs.
- `retry/`: retry pass that re-issued only the timed-out items (concurrency 32, client
  timeout 900 s, same prompts, greedy, same harness); `<tag>.retry<k>.jsonl` per attempt,
  `env.json` with the settings (written by hand from the session log, see its `note`),
  vLLM logs.
- `<tag>.jsonl`: merged records (each retried record replaces the first-pass record, last
  attempt wins), produced by `analysis/merge_codellama_retry.py`; `retry_log.csv` lists
  per run the timed-out items, retry files, items still failing and EX before/after.
  The `*.summary.json` next to them are the first-pass summaries.
- `vllm_*.log`: server logs of the first pass (copied from `raw/`; used to read the vLLM version).

After the retry pass one item (qid 1086 in `codellama-13b-clean.correct`) still failed: its
draft reached the 1024-token completion limit, and the repair prompt (2049 input tokens plus
1024 requested output tokens) exceeded the server's 3072-token context limit, so the server
rejected it (HTTP 400) in all three attempts. It is scored wrong.
