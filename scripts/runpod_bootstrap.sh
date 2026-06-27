#!/usr/bin/env bash
# Bootstrap a fresh RunPod GPU pod to run the NL2SQL benchmark, then we tear it
# down. Run this ON the pod (e.g. via `runpodctl exec` or ssh).
#
#   GPU: 1x A100 80GB comfortably serves Qwen2.5-Coder-32B in fp16 via vLLM.
#        7B/14B models fit on smaller cards (A6000 48GB).
#
# Usage on the pod:
#   bash runpod_bootstrap.sh Qwen/Qwen2.5-Coder-32B-Instruct qwen25-coder-32b
set -euo pipefail

HF_MODEL="${1:?pass the HF model id}"
MODEL_KEY="${2:?pass the config model key}"
LIMIT="${3:-}"   # optional: cap examples for a smoke run, e.g. 50

cd /workspace

# 1. Code
if [ ! -d nl2sql-onprem-bench ]; then
  git clone "${REPO_URL:?set REPO_URL to the GitHub repo}" nl2sql-onprem-bench
fi
cd nl2sql-onprem-bench
pip install -q -e . vllm

# 2. Data: BIRD dev (download once). Set BIRD_DEV_URL to the official zip.
if [ ! -f data/bird/dev.json ]; then
  mkdir -p data/bird
  echo "Download BIRD dev to data/bird/ (dev.json + dev_databases/)."
  echo "Official: https://bird-bench.github.io/  -> set BIRD_DEV_URL and unzip here."
  [ -n "${BIRD_DEV_URL:-}" ] && { curl -L "$BIRD_DEV_URL" -o /tmp/bird.zip && unzip -q /tmp/bird.zip -d data/bird; }
fi

# 3. Serve the model with vLLM (OpenAI-compatible on :8000)
pkill -f "vllm serve" 2>/dev/null || true
nohup vllm serve "$HF_MODEL" --port 8000 --max-model-len 8192 > /workspace/vllm.log 2>&1 &
echo "waiting for vLLM to come up..."
for i in $(seq 1 120); do
  curl -sf http://localhost:8000/v1/models >/dev/null 2>&1 && { echo "vLLM up"; break; }
  sleep 5
done

# 4. Run the ablation (results land in results/*.jsonl)
ARGS=(--config configs/run.yaml --model "$MODEL_KEY")
[ -n "$LIMIT" ] && ARGS+=(--limit "$LIMIT")
python scripts/run_bench.py "${ARGS[@]}"

echo "DONE. Pull results/ back, then terminate the pod."
