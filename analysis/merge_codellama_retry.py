"""Merge the retry pass into the CodeLlama clean re-run.

The first pass ran 64 concurrent requests against vLLM on 2x Tesla T4 with a 90 s
client timeout; requests that timed out were recorded as runtime errors (scored
wrong). The retry pass (kaggle/codellama_retry.py) re-issued exactly those items
at concurrency 32 with a 900 s timeout, same prompts, greedy decoding, same
harness. Here each retried item replaces the first-pass record (last attempt wins).
Run from the repository root:  python3 analysis/merge_codellama_retry.py
Inputs:  results/reruns/codellama_clean/raw/<tag>.jsonl,
         results/reruns/codellama_clean/retry/<tag>.retry<k>.jsonl
Outputs: results/reruns/codellama_clean/<tag>.jsonl (+ retry_log.csv)
"""
import glob, json, os, re
import pandas as pd
H = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "reruns", "codellama_clean")
log = []
for raw in sorted(glob.glob(f"{H}/raw/codellama-*-clean.*.jsonl")):
    tag = os.path.basename(raw)[:-6]
    rec = {json.loads(l)["qid"]: json.loads(l) for l in open(raw)}
    n_fail0 = sum((r.get("error") or "").startswith("runtime:") for r in rec.values())
    retries = sorted(glob.glob(f"{H}/retry/{tag}.retry*.jsonl"), key=lambda p: int(re.search(r"retry(\d+)", p).group(1)))
    for rp in retries:
        for l in open(rp):
            r = json.loads(l); r["retried"] = True; rec[r["qid"]] = r
    out = [rec[q] for q in sorted(rec)]
    with open(f"{H}/{tag}.jsonl", "w") as f:
        for r in out: f.write(json.dumps(r) + "\n")
    ex0 = 100 * sum(bool(json.loads(l)["correct"]) for l in open(raw)) / len(out)
    ex1 = 100 * sum(bool(r["correct"]) for r in out) / len(out)
    left = sum((r.get("error") or "").startswith("runtime:") for r in out)
    log.append(dict(run=tag, n=len(out), timed_out_first_pass=n_fail0, retry_files=len(retries),
                    still_runtime_error=left, ex_first_pass=round(ex0, 2), ex_merged=round(ex1, 2)))
L = pd.DataFrame(log); print(L.to_string(index=False)); L.to_csv(f"{H}/retry_log.csv", index=False)
