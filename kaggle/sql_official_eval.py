"""Re-score every released nl2sql-onprem-bench prediction file with the official BIRD
comparison (set(pred_rows) == set(gold_rows)) next to the harness comparison
(sorted multiset of repr), on the same BIRD dev databases. No model calls."""
import glob, json, os, sqlite3, subprocess, sys
from concurrent.futures import ProcessPoolExecutor

W = "/kaggle/working"
REPO = f"{W}/nl2sql-onprem-bench"
BIRD = f"{W}/bird"
OUT = f"{W}/official_eval"
os.makedirs(OUT, exist_ok=True)

def sh(c):
    print("$", c, flush=True); subprocess.run(c, shell=True, check=True)

if not os.path.isdir(REPO):
    sh(f"git clone -q https://github.com/beskvladimir-create/nl2sql-onprem-bench.git {REPO}")
sh(f"{sys.executable} -m pip install -q func-timeout")
if not os.path.isfile(f"{BIRD}/dev.json"):
    os.makedirs(BIRD, exist_ok=True)
    sh(f"cd {W} && wget -q -O dev.zip https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip")
    sh(f"cd {W} && unzip -q -o dev.zip -d bird_raw")
    dj = glob.glob(f"{W}/bird_raw/**/dev.json", recursive=True)[0]
    d = os.path.dirname(dj)
    sh(f"cp '{dj}' {BIRD}/dev.json")
    inner = glob.glob(f"{d}/dev_databases.zip")
    if inner:
        sh(f"cd {BIRD} && unzip -q -o '{inner[0]}'")
    else:
        sh(f"cp -r '{d}/dev_databases' {BIRD}/")

from func_timeout import func_timeout, FunctionTimedOut

def _run(db, sql):
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return c.execute(sql).fetchall()
    finally:
        c.close()

def _ex(db, sql):
    try:
        return func_timeout(30, _run, args=(db, sql))
    except (FunctionTimedOut, sqlite3.Error, Exception):
        return None

GOLD = {}
def score(item):
    db, pred, gold = item
    g = _ex(db, gold)
    if g is None:
        return (None, None)
    p = _ex(db, pred) if pred else None
    if p is None:
        return (False, False)
    return (sorted(map(repr, p)) == sorted(map(repr, g)), set(p) == set(g))

files = sorted(glob.glob(f"{REPO}/results/*.jsonl")) + sorted(glob.glob(f"{W}/sql_rerun/*.jsonl"))
rows = []
for f in files:
    recs = [json.loads(l) for l in open(f)]
    items = [(f"{BIRD}/dev_databases/{r['db_id']}/{r['db_id']}.sqlite", r.get("pred_sql") or "", r["gold_sql"]) for r in recs]
    with ProcessPoolExecutor(4) as ex:
        res = list(ex.map(score, items, chunksize=8))
    harness_released = sum(bool(r["correct"]) for r in recs) / len(recs)
    multiset = sum(1 for a, b in res if a) / len(recs)
    setcmp = sum(1 for a, b in res if b) / len(recs)
    gold_fail = sum(1 for a, b in res if a is None)
    per = [{"qid": r["qid"], "released": bool(r["correct"]), "multiset": a, "set": b} for r, (a, b) in zip(recs, res)]
    json.dump(per, open(f"{OUT}/{os.path.basename(f)}.official.json", "w"))
    row = dict(file=os.path.basename(f), n=len(recs), ex_released=round(100 * harness_released, 2),
               ex_rescored_multiset=round(100 * multiset, 2), ex_official_set=round(100 * setcmp, 2),
               gold_fail=gold_fail)
    print(row, flush=True)
    rows.append(row)

import csv
with open(f"{OUT}/official_vs_harness.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
sh(f"cd {W} && zip -q -r official_eval.zip official_eval")
print("DONE")
