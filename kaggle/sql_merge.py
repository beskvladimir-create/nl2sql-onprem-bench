"""Merge the per-item official re-scoring of sql-official-eval into one text CSV."""
import glob, json, os, csv
src = None
try:
    import kagglehub
    src = kagglehub.notebook_output_download("beskorovainyi/sql-official-eval")
except Exception as e:
    print("kagglehub failed:", e)
    cands = glob.glob("/kaggle/input/**/official_eval", recursive=True)
    src = cands[0] if cands else None
print("source:", src)
files = sorted(glob.glob(f"{src}/**/*.official.json", recursive=True))
print(len(files), "files")
os.makedirs("/kaggle/working/merged", exist_ok=True)
with open("/kaggle/working/merged/official_per_item.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["file", "qid", "released", "multiset", "set"])
    for f in files:
        name = os.path.basename(f).replace(".official.json", "")
        for r in json.load(open(f)):
            w.writerow([name, r["qid"], int(bool(r["released"])),
                        "" if r["multiset"] is None else int(r["multiset"]),
                        "" if r["set"] is None else int(r["set"])])
summ = glob.glob(f"{src}/**/official_vs_harness.csv", recursive=True)
if summ:
    import shutil; shutil.copy(summ[0], "/kaggle/working/merged/official_vs_harness.csv")
print(open("/kaggle/working/merged/official_vs_harness.csv").read() if summ else "no summary")
