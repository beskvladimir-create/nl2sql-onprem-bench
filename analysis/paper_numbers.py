"""Single source of every number quoted in the manuscript.

Run from the repository root:  python3 analysis/paper_numbers.py [--no-figures] [--update-tex PATH]
(paths are resolved relative to this file, so the working directory does not matter).

Inputs (read-only, all relative to the repository root):
  results/official_rescoring/official_per_item.csv  official set-comparison re-scoring of the released runs
  results/reruns/openrouter_v4/*.jsonl, *.summary.json
                                              Qwen2.5-Coder-32B via OpenRouter, v4 = MAIN API arm
                                              (whitespace repair applied to every completion)
  results/reruns/openrouter_v3/*.jsonl, *.rescored.json
                                              v3: identical harness, repair accidentally a no-op
                                              -> independent replication of v2 through the same provider
  results/reruns/openrouter_v2/*.jsonl        v2: superseded (corrections paragraph, robustness note only)
  results/reruns/codellama_clean/<tag>.jsonl  CodeLlama 7B/13B clean re-runs, MERGED first pass + retry
                                              (regenerate with analysis/merge_codellama_retry.py)
  results/reruns/codellama_clean/raw/<tag>.jsonl  first pass only (timeouts recorded as runtime errors)
  results/reruns/codellama_clean/retry/       retry pass (+ env.json)
  results/*.jsonl                             released per-item predictions (usage, errors, SQL)
  results/paper/inputs/wall_clock.csv, invalid_items_per_file.csv
                                              per-run facts of the released runs (wall clock, old-extractor counts)
  src/nl2sql_bench/textfix.py                 whitespace repair (used to count glued tokens)

Outputs:
  results/paper/paper_numbers.txt      everything printed below
  results/paper/paper_tests_holm.csv   every paired test (Holm family + descriptive), with clustered CIs
  results/paper/paper_usage.csv        usage per run
  results/paper/paper_ex.csv           EX per run after the uniform gold-failure rule
  results/paper/autonumbers.tex        LaTeX macro block (tables + CodeLlama values + Holm family size)
  Optional: with --update-tex PATH the block between "% BEGIN AUTONUMBERS" and "% END AUTONUMBERS"
  in the manuscript at PATH is replaced by autonumbers.tex.
  Finally runs analysis/make_figures.py (figures/Figure1-4.png) unless --no-figures is given.
"""
import glob, importlib.util, json, math, os, re, subprocess, sys
import numpy as np, pandas as pd
from scipy.stats import binomtest

HERE = os.path.dirname(os.path.abspath(__file__))  # analysis/
ROOT = os.path.dirname(HERE)                       # repository root
REL = f"{ROOT}/results"                            # released runs (preprint), unchanged
RR = f"{REL}/reruns"                               # v0.3.0 re-runs
OUT = f"{REL}/paper"                               # outputs of this script
INP = f"{OUT}/inputs"
D_OR2, D_OR3, D_ORM, D_CL = (f"{RR}/openrouter_v2", f"{RR}/openrouter_v3", f"{RR}/openrouter_v4",
                             f"{RR}/codellama_clean")
os.makedirs(OUT, exist_ok=True)
N = 1534
B_REPS, SEED = 10_000, 0
INVALID = {"qwen25-coder-32b.link_emb", "qwen25-coder-32b.link_emb_correct",
           "qwen25-coder-32b-api.base", "qwen25-coder-32b-api.link"}
EXCLUDED_MODELS = ("codellama-7b.", "codellama-13b.", "codellama-34b.", "ollama-")
OR2, OR3, ORM = "qwen25-coder-32b-or", "qwen25-coder-32b-orws", "qwen25-coder-32b-orws4"
CL = ["codellama-7b-clean", "codellama-13b-clean"]
OR_STAGES = ["base", "correct", "link", "link_emb", "link_emb_correct"]
CL_STAGES = ["base", "correct", "link", "link_correct", "link_emb"]

spec = importlib.util.spec_from_file_location("textfix", f"{ROOT}/src/nl2sql_bench/textfix.py")
textfix = importlib.util.module_from_spec(spec); spec.loader.exec_module(textfix)
GLUED = textfix._KW_AFTER

out = []
def P(*a):
    out.append(" ".join(str(x) for x in a))


def load_jsonl(path):
    return sorted((json.loads(l) for l in open(path)), key=lambda r: r["qid"])


def is_err(r, prefix):
    return (r.get("error") or "").startswith(prefix)


# ================================================================ correctness
off = pd.read_csv(f"{REL}/official_rescoring/official_per_item.csv")
V, RECS, GOLDFAIL, released = {}, {}, {}, {}
for f, g in off.groupby("file"):
    k = f.replace(".jsonl", "")
    if g.shape[0] != N:
        continue
    g = g.sort_values("qid")
    GOLDFAIL[k] = set(g.loc[g["set"].isna(), "qid"])
    V[k] = g["set"].fillna(0).astype(int).to_numpy().copy()
    released[k] = g["released"].to_numpy()
for f in sorted(glob.glob(f"{REL}/*.jsonl")):
    k = os.path.basename(f)[:-6]
    if k in V:
        RECS[k] = load_jsonl(f)
for d, tag in [(D_OR2, OR2), (D_ORM, ORM)]:
    for st in OR_STAGES:
        k = f"{tag}.{st}"
        r = load_jsonl(f"{d}/{k}.jsonl")
        RECS[k] = r
        V[k] = np.array([int(bool(x["correct"])) for x in r])
        GOLDFAIL[k] = {x["qid"] for x in r if is_err(x, "gold_sql")}
for st in OR_STAGES:  # v3: official field of the sequential re-scoring
    k = f"{OR3}.{st}"
    RECS[k] = load_jsonl(f"{D_OR3}/{k}.jsonl")
    rs = sorted(json.load(open(f"{D_OR3}/{k}.rescored.json")), key=lambda x: x["qid"])
    V[k] = np.array([int(bool(x["official"])) for x in rs])
    GOLDFAIL[k] = {x["qid"] for x in rs if x["gold_failed"]}
    assert [x["qid"] for x in rs] == list(range(N))
    P(f"v3 {st}: harness == official on {sum(x['harness'] == x['official'] for x in rs)}/{N} items")
CL_RAW = {}
for m in CL:
    for st in CL_STAGES:
        k = f"{m}.{st}"
        if os.path.exists(f"{D_CL}/{k}.jsonl"):
            r = load_jsonl(f"{D_CL}/{k}.jsonl")
            RECS[k] = r
            V[k] = np.array([int(bool(x["correct"])) for x in r])
            GOLDFAIL[k] = {x["qid"] for x in r if is_err(x, "gold_sql")}
        if os.path.exists(f"{D_CL}/raw/{k}.jsonl"):
            CL_RAW[k] = load_jsonl(f"{D_CL}/raw/{k}.jsonl")
            GOLDFAIL[f"{k}#firstpass"] = {x["qid"] for x in CL_RAW[k] if is_err(x, "gold_sql")}
for k in V:
    assert len(V[k]) == N, k
DB = np.array([x["db_id"] for x in RECS[f"{ORM}.base"]])
DBS = sorted(set(DB))

def used(k):
    return k not in INVALID and not k.startswith(EXCLUDED_MODELS)

# ---- uniform gold-failure rule (M7): any question whose gold query failed or timed out in ANY
# scoring pass of any run used in the paper is scored wrong in EVERY run.
G = sorted(set().union(*[v for k, v in GOLDFAIL.items() if used(k.split("#")[0])]))
changed = {}
for k in V:
    c = int(V[k][G].sum())
    if c:
        changed[k] = c
    V[k][G] = 0
gf_counts = {k: len(v) for k, v in GOLDFAIL.items() if used(k.split("#")[0])}
P("== uniform gold-failure rule")
P(f"questions with a failing/timed-out gold query in at least one scoring pass: {G} (n={len(G)})")
P("per scoring pass:", min(gf_counts.values()), "..", max(gf_counts.values()), "items;",
  "always failing:", sorted(set.intersection(*[GOLDFAIL[k] for k in gf_counts])))
P("items in G that some run had scored correct (now set wrong):", changed or "none")

ex = lambda k: 100 * V[k].mean()
P("\n== EX official (set), uniform gold rule; released harness EX and multiset->set uplift for released runs")
EXROWS = []
for k in sorted(V):
    if not used(k):
        continue
    rel = 100 * released[k].mean() if k in released else float("nan")
    EXROWS.append(dict(run=k, ex=ex(k), n_correct=int(V[k].sum())))
    P(f"{k:45s} EX={ex(k):6.2f} released={rel:6.2f} uplift={ex(k)-rel:+5.2f} gold_failed_this_pass={len(GOLDFAIL[k])}")
pd.DataFrame(EXROWS).to_csv(f"{OUT}/paper_ex.csv", index=False)
upl = [ex(k) - 100 * released[k].mean() for k in released if used(k)]
P(f"uplift range over retained released runs: {min(upl):.2f} .. {max(upl):.2f} (n={len(upl)})")
inv = pd.read_csv(f"{INP}/invalid_items_per_file.csv").set_index("run")
ret = [k for k in released if used(k)]
P(f"old-extractor prose starts per retained released run: {inv.loc[ret, 'pred_with_prose_extraction'].min()}"
  f"..{inv.loc[ret, 'pred_with_prose_extraction'].max()}; meta-tensor failures: "
  + str({k: int(v) for k, v in inv['infra_meta_tensor'].items() if v}))

# ================================================================ statistics helpers
RNG = np.random.default_rng(SEED)
BIDX = RNG.integers(0, len(DBS), size=(B_REPS, len(DBS)))
DBI = np.array([DBS.index(d) for d in DB])
NDB = np.bincount(DBI, minlength=len(DBS)).astype(float)


def cluster_ci(vec):
    """Database-clustered percentile bootstrap (resample the 11 databases) of 100*mean(vec)."""
    s = np.bincount(DBI, weights=vec.astype(float), minlength=len(DBS))
    b = 100 * s[BIDX].sum(1) / NDB[BIDX].sum(1)
    return np.percentile(b, 2.5), np.percentile(b, 97.5)


def mcnemar(a, b):
    x, y = V[a], V[b]
    g = int(((x == 0) & (y == 1)).sum()); l = int(((x == 1) & (y == 0)).sum())
    p = binomtest(g, g + l, 0.5).pvalue if g + l else 1.0
    se = math.sqrt((g + l) - (g - l) ** 2 / N) / N * 100
    lo, hi = cluster_ci(y - x)
    return dict(a=a, b=b, ex_a=ex(a), ex_b=ex(b), delta=ex(b) - ex(a), gained=g, lost=l, p=p,
                wald_lo=ex(b) - ex(a) - 1.96 * se, wald_hi=ex(b) - ex(a) + 1.96 * se, cl_lo=lo, cl_hi=hi)


def holm(ps):
    ps = np.asarray(ps, float); order = np.argsort(ps, kind="stable"); m = len(ps); adj = np.empty(m); run = 0
    for i, j in enumerate(order):
        run = max(run, (m - i) * ps[j]); adj[j] = min(1, run)
    return adj


def fmt_p(p):
    if p >= 0.995: return "1.00"
    if p >= 0.1: return f"{p:.2f}"
    if p >= 0.01: return f"{p:.3f}"
    if p >= 0.001: return f"{p:.4f}"
    e = math.floor(math.log10(p)); mant = p / 10 ** e
    if round(mant, 1) >= 10: mant, e = mant / 10, e + 1
    return f"${mant:.1f}\\times10^{{{e}}}$"


sd = lambda d: f"${d:+.2f}$".replace("-", "-")
ci = lambda lo, hi: f"$[{lo:.2f}, {hi:.2f}]$"

# ================================================================ Holm family (all recipe tests)
pairs = []
def add(a, b, label, family, model):
    if a in V and b in V and used(a) and used(b):
        pairs.append(dict(family=family, model=model, test=label, **mcnemar(a, b)))


for m in ["qwen25-coder-7b", "qwen25-coder-14b", "qwen25-coder-32b", "llama-3.1-8b", "llama-3.3-70b", ORM] + CL:
    fam = "codellama" if m.startswith("codellama") else "main"
    add(f"{m}.base", f"{m}.correct", "base -> +self-correction", fam, m)
    add(f"{m}.link", f"{m}.link_correct", "lexical linking -> +self-correction", fam, m)
    add(f"{m}.link_emb", f"{m}.link_emb_correct", "embedding linking -> +self-correction", fam, m)
    add(f"{m}.base", f"{m}.link", "base -> lexical linking", fam, m)
    add(f"{m}.base", f"{m}.link_emb", "base -> embedding linking", fam, m)
    add(f"{m}.link", f"{m}.link_emb", "lexical -> embedding linking", fam, m)
add("qwen25-coder-32b.correct", "qwen25-coder-32b.consistency", "+self-correction -> +self-consistency", "main", "qwen25-coder-32b")
add("qwen25-coder-32b.link_correct", "qwen25-coder-32b.link_correct_consistency", "link+correct -> +self-consistency", "main", "qwen25-coder-32b")
T = pd.DataFrame(pairs)
T["p_holm"] = holm(T.p)
M_FAMILY, M_CL = len(T), int((T.family == "codellama").sum())
T["sig_holm"] = T.p_holm < 0.05
T["cl_excludes0"] = (T.cl_lo > 0) | (T.cl_hi < 0)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
P(f"\n== Holm family: {M_FAMILY} paired exact McNemar tests (main {M_FAMILY - M_CL}, CodeLlama {M_CL});"
  f" clustered bootstrap: {B_REPS} reps over {len(DBS)} databases, seed {SEED}")
P(T[["model", "test", "ex_a", "ex_b", "delta", "gained", "lost", "p", "p_holm", "cl_lo", "cl_hi"]].round(4).to_string(index=False))
dis = T[T.sig_holm != T.cl_excludes0]
P("tests where Holm significance and the clustered CI disagree:")
P(dis[["model", "test", "delta", "p_holm", "cl_lo", "cl_hi"]].round(4).to_string(index=False) if len(dis) else "none")

# ================================================================ descriptive (exploratory) paired comparisons
D = []
def addd(a, b, label, group):
    if a in V and b in V:
        D.append(dict(family="descriptive", group=group, model="", test=label, **mcnemar(a, b)))


addd("qwen25-coder-7b.base", "qwen25-coder-14b.base", "Qwen 7B -> 14B (base)", "size")
addd("qwen25-coder-14b.base", "qwen25-coder-32b.base", "Qwen 14B -> 32B (base)", "size")
addd("llama-3.1-8b.base", "llama-3.3-70b.base", "Llama 8B -> 70B (base)", "size")
addd("codellama-7b-clean.base", "codellama-13b-clean.base", "CodeLlama 7B -> 13B (base)", "size")
for st in ["base", "correct", "link"]:
    addd(f"qwen25-coder-32b.{st}", f"{ORM}.{st}", f"Qwen-32B self-hosted -> OpenRouter v4 ({st})", "cross-serving")
addd("qwen25-coder-32b.base", f"{OR2}.base", "Qwen-32B self-hosted -> OpenRouter v2, unrepaired (base)", "cross-serving")
for st in OR_STAGES:
    addd(f"{OR2}.{st}", f"{OR3}.{st}", f"OpenRouter v2 -> v3 replication ({st})", "replication")
for st in OR_STAGES:
    addd(f"{OR3}.{st}", f"{ORM}.{st}", f"OpenRouter v3 (unrepaired) -> v4 (repaired) ({st})", "repair")
addd("codellama-7b-clean.base", "llama-3.1-8b.base", "CodeLlama-7B -> Llama-3.1-8B (base)", "generation")
addd("codellama-7b-clean.base", "qwen25-coder-7b.base", "CodeLlama-7B -> Qwen2.5-Coder-7B (base)", "generation")
DD = pd.DataFrame(D)
P("\n== descriptive paired comparisons (NOT in the Holm family): exact McNemar + clustered CI")
P(DD[["group", "test", "ex_a", "ex_b", "delta", "gained", "lost", "p", "cl_lo", "cl_hi"]].round(4).to_string(index=False))
pd.concat([T, DD], ignore_index=True).to_csv(f"{OUT}/paper_tests_holm.csv", index=False)

# ================================================================ OpenRouter whitespace defect
P("\n== OpenRouter (Cloudflare) whitespace defect")
for tag in [OR2, OR3, ORM]:
    for st in OR_STAGES:
        k = f"{tag}.{st}"
        r = RECS[k]
        glued = [x for x in r if GLUED.search(x["pred_sql"] or "")]
        rep = sum(textfix.repair_ws(x["pred_sql"] or "") != (x["pred_sql"] or "") for x in r)
        ge = sum(is_err(x, "pred_sql") for x in glued); gc = sum(V[k][x["qid"]] for x in glued)
        rt = sum(is_err(x, "runtime") for x in r)
        sfile = {OR2: f"{D_OR2}/{k}", OR3: f"{D_OR3}/{k}", ORM: f"{D_ORM}/{k}"}[tag] + ".summary.json"
        s = json.load(open(sfile)) if os.path.exists(sfile) else {}
        P(f"{k:42s} EX={ex(k):6.2f} glued digit+keyword={len(glued):3d} (non-exec {ge}, correct {gc}); "
          f"repairable preds={rep:3d}; runtime errors={rt}; providers={s.get('providers')}; "
          f"repaired completions={s.get('ws_repairs')}; wall={s.get('wall_s')}; max_tokens={s.get('max_tokens')}")
for k in ["qwen25-coder-32b.base", "qwen25-coder-32b.correct", "qwen25-coder-32b.link"]:
    P(f"self-hosted {k}: glued digit+keyword={sum(bool(GLUED.search(x['pred_sql'] or '')) for x in RECS[k])}")
for st in OR_STAGES:
    a, b = RECS[f"{OR2}.{st}"], RECS[f"{OR3}.{st}"]
    same = sum(x["pred_sql"] == y["pred_sql"] for x, y in zip(a, b))
    P(f"v2 vs v3 {st}: identical final predictions {same}/{N}")
ws = lambda s: re.sub(r"\s+", "", s or "")
for tag in [OR2, ORM]:
    a, b = RECS["qwen25-coder-32b.base"], RECS[f"{tag}.base"]
    diff = [(x, y) for x, y in zip(a, b) if x["pred_sql"] != y["pred_sql"]]
    wso = [(x, y) for x, y in diff if ws(x["pred_sql"]) == ws(y["pred_sql"])]
    lost = [(x, y) for x, y in zip(a, b) if V["qwen25-coder-32b.base"][x["qid"]] == 1 and V[f"{tag}.base"][x["qid"]] == 0]
    lost_ws = [(x, y) for x, y in lost if ws(x["pred_sql"]) == ws(y["pred_sql"])]
    P(f"self-hosted 32B base vs {tag}.base: differing predictions {len(diff)}, whitespace-only {len(wso)}; "
      f"items lost {len(lost)}, of which whitespace-only {len(lost_ws)} "
      f"({sum(is_err(y, 'pred_sql') for x, y in lost_ws)} non-executing)")

# ================================================================ usage / cost
def usage(k):
    r = RECS[k]
    pt = np.array([x["usage"]["prompt_tokens"] for x in r], float)
    ct = np.array([x["usage"]["completion_tokens"] for x in r], float)
    mc = np.array([x["usage"]["model_calls"] for x in r], float)
    lat = np.array([x["usage"]["latency_s"] for x in r], float)
    fail = sum(1 for x in r if x["error"] == "pred_sql failed to execute")
    Gs = set(G)
    wrong_nonG = [x for x in r if x["qid"] not in Gs and V[k][x["qid"]] == 0]
    return dict(run=k, ex=round(ex(k), 2), prompt_tok=pt.mean(), compl_tok=ct.mean(), total_tok=(pt + ct).mean(),
                model_calls=mc.mean(), max_compl=ct.max(), p50_call_time_s=np.percentile(lat, 50),
                p95_call_time_s=np.percentile(lat, 95), pred_exec_fail=fail,
                runtime_err=sum(is_err(x, "runtime") for x in r),
                wrong_nonexec_nonG=sum(x["error"] == "pred_sql failed to execute" for x in wrong_nonG),
                wrong_executing=sum(not x["error"] for x in wrong_nonG),
                wrong_other_nonG=sum(bool(x["error"]) and x["error"] != "pred_sql failed to execute" for x in wrong_nonG))


U = pd.DataFrame([usage(k) for k in sorted(RECS) if used(k)])
wc = pd.read_csv(f"{INP}/wall_clock.csv")[["run", "wall_s", "concurrency"]]
U = U.merge(wc, on="run", how="left")
for f in glob.glob(f"{RR}/openrouter_v*/*.summary.json") + glob.glob(f"{D_CL}/*.summary.json"):
    s = json.load(open(f)); k = os.path.basename(f).replace(".summary.json", "")
    U.loc[U.run == k, ["wall_s", "concurrency"]] = [s.get("wall_s"), s.get("concurrency")]
# CodeLlama: first-pass wall clock includes timed-out work and excludes the retry pass -> not a cost
U.loc[U.run.str.startswith("codellama"), "wall_s"] = np.nan
P("\n== usage per query (means), summed model-call time percentiles, final failure categories")
P("   (wrong_executing excludes the uniform gold-failure items; CodeLlama wall clock withheld)")
P(U.round(2).to_string(index=False))
U.to_csv(f"{OUT}/paper_usage.csv", index=False)
Ui = U.set_index("run")
P("\n== token overhead (total tokens per query, ratio), calls, p95, non-executing")
for a, b in [(f"{m}.base", f"{m}.correct") for m in ["qwen25-coder-7b", "qwen25-coder-14b", "qwen25-coder-32b",
                                                       "llama-3.1-8b", "llama-3.3-70b", ORM] + CL] + [
        ("qwen25-coder-32b.correct", "qwen25-coder-32b.consistency"), ("qwen25-coder-32b.base", "qwen25-coder-32b.link"),
        (f"{ORM}.base", f"{ORM}.link"), (f"{ORM}.base", f"{ORM}.link_emb"), (f"{ORM}.link_emb", f"{ORM}.link_emb_correct")]:
    if a in Ui.index and b in Ui.index:
        ua, ub = Ui.loc[a], Ui.loc[b]
        P(f"{a} -> {b}: tokens {ua.total_tok:.1f} -> {ub.total_tok:.1f} (x{ub.total_tok/ua.total_tok:.3f}); "
          f"calls {ua.model_calls:.3f} -> {ub.model_calls:.3f}; p95 {ua.p95_call_time_s:.2f} -> {ub.p95_call_time_s:.2f} s; "
          f"exec-fail {ua.pred_exec_fail} -> {ub.pred_exec_fail}; wall {ua.wall_s} -> {ub.wall_s}")
P(f"base prompt tokens, Qwen-32B self-hosted: {Ui.loc['qwen25-coder-32b.base', 'prompt_tok']:.1f}")

# ================================================================ CodeLlama retry pass
P("\n== CodeLlama first pass (concurrency 64, client timeout 90 s, 2 client retries) and retry pass")
env = {}
for f in glob.glob(f"{D_CL}/retry/env.json"):
    env = json.load(open(f))
RETRY_CONC = env.get("concurrency", 32); RETRY_TIMEOUT = env.get("client_timeout_s", 900)
P(f"retry pass settings: concurrency {RETRY_CONC}, client timeout {RETRY_TIMEOUT} s "
  f"({'from retry/env.json' if env else 'from kaggle/codellama_retry.py (env.json not found)'}); env={env}")
CLR = []
for k in sorted(CL_RAW):
    first = sum(is_err(x, "runtime") for x in CL_RAW[k])
    merged = RECS.get(k, CL_RAW[k])
    CLR.append(dict(run=k, timed_out_first_pass=first, retried=sum(bool(x.get("retried")) for x in merged),
                    still_runtime_error=sum(is_err(x, "runtime") for x in merged),
                    ex_first_pass=100 * np.mean([bool(x["correct"]) for x in CL_RAW[k]]),
                    ex_final=ex(k) if k in V else float("nan")))
CLR = pd.DataFrame(CLR)
MERGED = bool(len(CLR)) and int(CLR.retried.sum()) > 0
P(CLR.round(2).to_string(index=False) if len(CLR) else "(no CodeLlama files)")
P("MERGE STATUS:", "merged (retry records present)" if MERGED else
  "NOT MERGED: codellama_clean/<tag>.jsonl still equal the first pass -> CodeLlama numbers are PROVISIONAL")
vl = ""
for f in glob.glob(f"{D_CL}/vllm_*.log"):
    m_ = re.search(r"LLM engine \(v([\d.]+)\)", open(f, errors="ignore").read())
    vl = m_.group(1) if m_ else vl
P("CodeLlama vLLM version (from server log):", vl)

# ================================================================ self-correction mechanism / noise
P("\n== self-correction: within-run repair effect (primary) and base -> corr. decomposition")
MECH = []
for m in ["qwen25-coder-7b", "qwen25-coder-14b", "qwen25-coder-32b", "llama-3.1-8b", "llama-3.3-70b", ORM] + CL + [
        "qwen25-coder-7b|link", "qwen25-coder-14b|link", "qwen25-coder-32b|link", f"{ORM}|link_emb",
        "codellama-7b-clean|link", "codellama-13b-clean|link"]:
    mm, src = (m.split("|") + ["base"])[:2]
    a, b = f"{mm}.{src}", f"{mm}.{'correct' if src == 'base' else src + '_correct'}"
    if a not in RECS or b not in RECS:
        continue
    rep = np.array([x["usage"]["model_calls"] > 1 for x in RECS[b]])
    x, y = V[a], V[b]
    g_r = int(((x == 0) & (y == 1) & rep).sum()); l_r = int(((x == 1) & (y == 0) & rep).sum())
    g_n = int(((x == 0) & (y == 1) & ~rep).sum()); l_n = int(((x == 1) & (y == 0) & ~rep).sum())
    within = (rep & (y == 1)).astype(int)  # draft was non-executing (hence wrong); final correct
    lo, hi = cluster_ci(within)
    asym = binomtest(g_n, g_n + l_n, 0.5).pvalue if g_n + l_n else 1.0
    nonrep = (~rep).sum()
    same_sql = sum(ra["pred_sql"] == rb["pred_sql"] for ra, rb, r_ in zip(RECS[a], RECS[b], rep) if not r_)
    afail = sum(1 for r in RECS[a] if r["error"] == "pred_sql failed to execute" or is_err(r, "runtime"))
    bfail = sum(1 for r in RECS[b] if r["error"] == "pred_sql failed to execute" or is_err(r, "runtime"))
    k_ = g_n + l_n
    MECH.append(dict(model=mm, src=src, a=a, b=b, repaired=int(rep.sum()), gain_rep=g_r, loss_rep=l_r,
                     gain_unrep=g_n, loss_unrep=l_n, unrep_asym_p=asym, flip_pct=100 * k_ / N,
                     noise_halfwidth=196 * math.sqrt(k_) / N, sql_diverge_unrep_pct=100 * (1 - same_sql / nonrep),
                     within_n=int(within.sum()), within_pts=100 * within.mean(), within_lo=lo, within_hi=hi,
                     nonexec_a=afail, nonexec_b=bfail, delta=ex(b) - ex(a)))
MECH = pd.DataFrame(MECH)
P(MECH.round(4).to_string(index=False))
P("unrepaired-flip asymmetry p:", {f"{r.model}|{r.src}": f"{r.unrep_asym_p:.2e}" for _, r in MECH.iterrows()})
P("flip_pct = items whose correctness changed on UNREPAIRED items / 1534: a lower bound (unrepaired items are"
  " the easier ones) and a mixture of serving noise and any systematic between-run difference;"
  " noise_halfwidth = 1.96*sqrt(k)/1534 in points")

# ================================================================ remaining errors after self-correction
P("\n== failure categories of the final predictions (harness execution status, official correctness)")
for k in ["qwen25-coder-32b.correct", "llama-3.3-70b.correct", f"{ORM}.correct"]:
    u = Ui.loc[k]
    wrong = N - int(V[k].sum())
    P(f"{k}: wrong {wrong} = executes-but-wrong {int(u.wrong_executing)} + non-executing {int(u.wrong_nonexec_nonG)} "
      f"+ other {int(u.wrong_other_nonG)} + gold-failure items {len(G)}; executing share of wrong "
      f"{100*u.wrong_executing/wrong:.1f}%; other errors: "
      + str(sorted({x['error'][:60] for x in RECS[k] if x['qid'] not in set(G) and V[k][x['qid']] == 0
                    and x['error'] and x['error'] != 'pred_sql failed to execute'})))

# ================================================================ linker recall on the main OpenRouter arm
P("\n== linker recall on OpenRouter v4 (table_recall); pruning detected via prompt tokens vs base")
base = {x["qid"]: x["usage"]["prompt_tokens"] for x in RECS[f"{ORM}.base"]}
for st in ["link", "link_emb"]:
    r = RECS[f"{ORM}.{st}"]
    rec = np.array([x["table_recall"] if x.get("table_recall") is not None else np.nan for x in r], float)
    pr = np.array([x["usage"]["prompt_tokens"] < base[x["qid"]] for x in r])
    P(f"{st}: mean recall={100*np.nanmean(rec):.2f} (n={np.isfinite(rec).sum()}), full-recall share={100*np.nanmean(rec==1):.2f}, "
      f"needed table dropped={100*np.nanmean(rec<1):.2f}; pruned items={pr.sum()} ({100*pr.mean():.1f}%), recall on pruned={100*np.nanmean(rec[pr]):.2f}")
ra = np.array([x["table_recall"] for x in RECS[f"{ORM}.link"]], float)
rb = np.array([x["table_recall"] for x in RECS[f"{ORM}.link_emb"]], float)
pe = np.array([x["usage"]["prompt_tokens"] < base[x["qid"]] for x in RECS[f"{ORM}.link_emb"]])
P(f"same questions where the embedding linker prunes (n={pe.sum()}): embedding {100*np.nanmean(rb[pe]):.2f} vs lexical {100*np.nanmean(ra[pe]):.2f}")
P("emb_top_k per stage:", {st: json.load(open(f"{D_ORM}/{ORM}.{st}.summary.json"))["config"]["emb_top_k"] for st in OR_STAGES})
P("max completion tokens, OpenRouter v4:", {st: max(x["usage"]["completion_tokens"] for x in RECS[f"{ORM}.{st}"]) for st in OR_STAGES},
  "; self-hosted Qwen-32B base max:", max(x["usage"]["completion_tokens"] for x in RECS["qwen25-coder-32b.base"]))

# ================================================================ contamination probe (M8)
def norm(s):
    return re.sub(r"\s+", " ", (s or "").strip().lower()).rstrip(";").strip()


P("\n== contamination probe: normalized exact match of prediction and gold SQL (lower-case, collapsed whitespace, no trailing ;)")
EM = {}
for k in sorted(RECS):
    if not used(k) or k.split(".")[0] in (OR2, OR3):
        continue
    EM[k] = 100 * np.mean([norm(x["pred_sql"]) == norm(x["gold_sql"]) for x in RECS[k]])
for k, v in EM.items():
    P(f"{k:45s} {v:5.2f}%")
P(f"range over all runs: {min(EM.values()):.2f} .. {max(EM.values()):.2f}; base runs: "
  + ", ".join(f"{k.rsplit('.', 1)[0]} {v:.2f}" for k, v in EM.items() if k.endswith(".base")))

txt = "\n".join(out)

# ================================================================ LaTeX macro block
NAME = {"qwen25-coder-7b": "Qwen-7B", "qwen25-coder-14b": "Qwen-14B", "qwen25-coder-32b": "Qwen-32B",
        "llama-3.1-8b": "Llama-8B", "llama-3.3-70b": "Llama-70B", ORM: "Qwen-32B (OR)",
        "codellama-7b-clean": "CodeLlama-7B", "codellama-13b-clean": "CodeLlama-13B"}
COMP = {"base -> +self-correction": r"base $\to$ + corr.", "lexical linking -> +self-correction": r"lexical $\to$ + corr.",
        "embedding linking -> +self-correction": r"embedding $\to$ + corr.", "base -> lexical linking": r"base $\to$ lexical",
        "base -> embedding linking": r"base $\to$ embedding", "lexical -> embedding linking": r"lexical $\to$ embedding",
        "+self-correction -> +self-consistency": r"+ corr. $\to$ + self-consistency",
        "link+correct -> +self-consistency": r"lexical + corr. $\to$ + self-consistency"}
GROUP = lambda t: 0 if "self-correction" in t and "consistency" not in t else (2 if "consistency" in t else 1)
MODEL_ORDER = list(NAME)


def test_rows(df):
    df = df.assign(g=df.test.map(GROUP), mo=df.model.map(MODEL_ORDER.index),
                   to=df.test.map(list(COMP).index)).sort_values(["g", "to", "mo"])
    rows, last = [], None
    for _, r in df.iterrows():
        if last is not None and r.g != last:
            rows.append(r"\midrule")
        last = r.g
        rows.append(f"{NAME[r.model]} & {COMP[r.test]} & {sd(r.delta)} & {r.gained}/{r.lost} & {fmt_p(r.p)} & "
                    f"{fmt_p(r.p_holm)} & {ci(r.cl_lo, r.cl_hi)} \\\\")
    return "\n".join(rows)


def cell(k):
    return f"{ex(k):.2f}" if k in V and used(k) else "--"


def ex_rows():
    rows = []
    blocks = [("Self-hosted,", ["qwen25-coder-7b", "qwen25-coder-14b", "qwen25-coder-32b"] + CL),
              ("API, FP8", ["llama-3.1-8b", "llama-3.3-70b"]), ("API, OpenRouter", [ORM])]
    lbl = {"qwen25-coder-7b": "Qwen2.5-Coder-7B", "qwen25-coder-14b": "Qwen2.5-Coder-14B", "qwen25-coder-32b": "Qwen2.5-Coder-32B",
           "codellama-7b-clean": "CodeLlama-7B", "codellama-13b-clean": "CodeLlama-13B", "llama-3.1-8b": "Llama-3.1-8B",
           "llama-3.3-70b": "Llama-3.3-70B", ORM: "Qwen2.5-Coder-32B"}
    for bi, (bname, ms) in enumerate(blocks):
        if bi:
            rows.append(r"\midrule")
        for i, m in enumerate(ms):
            first = bname if i == 0 else ("16-bit" if bi == 0 and i == 1 else ("(repaired)" if bi == 2 and i == 1 else ""))
            st = ["base", "correct", "link", "link_correct", "link_emb", "link_emb_correct"]
            rows.append(f"{first} & {lbl[m]} & " + " & ".join(cell(f"{m}.{s}") for s in st) + r" \\")
    return "\n".join(rows)


def mech_rows():
    rows = []
    order = list(NAME)
    M2 = MECH.assign(o=MECH.model.map(order.index), s=(MECH.src != "base").astype(int)).sort_values(["s", "o"])
    last = None
    for _, r in M2.iterrows():
        if last is not None and r.s != last:
            rows.append(r"\midrule")
        last = r.s
        u_a, u_b = Ui.loc[r.a], Ui.loc[r.b]
        nm = NAME[r.model] + (" (FP8 API)" if r.model.startswith("llama") else "")
        nm += {"base": "", "link": ", lexical", "link_emb": ", embedding"}[r.src]
        rows.append(f"{nm} & {r.repaired} & {r.within_n} & {sd(r.within_pts)} {ci(r.within_lo, r.within_hi)} & {r.gain_rep}/{r.loss_rep} & "
                    f"{r.gain_unrep}/{r.loss_unrep} & {r.nonexec_a} $\\to$ {r.nonexec_b} & "
                    f"{u_a.total_tok:.0f} $\\to$ {u_b.total_tok:.0f} & {u_b.model_calls:.2f} \\\\")
    return "\n".join(rows)


def cost_rows():
    def p95(k): return f"{Ui.loc[k, 'p95_call_time_s']:.1f}"
    def wall(k):
        w = Ui.loc[k, "wall_s"]
        return "--" if not np.isfinite(w) else f"{w:.0f}"
    spec_ = [("Qwen-7B (self-hosted)", "qwen25-coder-7b", [("base", "correct")]),
             ("Qwen-14B (self-hosted)", "qwen25-coder-14b", [("base", "correct")]),
             ("Qwen-32B (self-hosted)", "qwen25-coder-32b", [("base", "correct"), ("link", "link_correct"), ("correct", "consistency")]),
             ("CodeLlama-7B (self-hosted)", "codellama-7b-clean", [("base", "correct")]),
             ("CodeLlama-13B (self-hosted)", "codellama-13b-clean", [("base", "correct")]),
             None,
             ("Llama-8B (FP8 API)", "llama-3.1-8b", [("base", "correct")]),
             ("Llama-70B (FP8 API)", "llama-3.3-70b", [("base", "correct")]),
             ("Qwen-32B (OpenRouter)", ORM, [("base", "correct"), ("link_emb", "link_emb_correct")])]
    cfgname = {("base", "correct"): "base / + corr.", ("link", "link_correct"): "lexical / + corr.",
               ("correct", "consistency"): "+ corr. + self-consistency", ("link_emb", "link_emb_correct"): "embedding / + corr."}
    rows = []
    for s in spec_:
        if s is None:
            rows.append(r"\midrule"); continue
        nm, m, cfgs = s
        for j, (x, y) in enumerate(cfgs):
            a, b = f"{m}.{x}", f"{m}.{y}"
            if a not in Ui.index or b not in Ui.index:
                continue
            cl_ = m.startswith("codellama")
            if (x, y) == ("correct", "consistency"):
                rows.append(f" & {cfgname[(x, y)]} & {Ui.loc[b, 'total_tok']:.0f} & {Ui.loc[b, 'model_calls']:.2f} & "
                            f"{p95(b)} & {wall(b)} \\\\")
                continue
            w = "--" if (not np.isfinite(Ui.loc[a, "wall_s"]) and not np.isfinite(Ui.loc[b, "wall_s"])) else f"{wall(a)} / {wall(b)}"
            rows.append(f"{nm if j == 0 else ''} & {cfgname[(x, y)]} & {Ui.loc[a, 'total_tok']:.0f} / {Ui.loc[b, 'total_tok']:.0f} & "
                        f"{Ui.loc[a, 'model_calls']:.2f} / {Ui.loc[b, 'model_calls']:.2f} & "
                        + ("-- & --" if cl_ else f"{p95(a)} / {p95(b)} & {w}") + r" \\")
    return "\n".join(rows)


def desc_rows():
    lab = {"Qwen 7B -> 14B (base)": r"Qwen2.5-Coder 7B $\to$ 14B, base",
           "Qwen 14B -> 32B (base)": r"Qwen2.5-Coder 14B $\to$ 32B, base$^{a}$",
           "Llama 8B -> 70B (base)": r"Llama-3.1-8B $\to$ Llama-3.3-70B, base",
           "CodeLlama 7B -> 13B (base)": r"CodeLlama 7B $\to$ 13B, base",
           "CodeLlama-7B -> Llama-3.1-8B (base)": r"CodeLlama-7B $\to$ Llama-3.1-8B, base$^{b}$",
           "CodeLlama-7B -> Qwen2.5-Coder-7B (base)": r"CodeLlama-7B $\to$ Qwen2.5-Coder-7B, base$^{a}$"}
    for st, nm in [("base", "base"), ("correct", "+ corr."), ("link", "lexical")]:
        lab[f"Qwen-32B self-hosted -> OpenRouter v4 ({st})"] = f"Qwen-32B self-hosted $\\to$ OR v4, {nm}"
    lab["Qwen-32B self-hosted -> OpenRouter v2, unrepaired (base)"] = r"Qwen-32B self-hosted $\to$ OR v2 (unrepaired), base"
    stn = {"base": "base", "correct": "+ corr.", "link": "lexical", "link_emb": "embedding", "link_emb_correct": "emb. + corr."}
    for st in OR_STAGES:
        lab[f"OpenRouter v2 -> v3 replication ({st})"] = f"OR v2 $\\to$ v3 (replication), {stn[st]}"
        lab[f"OpenRouter v3 (unrepaired) -> v4 (repaired) ({st})"] = f"OR v3 $\\to$ v4 (repair), {stn[st]}"
    rows, last = [], None
    order = ["size", "generation", "cross-serving", "replication", "repair"]
    for _, r in DD.assign(o=DD.group.map(order.index)).sort_values("o", kind="stable").iterrows():
        if last is not None and r.group != last:
            rows.append(r"\midrule")
        last = r.group
        rows.append(f"{lab[r.test]} & {r.ex_a:.2f} & {r.ex_b:.2f} & {sd(r.delta)} & {r.gained}/{r.lost} & {fmt_p(r.p)} & "
                    f"{ci(r.cl_lo, r.cl_hi)} \\\\")
    return "\n".join(rows)


mac = ["% Generated by analysis/paper_numbers.py -- do not edit by hand.",
       f"% CodeLlama status: {'MERGED (final)' if MERGED else 'PROVISIONAL first pass (retry not merged)'}",
       f"\\newcommand{{\\holmM}}{{{M_FAMILY}}}",
       f"\\newcommand{{\\holmMain}}{{{M_FAMILY - M_CL}}}",
       f"\\newcommand{{\\holmCL}}{{{M_CL}}}",
       f"\\newcommand{{\\goldFailN}}{{{len(G)}}}",
       f"\\newcommand{{\\goldFailIDs}}{{{', '.join(map(str, G))}}}",
       f"\\newcommand{{\\goldMaxEX}}{{{100 * (N - len(G)) / N:.2f}}}",
       "\\newcommand{\\TabEXBody}{%\n" + ex_rows() + "}",
       "\\newcommand{\\TabSigBody}{%\n" + test_rows(T[T.family == "main"]) + "}",
       "\\newcommand{\\TabCLBody}{%\n" + test_rows(T[T.family == "codellama"]) + "}",
       "\\newcommand{\\TabMechBody}{%\n" + mech_rows() + "}",
       "\\newcommand{\\TabCostBody}{%\n" + cost_rows() + "}",
       "\\newcommand{\\TabDescBody}{%\n" + desc_rows() + "}"]
MW = {"qwen25-coder-7b": "QwenSeven", "qwen25-coder-14b": "QwenFourteen", "qwen25-coder-32b": "QwenThirtyTwo",
      "llama-3.1-8b": "LlamaEight", "llama-3.3-70b": "LlamaSeventy", ORM: "OR"}
TW = {"base -> +self-correction": "Corr", "lexical linking -> +self-correction": "LinkCorr",
      "embedding linking -> +self-correction": "EmbCorr", "base -> lexical linking": "Link",
      "base -> embedding linking": "Emb", "lexical -> embedding linking": "EmbLex",
      "+self-correction -> +self-consistency": "SC", "link+correct -> +self-consistency": "SCLink"}
for _, r in T[T.family == "main"].iterrows():  # Holm-adjusted p of the main tests (depend on the full family)
    mac.append(f"\\newcommand{{\\pH{MW[r.model]}{TW[r.test]}}}{{{fmt_p(r.p_holm)}}}")
NC = MECH[~MECH.model.isin(CL)]
QW = NC[NC.model.isin(["qwen25-coder-7b", "qwen25-coder-14b", "qwen25-coder-32b"])]
ORF = NC[NC.model == ORM]
NB = NC[NC.src == "base"]
mac += [f"\\newcommand{{\\flipQwenRange}}{{{QW.flip_pct.min():.2f}--{QW.flip_pct.max():.2f}}}",
        f"\\newcommand{{\\flipQwenShort}}{{{QW.flip_pct.min():.1f}--{QW.flip_pct.max():.1f}}}",
        f"\\newcommand{{\\flipORRange}}{{{ORF.flip_pct.min():.2f}--{ORF.flip_pct.max():.2f}}}",
        f"\\newcommand{{\\halfAllRange}}{{{NC.noise_halfwidth.min():.2f} to {NC.noise_halfwidth.max():.2f}}}",
        f"\\newcommand{{\\halfBaseRange}}{{{NB.noise_halfwidth.min():.2f} to {NB.noise_halfwidth.max():.2f}}}",
        f"\\newcommand{{\\withinNRange}}{{{NC.within_n.min()} to {NC.within_n.max()}}}",
        f"\\newcommand{{\\withinPtsRange}}{{${NC.within_pts.min():+.2f}$ to ${NC.within_pts.max():+.2f}$}}",
        f"\\newcommand{{\\withinNConf}}{{{len(NC)}}}",
        f"\\newcommand{{\\withinAllCIpos}}{{{'every' if (MECH.within_lo > 0).all() else 'not every'}}}"]
WORD = {"codellama-7b-clean": "Seven", "codellama-13b-clean": "Thirteen"}
for m, w in WORD.items():
    for st, nm in [("base", "Base"), ("correct", "Corr"), ("link", "Link"), ("link_correct", "LinkCorr"), ("link_emb", "Emb")]:
        k = f"{m}.{st}"
        mac.append(f"\\newcommand{{\\cl{w}{nm}}}{{{ex(k):.2f}}}" if k in V else f"\\newcommand{{\\cl{w}{nm}}}{{\\textbf{{??}}}}")
    for lab_, nm in [("base -> +self-correction", "Corr"), ("base -> lexical linking", "Link"),
                     ("lexical linking -> +self-correction", "LinkCorr"), ("base -> embedding linking", "Emb"),
                     ("lexical -> embedding linking", "EmbLex")]:
        r = T[(T.model == m) & (T.test == lab_)]
        if len(r):
            r = r.iloc[0]
            mac += [f"\\newcommand{{\\cl{w}{nm}D}}{{{sd(r.delta)}}}", f"\\newcommand{{\\cl{w}{nm}P}}{{{fmt_p(r.p)}}}",
                    f"\\newcommand{{\\cl{w}{nm}H}}{{{fmt_p(r.p_holm)}}}", f"\\newcommand{{\\cl{w}{nm}GL}}{{{r.gained}/{r.lost}}}",
                    f"\\newcommand{{\\cl{w}{nm}CI}}{{{ci(r.cl_lo, r.cl_hi)}}}"]
    me = MECH[(MECH.model == m) & (MECH.src == "base")]
    if len(me):
        me = me.iloc[0]
        mac += [f"\\newcommand{{\\cl{w}Rep}}{{{me.repaired}}}", f"\\newcommand{{\\cl{w}Within}}{{{sd(me.within_pts)}}}",
                f"\\newcommand{{\\cl{w}WithinCI}}{{{ci(me.within_lo, me.within_hi)}}}",
                f"\\newcommand{{\\cl{w}NonExec}}{{{me.nonexec_a} $\\to$ {me.nonexec_b}}}",
                f"\\newcommand{{\\cl{w}Flip}}{{{me.flip_pct:.2f}}}"]
    for st, nm in [("base", "Base"), ("correct", "Corr")]:
        k = f"{m}.{st}"
        if k in Ui.index:
            mac += [f"\\newcommand{{\\cl{w}Tok{nm}}}{{{Ui.loc[k, 'total_tok']:.0f}}}",
                    f"\\newcommand{{\\cl{w}Calls{nm}}}{{{Ui.loc[k, 'model_calls']:.2f}}}"]
    if f"{m}.base" in Ui.index and f"{m}.correct" in Ui.index:
        mac.append(f"\\newcommand{{\\cl{w}TokRatio}}{{{Ui.loc[f'{m}.correct', 'total_tok'] / Ui.loc[f'{m}.base', 'total_tok']:.2f}}}")
    if f"{m}.base" in EM:
        mac.append(f"\\newcommand{{\\cl{w}EM}}{{{EM[f'{m}.base']:.2f}}}")
for lab_, nm in [("CodeLlama-7B -> Llama-3.1-8B (base)", "Llama"), ("CodeLlama-7B -> Qwen2.5-Coder-7B (base)", "Qwen")]:
    r = DD[DD.test == lab_]
    if len(r):
        r = r.iloc[0]
        mac += [f"\\newcommand{{\\clGap{nm}}}{{{r.delta:.2f}}}", f"\\newcommand{{\\clGap{nm}P}}{{{fmt_p(r.p)}}}",
                f"\\newcommand{{\\clGap{nm}CI}}{{{ci(r.cl_lo, r.cl_hi)}}}"]
r = DD[DD.test == "CodeLlama 7B -> 13B (base)"]
if len(r):
    r = r.iloc[0]
    mac += [f"\\newcommand{{\\clSizeD}}{{{sd(r.delta)}}}", f"\\newcommand{{\\clSizeP}}{{{fmt_p(r.p)}}}",
            f"\\newcommand{{\\clSizeCI}}{{{ci(r.cl_lo, r.cl_hi)}}}"]
if len(CLR):
    mac += [f"\\newcommand{{\\clTimedOutTotal}}{{{int(CLR.timed_out_first_pass.sum())}}}",
            f"\\newcommand{{\\clTimedOutMin}}{{{int(CLR.timed_out_first_pass.min())}}}",
            f"\\newcommand{{\\clTimedOutMax}}{{{int(CLR.timed_out_first_pass.max())}}}",
            f"\\newcommand{{\\clTimedOutMaxRun}}{{\\texttt{{{CLR.loc[CLR.timed_out_first_pass.idxmax(), 'run'].replace('_', '\\_')}}}}}",
            f"\\newcommand{{\\clStillFailed}}{{{int(CLR.still_runtime_error.sum())}}}"]
mac += [f"\\newcommand{{\\clRetryConc}}{{{RETRY_CONC}}}", f"\\newcommand{{\\clRetryTimeout}}{{{RETRY_TIMEOUT}}}",
        f"\\newcommand{{\\clVllm}}{{{vl or '??'}}}"]
cl_flip = MECH[MECH.model.isin(CL)]
if len(cl_flip):
    mac.append(f"\\newcommand{{\\clFlipRange}}{{{cl_flip.flip_pct.min():.2f}--{cl_flip.flip_pct.max():.2f}}}")
    mac.append(f"\\newcommand{{\\clFlipRangeShort}}{{{cl_flip.flip_pct.min():.2f}--{cl_flip.flip_pct.max():.2f}}}")
    mac.append(f"\\newcommand{{\\clHalfRange}}{{{cl_flip.noise_halfwidth.min():.2f} to {cl_flip.noise_halfwidth.max():.2f}}}")
    mac.append(f"\\newcommand{{\\clWithinRange}}{{{cl_flip.within_n.min()} to {cl_flip.within_n.max()} items, "
               f"${cl_flip.within_pts.min():+.2f}$ to ${cl_flip.within_pts.max():+.2f}$ points}}")
PLACEHOLDER = not MERGED and "--fill-provisional" not in sys.argv
QQ = r"\textbf{??}"


def blank_cl_rows(body, n_label):
    rows = []
    for line in body.split("\n"):
        if "CodeLlama" in line and "&" in line:
            cells = line.rstrip().removesuffix("\\\\").rstrip().split(" & ")
            line = " & ".join(cells[:n_label] + [QQ] * (len(cells) - n_label)) + " \\\\"
        rows.append(line)
    return "\n".join(rows)


if PLACEHOLDER:
    keep = ("clRetryConc", "clRetryTimeout", "clVllm", "clTimedOutTotal", "clTimedOutMin", "clTimedOutMax", "clTimedOutMaxRun")
    NL = {"TabEXBody": 2, "TabCLBody": 2, "TabMechBody": 1, "TabCostBody": 2, "TabDescBody": 1}
    for i, mline in enumerate(mac):
        m_ = re.match(r"\\newcommand\{\\(\w+)\}", mline)
        if not m_:
            continue
        name = m_.group(1)
        if name in NL:
            head, body = mline.split("{%\n", 1)
            mac[i] = head + "{%\n" + blank_cl_rows(body[:-1], NL[name]) + "}"
        elif name.startswith("cl") and name not in keep:
            mac[i] = f"\\newcommand{{\\{name}}}{{{QQ}}}"
    mac.insert(2, "% CodeLlama cells are placeholders until the retry pass is merged; p_Holm of the other"
                  " tests uses the provisional CodeLlama p-values (family size is final).")
block = "\n".join(mac)
open(f"{OUT}/autonumbers.tex", "w").write(block + "\n")

txt += "\n\n== LaTeX macro block (also written to results/paper/autonumbers.tex)\n" + block
print(txt)
open(f"{OUT}/paper_numbers.txt", "w").write(txt + "\n")

if "--update-tex" in sys.argv:  # optional: --update-tex PATH/TO/manuscript.tex
    i = sys.argv.index("--update-tex")
    if i + 1 >= len(sys.argv) or sys.argv[i + 1].startswith("--"):
        sys.exit("--update-tex needs the path of the manuscript .tex file")
    p = sys.argv[i + 1]
    s = open(p).read()
    a, b = s.index("% BEGIN AUTONUMBERS\n"), s.index("% END AUTONUMBERS")
    open(p, "w").write(s[:a] + "% BEGIN AUTONUMBERS\n" + block + "\n" + s[b:])
    print(f"{p}: autonumbers block replaced")
if "--no-figures" not in sys.argv:
    subprocess.run([sys.executable, f"{HERE}/make_figures.py"], check=True)
