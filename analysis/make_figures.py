"""Regenerate Figure1-4 of the revised manuscript.

Run from the repository root:  python3 analysis/make_figures.py
Reads results/paper/paper_ex.csv, results/paper/paper_tests_holm.csv and results/paper/paper_usage.csv,
all written by analysis/paper_numbers.py (which calls this script at the end). Writes
figures/Figure1.png .. Figure4.png as RGB PNGs (no alpha channel), at least 1300 px wide.
"""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))  # analysis/
ROOT = os.path.dirname(HERE)
PAPER = f"{ROOT}/results/paper"
OUT = f"{ROOT}/figures"
os.makedirs(OUT, exist_ok=True)
N = 1534
ORM = "qwen25-coder-32b-orws4"
EX = pd.read_csv(f"{PAPER}/paper_ex.csv", index_col="run")["ex"]
T = pd.read_csv(f"{PAPER}/paper_tests_holm.csv")
U = pd.read_csv(f"{PAPER}/paper_usage.csv").set_index("run")

BLUE, ORANGE, AQUA, YELLOW, VIOLET, GREEN = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6", "#3c8d2f"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "savefig.dpi": 220, "figure.facecolor": "white",
})


def save(fig, name):
    p = f"{OUT}/{name}"
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    Image.open(p).convert("RGB").save(p, dpi=(300, 300))  # PeerJ: no alpha channel, 300 dpi metadata
    print(name, Image.open(p).size)


# ---------------------------------------------------------------- Figure 1: EX vs model size
fig, ax = plt.subplots(figsize=(6.8, 4.4))
series = [("qwen25-coder-{}b", [7, 14, 32], BLUE, "o", "Qwen2.5-Coder (self-hosted, 16-bit)"),
          ("codellama-{}b-clean", [7, 13], GREEN, "^", "CodeLlama (self-hosted, fp16)"),
          ("llama-{}", [8, 70], ORANGE, "s", "Llama-3.1-8B / 3.3-70B (FP8 API)")]
for pat, sizes, c, mk, name in series:
    key = (lambda s: pat.format("3.1-8b" if s == 8 else "3.3-70b")) if pat.startswith("llama") else (lambda s: pat.format(s))
    b = [EX.get(f"{key(s)}.base", np.nan) for s in sizes]
    k = [EX.get(f"{key(s)}.correct", np.nan) for s in sizes]
    if np.all(np.isnan(b)):
        continue
    ax.plot(sizes, b, "-" + mk, color=c, lw=2, ms=7, label=f"{name}, single call")
    ax.plot(sizes, k, "--" + mk, color=c, lw=2, ms=7, mfc="white", mew=2, label=f"{name}, + self-correction")
    for x, y in zip(sizes, b):
        ax.annotate(f"{y:.1f}", (x, y), xytext=(0, -15), textcoords="offset points", ha="center", color=INK2, fontsize=8.5)
ax.plot([32], [EX[f"{ORM}.base"]], "D", color=VIOLET, ms=6, zorder=4, label="Qwen2.5-Coder-32B (OpenRouter, repaired), single call")
ax.annotate(f"OR {EX[f'{ORM}.base']:.1f}", (32, EX[f"{ORM}.base"]), xytext=(9, -3), textcoords="offset points",
            ha="left", color=VIOLET, fontsize=8.5)
ax.set_xscale("log")
ax.set_xticks([7, 8, 13, 14, 32, 70]); ax.set_xticklabels(["7", "8", "13", "14", "32", "70"])
for t in ax.get_xticklabels():
    if t.get_text() == "13": t.set_horizontalalignment("right")
    if t.get_text() == "14": t.set_horizontalalignment("left")
ax.minorticks_off()
ax.set_xlabel("Parameters (billions, log scale)")
ax.set_ylabel("Execution accuracy (%)")
ax.set_ylim(15, 60)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1, fontsize=9.5)
save(fig, "Figure1.png")

# ---------------------------------------------------------------- Figure 2: paired effects (forest plot)
label = {"qwen25-coder-7b": "Qwen-7B", "qwen25-coder-14b": "Qwen-14B", "qwen25-coder-32b": "Qwen-32B",
         "llama-3.1-8b": "Llama-8B (API)", "llama-3.3-70b": "Llama-70B (API)", ORM: "Qwen-32B (OR)",
         "codellama-7b-clean": "CodeLlama-7B", "codellama-13b-clean": "CodeLlama-13B"}
comp = {"base -> +self-correction": "self-correction", "lexical linking -> +self-correction": "self-correction after lex. linking",
        "embedding linking -> +self-correction": "self-correction after emb. linking",
        "base -> lexical linking": "lexical linking", "base -> embedding linking": "embedding linking",
        "lexical -> embedding linking": "embedding vs lexical",
        "+self-correction -> +self-consistency": "self-consistency", "link+correct -> +self-consistency": "self-consistency after linking"}
F = T[T.family.isin(["main", "codellama"])]
rows = []
for _, r in F.iterrows():
    c = comp[r.test]
    group = ("Self-correction" if "self-correction" in c and "consistency" not in c
             else "Self-consistency" if "consistency" in c else "Schema linking")
    rows.append(dict(name=f"{label[r.model]}: {c}", d=r.delta, lo=r.wald_lo, hi=r.wald_hi, clo=r.cl_lo, chi=r.cl_hi,
                     sig=r.p_holm < 0.05, group=group))
D = pd.DataFrame(rows)
order = ["Self-correction", "Schema linking", "Self-consistency"]
D["g"] = D.group.map({g: i for i, g in enumerate(order)})
D = D.sort_values(["g", "d"], ascending=[True, True]).reset_index(drop=True)
ypos, y, last = [], 0, None
for g in D.g:
    if last is not None and g != last:
        y += 0.8
    ypos.append(y); y += 1; last = g
D["y"] = ypos
fig, ax = plt.subplots(figsize=(7.2, 0.30 * len(D) + 1.8))
col = {"Self-correction": BLUE, "Schema linking": ORANGE, "Self-consistency": VIOLET}
for _, r in D.iterrows():
    c = col[r.group]
    ax.plot([r.lo, r.hi], [r.y, r.y], color=c, lw=3, solid_capstyle="round")
    yc = r.y + 0.32  # clustered CI drawn just below the row, with end caps, so it stays visible when nested
    ax.plot([r.clo, r.chi], [yc, yc], color=INK2, lw=1.1)
    ax.plot([r.clo, r.clo], [yc - 0.12, yc + 0.12], color=INK2, lw=1.1)
    ax.plot([r.chi, r.chi], [yc - 0.12, yc + 0.12], color=INK2, lw=1.1)
    ax.plot(r.d, r.y, "o", ms=7, color=c if r.sig else "white", mec=c, mew=2, zorder=3)
ax.axvline(0, color=INK2, lw=1)
ax.set_yticks(D.y); ax.set_yticklabels(D.name, fontsize=9.5)
ax.tick_params(axis="x", labelsize=10)
ax.invert_yaxis()
ax.set_xlabel("Paired difference in execution accuracy (percentage points)")
ax.grid(axis="y", visible=False)
h = [Line2D([], [], color=col[g], lw=3, label=g) for g in order]
h += [Line2D([], [], color=INK2, lw=1.1, marker="|", ms=8, label="database-clustered 95% CI (below each row)"),
      Line2D([], [], color=INK2, lw=0, marker="o", ms=7, label="significant after Holm"),
      Line2D([], [], color=INK2, lw=0, marker="o", ms=7, mfc="white", mew=2, label="not significant")]
ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.25, -0.7 / (0.30 * len(D) + 1.8)), ncol=2, fontsize=9)
save(fig, "Figure2.png")

# ---------------------------------------------------------------- Figure 3: linking on the OpenRouter arm
fig, ax = plt.subplots(figsize=(6.6, 3.4))
cfg = [("base", "correct", "No linking"), ("link", None, "Lexical linking"),
       ("link_emb", "link_emb_correct", "Embedding linking (k = 6)")]
vals = []
for i, (s1, s2, name) in enumerate(cfg):
    y = len(cfg) - 1 - i
    a1 = EX[f"{ORM}.{s1}"]; vals.append(a1)
    ax.plot(a1, y, "o", ms=9, color=BLUE, zorder=3)
    ax.annotate(f"{a1:.1f}", (a1, y), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8.5)
    if s2:
        a2 = EX[f"{ORM}.{s2}"]; vals.append(a2)
        ax.plot([a1, a2], [y, y], color=GRID, lw=2, zorder=1)
        ax.plot(a2, y, "D", ms=8, color=AQUA, zorder=3)
        ax.annotate(f"{a2:.1f}", (a2, y), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8.5)
    else:
        ax.annotate("self-correction not run", (a1, y), xytext=(10, -3), textcoords="offset points",
                    fontsize=8.5, color=INK2)
ax.set_yticks(range(len(cfg))); ax.set_yticklabels([c[2] for c in cfg][::-1])
ax.set_ylim(-0.6, len(cfg) - 0.4)
ax.set_xlim(np.floor(min(vals)) - 1, np.ceil(max(vals)) + 1)
ax.set_xlabel("Execution accuracy (%)")
ax.grid(axis="y", visible=False)
ax.legend(handles=[Line2D([], [], lw=0, marker="o", ms=8, color=BLUE, label="single call"),
                   Line2D([], [], lw=0, marker="D", ms=7, color=AQUA, label="+ self-correction")],
          loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=8.5)
save(fig, "Figure3.png")

# ---------------------------------------------------------------- Figure 4: accuracy vs tokens, Qwen-32B self-hosted
fig, ax = plt.subplots(figsize=(6.6, 4.0))
pts = [("qwen25-coder-32b.base", "Base", (8, -4), "left"),
       ("qwen25-coder-32b.correct", "+ corr.", (8, -4), "left"),
       ("qwen25-coder-32b.link", "Link", (8, -4), "left"),
       ("qwen25-coder-32b.link_correct", "Link + corr.", (-8, -4), "right"),
       ("qwen25-coder-32b.consistency", "+ corr. + vote", (-8, -4), "right"),
       ("qwen25-coder-32b.link_correct_consistency", "Link + corr. + vote", (-8, -4), "right")]
for k, n, (dx, dy), ha in pts:
    xx, yy = U.loc[k, "total_tok"], EX[k]
    c = VIOLET if "consistency" in k else BLUE
    ax.plot(xx, yy, "o", ms=8, color=c)
    ax.annotate(f"{n} ({yy:.1f})", (xx, yy), xytext=(dx, dy), textcoords="offset points",
                ha=ha, fontsize=8.5, color=INK)
ax.set_xscale("log")
ax.set_xticks([200, 300, 500, 1000, 2000, 3000]); ax.set_xticklabels(["200", "300", "500", "1,000", "2,000", "3,000"])
ax.minorticks_off()
ax.set_xlim(200, 3400); ax.set_ylim(53.4, 56.6)
ax.set_xlabel("Mean tokens per question (prompt + completion, all calls; log scale)")
ax.set_ylabel("Execution accuracy (%)")
save(fig, "Figure4.png")
print("figures written to", OUT)
