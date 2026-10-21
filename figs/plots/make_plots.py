"""Charts for the paper (vector PDF, TrueType fonts). Data: the evaluation JSONs of the final 526-clip run and the
SAT human-study counts (4 raters after the pre-stated quality-control exclusion; see the supplement).
Usage: python make_plots.py <eval_vlm.json> <out_dir>"""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral"], "font.size": 8,
                     "axes.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
C = {"orig": "#9e9e9e", "steervte": "#E69F00", "B1": "#0072B2"}
E = json.load(open(sys.argv[1]))["per_clip"]; O = Path(sys.argv[2]); O.mkdir(parents=True, exist_ok=True)
ok = lambda m, u: E[m][u]["fixed"] >= 0.6

# per-material readability (all 526 clips)
mats = ["neon", "signboard", "wood", "metal", "glass", "paper", "screen"]
fig, ax = plt.subplots(figsize=(3.3, 1.55))
w = 0.27
for i, (m, lab) in enumerate([("orig", "Original"), ("steervte", "SteerVTE"), ("B1", "Banksy")]):
    vals = [100 * sum(ok(m, u) for u in E["orig"] if u.split("_")[2] == mt) / sum(1 for u in E["orig"] if u.split("_")[2] == mt) for mt in mats]
    ax.bar([k + (i - 1) * w for k in range(len(mats))], vals, w, color=C[m], label=lab, linewidth=0)
ax.set_xticks(range(len(mats))); ax.set_xticklabels(mats); ax.set_ylabel("Readability (%)"); ax.set_ylim(0, 100)
ax.legend(ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), handlelength=1.0, columnspacing=1.0)
ax.yaxis.grid(True, linewidth=0.3, alpha=0.6); ax.set_axisbelow(True)
fig.tight_layout(pad=0.2); fig.savefig(O / "material_readability.pdf"); plt.close(fig)

# SAT human study: 53 clips x 4 raters = 212 judgements per question
Q = [("Material closer\nto the original", 158, 30, 24), ("Fewer\nartifacts", 147, 21, 44), ("More stable\n(less flicker)", 60, 116, 36)]
fig, ax = plt.subplots(figsize=(3.3, 1.25))
for i, (lab, a, t, b) in enumerate(Q):
    n = a + t + b
    ax.barh(i, 100 * a / n, color=C["B1"], height=0.6, linewidth=0)
    ax.barh(i, 100 * t / n, left=100 * a / n, color="#d9d9d9", height=0.6, linewidth=0)
    ax.barh(i, 100 * b / n, left=100 * (a + t) / n, color="#bdbdbd", hatch="////", edgecolor="white", height=0.6, linewidth=0)
    ax.text(100 * a / n / 2, i, f"{a}", ha="center", va="center", color="white", fontsize=7)
    ax.text(100 * (a + t / 2) / n, i, f"{t}", ha="center", va="center", fontsize=7)
    ax.text(100 * (a + t + b / 2) / n, i, f"{b}", ha="center", va="center", fontsize=7)
ax.set_yticks(range(len(Q))); ax.set_yticklabels([q[0] for q in Q]); ax.invert_yaxis(); ax.set_xlim(0, 100)
ax.set_xlabel("Share of judgements (%)")
from matplotlib.patches import Patch
ax.legend([Patch(color=C["B1"]), Patch(color="#d9d9d9"), Patch(facecolor="#bdbdbd", hatch="////", edgecolor="white")],
          ["+ SAT", "tie", "paste only"], ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(0.45, 1.0), handlelength=1.0)
fig.tight_layout(pad=0.2); fig.savefig(O / "sat_human.pdf"); plt.close(fig)
print("ok")

# teaser side panel: headline numbers of the final 526-clip run (SteerVTE vs Banksy)
fig, axs = plt.subplots(3, 1, figsize=(2.05, 2.35))
for ax, (lab, a, b, fmt) in zip(axs, [("Readability (%) $\\uparrow$", 84.0, 91.4, "{:.1f}"), ("Net repairs (clips) $\\uparrow$", 158, 197, "{:d}"),
                                      ("Background PSNR (dB) $\\uparrow$", 36.5, 42.9, "{:.1f}")]):
    ax.barh([1, 0], [a, b], color=[C["steervte"], C["B1"]], height=0.62, linewidth=0)
    ax.set_yticks([1, 0]); ax.set_yticklabels(["SteerVTE", "Banksy"]); ax.set_xticks([]); ax.spines["bottom"].set_visible(False)
    ax.set_title(lab, fontsize=8, loc="left", pad=2)
    for y, v in ((1, a), (0, b)): ax.text(v, y, " " + fmt.format(v), va="center", fontsize=7.5)
    ax.set_xlim(0, max(a, b) * 1.28)
fig.tight_layout(pad=0.25, h_pad=0.6); fig.savefig(O / "teaser_stats.pdf"); plt.close(fig)
print("teaser ok")
