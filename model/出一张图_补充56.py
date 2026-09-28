# -*- coding: utf-8 -*-
"""Supplementary Figures 5 and 6 for the manuscript (2026-09-28).

Reads only the logs of the runs cited in R15 and writes two PNGs into
paper/figures_NatureMI/.  Nothing is simulated here.
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "engine_v2" / "artifacts"
OUT = ROOT / "paper" / "figures_NatureMI"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 200})

DEC = re.compile(r"\d+\.\d+")


def read(name):
    return (ART / name).read_text(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------- Figure 5
rows = []
for line in read("probe_spatial_dimension.log").splitlines():
    low = line.lower()
    vals = [float(x) for x in DEC.findall(line.split("(")[0])][:3]
    if low.startswith("input at random") and len(vals) == 3:
        rows.append(("random wiring", vals))
    elif low.startswith("flat sheet") and len(vals) == 3:
        rows.append(("flat coordinates (2D)", vals))
    elif low.startswith("solid block") and len(vals) == 3:
        rows.append(("solid coordinates (3D)", vals))
assert len(rows) == 3, rows
print("fig5 rows:", rows)

measures = ["two unrelated objects\n(lower is better)",
            "same object nudged\n(higher is better)",
            "same object, other depth\n(lower is better)"]
colours = ["#c9c9c9", "#4c72b0", "#dd8452"]

fig, ax = plt.subplots(figsize=(6.6, 3.4))
x = np.arange(3)
width = 0.26
for k, (name, vals) in enumerate(rows):
    ax.bar(x + (k - 1) * width, vals, width, label=name, color=colours[k],
           edgecolor="black", linewidth=0.5)
    for xx, vv in zip(x + (k - 1) * width, vals):
        ax.text(xx, vv + 0.02, "%.3f" % vv, ha="center", va="bottom", fontsize=7.5)
ax.set_xticks(x)
ax.set_xticklabels(measures, fontsize=8)
ax.set_ylabel("overlap of the active cells (Jaccard)")
ax.set_ylim(0, 1.08)
ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=3)
ax.set_title("1,000 cells, 6 local + 2 random edges, same input, same seed", fontsize=8.5)
fig.tight_layout()
fig.savefig(OUT / "Supplementary_Fig5_spatial_dimension.png")
plt.close(fig)

# ---------------------------------------------------------------- Figure 6
cost = []
for line in read("probe_sparse_cost.log").splitlines():
    parts = line.split()
    if len(parts) == 5 and parts[0].endswith("%") and parts[3].replace(".", "", 1).isdigit():
        cost.append((int(parts[2]), float(parts[3])))
assert len(cost) == 6, cost

sparse = []
for line in read("check_sparse_engine.log").splitlines():
    parts = line.split()
    if len(parts) == 5 and parts[0].isdigit() and parts[1].isdigit():
        lit, sp, base = int(parts[1]), float(parts[2]), float(parts[3])
        if base > 1.0:
            sparse.append((lit, sp))
assert len(sparse) == 5, sparse

scale = []
for line in read("probe_sparse_scaling.log").splitlines():
    parts = line.split()
    if len(parts) == 4 and parts[0].isdigit() and parts[3].endswith("x"):
        scale.append((int(parts[0]), float(parts[2])))
assert len(scale) == 4, scale
print("fig6 cost:", cost)
print("fig6 sparse:", sparse)
print("fig6 scale:", scale)

fig, axs = plt.subplots(1, 3, figsize=(10.0, 3.2))
axs[0].plot([c[0] for c in cost], [c[1] for c in cost], "o-", color="#c44e52")
axs[0].set_xlabel("cells alight in the step")
axs[0].set_ylabel("ms per step")
axs[0].set_ylim(0, 15)
axs[0].set_title("(a) original engine, 100,000 cells\n0 to 86,064 alight, 5.17-5.71 ms", fontsize=8.5)

axs[1].plot([s[0] for s in sparse], [s[1] for s in sparse], "o-", color="#4c72b0")
axs[1].set_xlabel("cells alight in the step")
axs[1].set_ylim(0, 15)
axs[1].set_title("(b) rebuilt engine, 100,000 cells\n0.04 ms at 0, 13.71 ms at 11,818", fontsize=8.5)

axs[2].plot([s[0] for s in scale], [s[1] for s in scale], "o-", color="#55a868")
axs[2].set_xscale("log")
axs[2].minorticks_off()
axs[2].set_xlabel("cells in the sheet (log scale)")
axs[2].set_ylabel("ms per step, drive fixed at 1,000")
axs[2].set_ylim(0, 3)
axs[2].set_xticks([s[0] for s in scale])
axs[2].set_xticklabels(["100k", "200k", "400k", "800k"])
axs[2].set_title("(c) eight times the cells,\n1.06 times the time", fontsize=8.5)
fig.tight_layout()
fig.savefig(OUT / "Supplementary_Fig6_sparse_cost.png")
plt.close(fig)
print("wrote two figures to", OUT)
