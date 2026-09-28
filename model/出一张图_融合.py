# -*- coding: utf-8 -*-
"""Figure 11 for the manuscript (2026-09-28): two lessons, one brain.

Reads only the two fusion logs cited in R18 and writes one PNG into
paper/figures/.  Nothing is simulated here.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "engine_v2" / "artifacts"
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 200})

CELL_ARMS = [("birth", "birth"),
             ("parent_A", "parent A"),
             ("parent_B", "parent B"),
             ("mean(A,B)", "mean(A,B)"),
             ("keep(A,B)", "keep(A,B)"),
             ("sum(A,B)", "sum(A,B)"),
             ("mean(A,A)", "mean(A,A)"),
             ("mean(birth,birth)", "mean(birth,birth)"),
             ("mean(swapped lessons)", "swapped lessons"),
             ("mean(C,D) conflict", "conflict")]

BODY_ARMS = [("birth", "birth"),
             ("parent_walk", "parent walk"),
             ("parent_brace", "parent brace"),
             ("mean(walk,brace)", "mean"),
             ("keep(walk,brace)", "keep"),
             ("mean(walk,walk)", "self-merge")]


def load(name):
    return json.loads((ART / name).read_text(encoding="utf-8"))


cells = load("fusion_parents.json")
body = load("fusion_robot_parents.json")

fig, axs = plt.subplots(1, 3, figsize=(12.4, 4.3),
                        gridspec_kw=dict(width_ratios=[1.6, 1.0, 1.0]))

# ------------------------------------------------- (a) the 17-cell graph
cell_arms = [(name, text) for name, text in CELL_ARMS
             if name in cells["cases"][0]["arms"]]
x = np.arange(len(cell_arms))
w = 0.38
a_vals = [float(np.mean([c["arms"][name]["score"][0] for c in cells["cases"]]))
          for name, _ in cell_arms]
b_vals = [float(np.mean([c["arms"][name]["score"][1] for c in cells["cases"]]))
          for name, _ in cell_arms]
axs[0].axhline(0, color="black", linewidth=0.7)
axs[0].bar(x - w / 2, a_vals, w, color="#4c72b0", edgecolor="black", linewidth=0.5,
           label="cue A, whose answer is motor 0")
axs[0].bar(x + w / 2, b_vals, w, color="#dd8452", edgecolor="black", linewidth=0.5,
           label="cue B, whose answer is motor 1")
axs[0].set_xticks(x)
axs[0].set_xticklabels([text for _, text in cell_arms], rotation=38, ha="right", fontsize=7.5)
axs[0].set_ylim(-1.15, 1.2)
axs[0].set_ylabel("answer margin")
axs[0].legend(frameon=False, fontsize=7.5, loc="lower left")
axs[0].set_title("(a) five seeds, the 17-cell graph: how far the motor cell a cue\n"
                 "is meant to drive beats the other one", fontsize=8.5)

# ------------------------------------- (b), (c) the same body, two commands
body_arms = [(name, text) for name, text in BODY_ARMS
             if name in body["cases"][0]["arms"]]
bx = np.arange(len(body_arms))
walk = [float(np.mean([c["arms"][name]["walks_on_A"] for c in body["cases"]]))
        for name, _ in body_arms]
brace = [float(np.mean([c["arms"][name]["brace_on_B"] for c in body["cases"]]))
         for name, _ in body_arms]
labels = [text for _, text in body_arms]

for ax, values, colour, ylabel, title in (
        (axs[1], walk, "#4c72b0", "metres walked in 8 s on cue A",
         "(b) the same body, \"walk\" command"),
        (axs[2], brace, "#dd8452", "flexion cell activity on cue B",
         "(c) the same body, \"brace\" command")):
    ax.bar(bx, values, 0.6, color=colour, edgecolor="black", linewidth=0.5)
    for xx, vv in zip(bx, values):
        ax.text(xx, vv + 0.02, "%.2f" % vv, ha="center", va="bottom", fontsize=7)
    ax.set_xticks(bx)
    ax.set_xticklabels(labels, rotation=38, ha="right", fontsize=7.5)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=8.5)

fig.suptitle("Two brains, each taught one different action, added and halved into one brain "
             "(1,813 cells, 17,745 edges, five seeds)", fontsize=9.5)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(OUT / "figure11_fusion.png")
plt.close(fig)
print("body:", list(zip(labels, walk, brace)))
print("wrote", OUT / "figure11_fusion.png")
