import pathlib

# -*- coding: utf-8 -*-
"""把身体从侧面画出来（我们自己看，模型看不到）。"""
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import 世界_world as W

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]

def 画(ax, w, 标题):
    p = w.位置
    for a, b in W.骨架:
        i, j = W.序号[a], W.序号[b]
        ax.plot([p[i,0], p[j,0]], [p[i,2], p[j,2]], "-", color="tab:blue", lw=2.2, zorder=3)
    for (_, a, b) in [(x[0], x[1], x[2]) for x in W.被动表]:
        i, j = W.序号[a], W.序号[b]
        ax.plot([p[i,0], p[j,0]], [p[i,2], p[j,2]], "--", color="tab:gray", lw=0.9, zorder=2)
    ax.scatter(p[:,0], p[:,2], s=(W.质量*2.2+8), color="tab:orange", zorder=4)
    ax.scatter(p[W.序号["头"],0], p[W.序号["头"],2], s=60, color="tab:red", zorder=5)
    # 地面和脚底
    ax.axhspan(-0.05, W.最小离地, color="0.7", zorder=1)
    ax.axhline(W.最小离地, color="0.4", lw=1)
    ax.set_aspect("equal")
    ax.set_xlim(-1.4, 1.4); ax.set_ylim(-0.1, 1.9)
    ax.set_title(标题, fontsize=11)
    ax.grid(alpha=0.25)

w = W.世界()
fig, ax = plt.subplots(1, 4, figsize=(16, 4.4))
画(ax[0], w, "站姿（默认，不用使劲就站住）")
for _ in range(240): w.步进()
画(ax[1], w, "站 1 秒后")

w.摆成躺着(脸朝上=True)
画(ax[2], w, "仰面躺下")
for _ in range(240): w.步进()
画(ax[3], w, "躺 1 秒后")
print("躺下后：", w.摘要())
print("头部看出去：\n", w.看()[:, :, 0])
plt.tight_layout()
plt.savefig(str(pathlib.Path(__file__).resolve().parent.parent / "snapshots" / "世界_身体侧视.png"), dpi=95, bbox_inches="tight")
print("存图了")
