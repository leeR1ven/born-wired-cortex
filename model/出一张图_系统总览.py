# -*- coding: utf-8 -*-
"""出一张图_系统总览.py —— 画整张系统总览图（论文 Figure 1）。

命令：python 出一张图_系统总览.py
"""
from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

出图目录 = (pathlib.Path(__file__).resolve().parent.parent / "paper"
          / "figures")
出图目录.mkdir(parents=True, exist_ok=True)

蓝 = "#2b7bba"; 红 = "#b03a2e"; 紫 = "#7d3c98"; 绿 = "#1e7b4f"; 灰 = "#666666"

图, 轴 = plt.subplots(figsize=(14.0, 7.2))
轴.set_xlim(0, 140); 轴.set_ylim(0, 78); 轴.axis("off")


def 框(x, y, w, h, 文字, 边=蓝, 底="#eef4fa", 号=8.0, 粗=1.4):
    轴.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35,rounding_size=1.2",
                                linewidth=粗, edgecolor=边, facecolor=底))
    轴.text(x + w / 2, y + h / 2, 文字, ha="center", va="center", fontsize=号, color="#111111")


def 箭头(x1, y1, x2, y2, 色=灰, 粗=1.4, 曲=0.0, 虚线=False):
    轴.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
                                 linewidth=粗, color=色, linestyle="--" if 虚线 else "-",
                                 connectionstyle="arc3,rad=%f" % 曲))


# ---- 外界 ----
框(2, 46, 20, 14, "world\nMuJoCo Unitree Go2\n16 muscles\n10 physics sub-steps / tick",
   边=绿, 底="#eaf6ef", 号=8)
框(2, 22, 20, 20, "senses\ncamera 1920x1080 RGB\near 4096-bin spectrum\n"
                  "joint angles / trunk pose", 边=绿, 底="#eaf6ef", 号=8)
箭头(22, 32, 28, 32, 绿)

# ---- 感觉网（压缩） ----
框(28, 24, 20, 20,
   "sensory hierarchies\n\nvisual: 4 layers\n24x16 macro-pixels\n-> 23,040 cells\n\n"
   "auditory: spectrum\n-> 30,000 cells", 边=蓝, 底="#eef4fa", 号=8)
箭头(48, 32, 54, 32, 蓝)

# ---- 皮层 ----
轴.add_patch(FancyBboxPatch((54, 6), 54, 54, boxstyle="round,pad=0.6,rounding_size=1.6",
                            linewidth=2.0, edgecolor="#222222", facecolor="#fbfbfd"))
轴.text(81, 56.5, "one recurrent cortical sheet - 143,596 binary cells - 20 ms tick",
        ha="center", va="center", fontsize=9.2, color="#111111")

框(55, 46, 20, 8, "visual cortex\n23,040 (compressed)", 号=7.2)
框(87, 46, 20, 8, "auditory cortex\n30,000", 号=7.2)
框(55, 34, 20, 8, "visual detail\n23,040 (uncompressed)", 号=7.2)
框(87, 34, 20, 8, "prefrontal\n53,200 (compresses again)", 边=紫, 底="#f4ecf7", 号=7.2)
框(55, 22, 20, 8, "proprioception 480\nvestibular 120\nbody state 60", 号=6.8)
框(87, 22, 20, 8, "motor 160\nmotor memory 3,576", 边=红, 底="#fdecea", 号=7.2)
框(55, 10, 20, 8, "visual motion 1,920\n(frame differences)", 号=6.8)
框(87, 10, 20, 8, "inhibitory pool 8,000\n(suppression is cells,\nnever a negative weight)",
   边=紫, 底="#f4ecf7", 号=6.8)

轴.text(81, 8.9, "instincts: 304,622 pre-specified synapses  |  585 rules",
        ha="center", fontsize=7.0, color="#111111")
轴.text(81, 7.0, "plus sparse random background connectivity",
        ha="center", fontsize=7.0, color="#111111")

# 压缩信号进前额叶 / 返回线
箭头(76, 47.5, 87.5, 43.5, 紫, 粗=1.5)
轴.text(81.5, 49.4, "forward", fontsize=6.6, color=紫, ha="center")
箭头(87.0, 36.5, 76.0, 39.5, 紫, 粗=1.5)
轴.text(81.5, 34.0, "return line\n(imagery)", fontsize=6.6, color=紫, ha="center", va="top")

# 前额叶 -> 运动记忆（本能规则）
箭头(97, 34.0, 97, 30.2, 红, 粗=1.6)

箭头(107, 26, 114, 30, 红)
框(114, 26, 20, 18, "motor cortex\n160 cells\n(= 16 muscles x 10)\n\n"
                   "the number of lit cells\nis the force", 边=红, 底="#fdecea", 号=7.4)
箭头(124, 44, 124, 48, 红)
框(114, 48, 20, 14, "muscles\n16 joint-angle servos\nstiffness 80, damping 4\ntorque 25 Nm",
   边=绿, 底="#eaf6ef", 号=7.6)

# 行为是一条轨迹的说明
轴.text(112, 23.0, "an action is a trajectory\nthrough the cortex:\n"
                  "the walk is a 15-tick cycle\nliving in these synapses", fontsize=7.0,
        color=红, ha="left", va="top")

# 闭环虚线
箭头(124, 62, 12, 71, 绿, 曲=-0.10, 粗=1.2, 虚线=True)
轴.text(63, 73.2, "the body moves, the world changes", fontsize=7.6, color=绿, ha="center")

# 节拍顺序
轴.text(2, 19.5, "one tick (20 ms)", fontsize=9)
轴.text(2, 16.6, "1  time steps\n2  senses -> sensory cortices\n3  visual + auditory + motor -> prefrontal\n"
                 "4  prefrontal -> return line -> sensory / motor\n5  whole sheet: instincts + local plasticity\n"
                 "6  motor memory -> motor cortex -> muscles\n7  plasticity: what was co-active\n"
                 "     (no reward, no error signal, no gradient)",
        fontsize=7.2, va="top")

轴.set_title("Figure 1  One tick of the system: a compressor, a recurrent cortical sheet, and a body",
             fontsize=11)
p = 出图目录 / "figure1_system_overview.png"
图.savefig(p, dpi=300, bbox_inches="tight")
plt.close(图)
print("写好", p)
