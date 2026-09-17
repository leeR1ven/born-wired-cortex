# -*- coding: utf-8 -*-
"""出一张图_走路.py —— 看见红色以后一直往前走：路线图 + 动图。"""
from __future__ import annotations
import pathlib

import os
import sys

os.chdir(str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

import 本能工具_instincts as 本能
import 运动记忆区_motor_memory as 运动记忆
import 运动输出区_motor_output as 运动
import 身体_go2 as 身体
import 感觉区_共同 as 感觉
import 视觉前处理_visual_preprocess as 视觉

秒 = 10.0
网, _ = 本能.装(说=False)
记起 = 网.区起点["运动记忆"]; 运起 = 网.区起点["运动"]
视起 = 网.区起点["视觉"]; 视宽 = 网.区宽["视觉"]
谁 = {}
for 名, 段 in 运动记忆.时刻段.items():
    for t in 段:
        谁[t] = 名

图 = np.zeros((1080, 1920, 3), dtype=np.uint8)
图[:, 640:1280, 0] = 255
信 = 视觉.图像转信号(图)

宽, 高 = 480, 340
身 = 身体.身体(渲染宽=宽, 渲染高=高)
身.摆成站姿(); 身.走(0.5, None)
亮 = np.zeros(网.总数, dtype=bool)
路, 高表, 歪表, 动作表, 帧 = [], [], [], [], []
倒 = None
帧数 = int(秒 / 0.02)
for k in range(帧数):
    给 = 亮.copy()
    本能.灌视觉(给, 网, 图)
    感觉.灌感觉(给, 网, 身)
    出 = 网.步进(给)
    时 = [int(x) for x in np.nonzero(出[记起:记起 + 运动记忆.输出层数量])[0]]
    身.步进(运动.解码发力(出[运起:运起 + 运动.运动区宽度]))
    亮 = 出
    p = 身.数据.qpos
    路.append((float(p[0]), float(p[1])))
    高表.append(身.机身高度()); 歪表.append(身.机身歪了())
    动作表.append(时[0] if 时 else -1)
    if 倒 is None and 身.机身歪了() > 75:
        倒 = k * 0.02
    if k % 10 == 0:
        帧.append(身.画(相机=100.0, 俯角=-20.0, 距离=1.35,
                        看向=(float(p[0]), float(p[1]), 0.30)))

路 = np.array(路)
图1 = plt.figure(figsize=(13, 4.6))
a1 = 图1.add_subplot(1, 3, 1)
a1.plot(路[:, 0], 路[:, 1], "-", color="#1f77b4", lw=1.6)
a1.plot(路[0, 0], 路[0, 1], "o", color="#2ca02c", ms=9, label="起点（站着）")
a1.plot(路[-1, 0], 路[-1, 1], "o", color="#d62728", ms=9, label="10 秒后")
for i in range(0, len(路), 50):
    a1.annotate("%.0f秒" % (i * 0.02), (路[i, 0], 路[i, 1]), fontsize=8, color="#555555")
a1.set_title("① 看见红以后 10 秒的路线（从上往下看）", fontsize=12)
a1.set_xlabel("前后（米）"); a1.set_ylabel("左右（米）")
a1.axis("equal"); a1.legend(fontsize=8)

a2 = 图1.add_subplot(1, 3, 2)
a2.plot(np.arange(len(高表)) * 0.02, 高表, color="#1f77b4", label="机身高度（米）")
a2.plot(np.arange(len(歪表)) * 0.02, np.array(歪表) / 180 * 0.4, color="#e6b800",
        label="歪了多少度（0.4 米 ≈ 180°）")
a2.axhline(0.18, ls="--", color="#d62728", lw=1)
a2.annotate("掉到这条线以下就算趴了", (6, 0.185), fontsize=8, color="#d62728")
a2.set_title("② 一直站着没倒（%s）" % ("一次没翻 ✓" if 倒 is None else "%.2f 秒翻了" % 倒),
             fontsize=12)
a2.set_xlabel("秒"); a2.legend(fontsize=8)

a3 = 图1.add_subplot(1, 3, 3)
a3.plot(np.arange(len(动作表)) * 0.02, 动作表, ".", ms=1, color="#d62728")
a3.set_title("③ 这 10 秒里亮的是哪个时刻神经元\n（%s 的那一串在转圈）"
             % (谁.get(动作表[len(动作表) // 2], "?")), fontsize=12)
a3.set_xlabel("秒"); a3.set_ylabel("时刻编号（在运动记忆区里的位置）")
图1.tight_layout()
图1.savefig(str(pathlib.Path(__file__).resolve().parent.parent / "snapshots" / "_看_走路_路线.png"), dpi=110)
print("存了 _看_走路_路线.png")

if 帧:
    小 = [Image.fromarray(f).resize((400, 283)) for f in 帧]
    小[0].save(str(pathlib.Path(__file__).resolve().parent.parent / "snapshots" / "_看_走路.gif"), save_all=True,
               append_images=小[1:], duration=140, loop=0)
    print("存了 _看_走路.gif（%d 帧）" % len(小))

p = 身.数据.qpos
路程 = float(np.sum(np.hypot(np.diff(路[:, 0]), np.diff(路[:, 1]))))
print("10 秒：走了 %.2f 米路程，直线挪了 %.2f 米  %s"
      % (路程, float(np.hypot(p[0] - 路[0, 0], p[1] - 路[0, 1])),
         ("第 %.2f 秒翻倒" % 倒) if 倒 else "一次没翻"))
