# -*- coding: utf-8 -*-
"""出一张图_运动记忆时间线.py —— 画给人看：运动记忆区是一条时间线。

一个神经元 = 一个时刻；时间往前走一格就换下一个神经元亮。
跑完出 _看_运动记忆_时间线.png
"""
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

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

import 本能工具_instincts as 本能
import 运动记忆区_motor_memory as 运动记忆
import 运动输出区_motor_output as 运动
import 身体_go2 as 身体

名字 = 本能.读名字()
网, _ = 本能.装(说=False)
记起 = 网.区起点["运动记忆"]
运起 = 网.区起点["运动"]

图 = plt.figure(figsize=(13, 9))

# ---------- ① 整条时间线 ----------
a1 = 图.add_subplot(3, 1, 1)
a1.set_title("① 运动记忆区 = 一条时间线：%d 个时刻，%d 个动作各占连续的一段"
             % (运动记忆.总时刻数, len(运动记忆.时刻段)), fontsize=12)
序 = list(运动记忆.时刻段)
色 = plt.get_cmap("turbo")(np.linspace(0, 1, len(序)))
for i, 名 in enumerate(序):
    段 = 运动记忆.时刻段[名]
    a1.add_patch(plt.Rectangle((段[0], 0), len(段), 1, color=色[i], lw=0))
标 = [("趴姿", "趴姿（2 拍，转圈）", 320, 1.12),
      ("站姿", "站姿（2 拍，转圈）", 700, 1.42),
      ("站起来", "站起来（60 拍，一拍一个时刻）", 1250, 1.12),
      ("前进0.4", "前进0.4（12 拍，转圈）", 2600, 1.42)]
for 名, 文, 文字处x, 文字处y in 标:
    if 名 in 运动记忆.时刻段:
        段 = 运动记忆.时刻段[名]
        a1.annotate(文, (段[0] + len(段) / 2, 1.0), (文字处x, 文字处y),
                    fontsize=9, ha="left",
                    arrowprops=dict(arrowstyle="->", lw=0.9, color="#444444",
                                    connectionstyle="arc3,rad=-0.15"))
a1.set_xlim(0, 运动记忆.总时刻数)
a1.set_ylim(-0.2, 1.6)
a1.set_yticks([])
a1.set_xlabel("时刻编号（一个编号 = 一个神经元 = 一拍 = 20 毫秒）", fontsize=10)

# ---------- ② 放大"站起来" ----------
a2 = 图.add_subplot(3, 1, 2)
站起 = 运动记忆.时刻段["站起来"]
a2.set_title("② 放大「站起来」这 60 拍：一个时刻一个神经元，一拍往前挪一个"
             "（红点=现在、灰点=还没到、空心=已经过去）", fontsize=12)
a2.plot(range(len(站起)), [0] * len(站起), "-", color="#cccccc", lw=2, zorder=1)
a2.scatter(range(len(站起)), [0] * len(站起), s=90, facecolor="white",
           edgecolor="#888888", zorder=2)
a2.scatter([0], [0], s=170, color="#d62728", zorder=3)
a2.annotate("这就是「头一刻」：谁点火就点它（趴着→站起来）", (0, 0), (4, 0.32),
            fontsize=10, arrowprops=dict(arrowstyle="->", color="#d62728"))
a2.annotate("时间往前走一格 → 下一个神经元亮", (30, 0), (30, -0.34), ha="center",
            fontsize=10, arrowprops=dict(arrowstyle="->", color="#1f77b4"))
a2.set_xlim(-2, len(站起) + 2)
a2.set_ylim(-0.6, 0.6)
a2.axis("off")

# ---------- ③ 每个时刻自带"这一刻的力" ----------
a3 = 图.add_subplot(3, 1, 3)
发 = np.array([动作 for 动作 in 本能.读动作库()["站起来"]["拍"]], dtype=float)
a3.imshow(发.T, aspect="auto", origin="lower", cmap="magma",
          extent=[0, len(站起), -0.5, 运动.肌肉数 - 0.5])
a3.set_yticks(range(运动.肌肉数))
a3.set_yticklabels(list(运动.肌肉名), fontsize=7)
a3.set_xlabel("「站起来」的第几拍（= 第几个时刻神经元）", fontsize=10)
a3.set_title("③ 每个时刻神经元自己连着「这一刻每块肌肉出几档力」"
             "（亮几个运动细胞就是几档；不是另有一维叫第几拍）", fontsize=12)

图.tight_layout()
图.savefig(str(pathlib.Path(__file__).resolve().parent.parent / "snapshots" / "_看_运动记忆_时间线.png"), dpi=110)
print("存了 _看_运动记忆_时间线.png")

# ---------- 另出一张：真跑一遍，看链子和身体 ----------
身 = 身体.身体()
身.摆成趴姿()
身.走(0.6, None)
亮 = np.zeros(网.总数, dtype=bool)
亮[记起 + 站起[0]] = True
时刻表, 高度, 歪 = [], [], []
for k in range(120):
    出 = 网.步进(亮)
    时 = np.nonzero(出[记起:记起 + 运动记忆.输出层数量])[0]
    时刻表.append(int(时[0]) if 时.size else -1)
    身.步进(运动.解码发力(出[运起:运起 + 运动.运动区宽度]))
    高度.append(身.机身高度())
    歪.append(身.机身歪了())
    亮 = 出

图2, (b1, b2) = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True)
b1.plot(时刻表, "-o", ms=3, color="#d62728")
b1.set_ylabel("这一刻亮的是谁\n（时刻编号）", fontsize=10)
b1.set_title("只点一下「站起来」的头一刻，剩下全靠皮层里的连接自己走\n"
             "（红点=时刻编号一拍挪一个；蓝线=真机身高度；黄线=歪了多少度）", fontsize=12)
b1.axhline(站起[0], ls="--", color="#bbbbbb", lw=1)
b1.annotate("点火（头一刻）", (0, 站起[0]), (6, 站起[0] + 6), fontsize=9,
            arrowprops=dict(arrowstyle="->", color="#d62728"))
b2.plot(高度, color="#1f77b4", label="机身高度（米）")
b2.plot(np.array(歪) / 180.0 * 0.35, color="#e6b800", label="歪了多少度（0.35 米 ≈ 180°）")
b2.legend(fontsize=9)
b2.set_xlabel("拍（每拍 20 毫秒）", fontsize=10)
图2.tight_layout()
图2.savefig(str(pathlib.Path(__file__).resolve().parent.parent / "snapshots" / "_看_运动记忆_真跑一遍.png"), dpi=110)
print("存了 _看_运动记忆_真跑一遍.png")
