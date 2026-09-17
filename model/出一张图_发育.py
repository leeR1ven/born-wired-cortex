# -*- coding: utf-8 -*-
"""出一张图_发育.py —— 论文 Figure 10（R9 发育实验）从日志里读数据画出来。

三个面板：
  (a) 三个量过的阶段里"走路动作亮着几拍"（5 个种子）：教之前 0/30，教之后 199~200/200
  (b) 教学那 200 拍写进多少根新连接：点亮多巴胺 vs 不点亮
  (c) 考试最后站到多高：点亮多巴胺（会自己站起来）vs 不点亮（一直趴着）

数据来源：
  日志_发育多种子_A.log / _B.log    有环版（论文 R9 正文用这一版）
  日志_无奖励对照.log               多巴胺不点亮（对照）

命令：python 出一张图_发育.py
"""
from __future__ import annotations

import io
import pathlib
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

根 = pathlib.Path(__file__).resolve().parent
出图目录 = 根.parent / "paper" / "figures"
出图目录.mkdir(parents=True, exist_ok=True)

色 = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def 读(*名):
    t = ""
    for n in 名:
        p = 根 / n
    if not p.exists():
        p = 根.parent / "logs" / n
        if p.exists():
            t += io.open(p, encoding="utf-8").read()
    return t


def 有环():
    出 = {}
    t = 读("日志_发育多种子_A.log", "日志_发育多种子_B.log")
    for m in re.finditer(r"### 种子 (\d+)", t):
        s = int(m.group(1))
        段 = t[m.end(): m.end() + 900]
        b = re.search(r"出生：只放声音\s+挪 ([\d.]+) 米\s+最高 ([\d.]+) 米\s+走路拍了 (\d+)/(\d+)", 段)
        i = re.search(r"天生：看红\s+挪 ([\d.]+) 米\s+走路拍了 (\d+)/(\d+)", 段)
        e = re.search(r"长出 (\d+) 根新连接", 段)
        x = re.search(r"考试：只放声音\s+挪 ([\d.]+) 米\s+最高站到 ([\d.]+) 米\s+走路拍了 (\d+)/(\d+)", 段)
        if b and i and e and x:
            出[s] = dict(出生走=int(b.group(3)), 出生总=int(b.group(4)), 出生高=float(b.group(2)),
                         天生走=int(i.group(2)), 天生总=int(i.group(3)),
                         新连接=int(e.group(1)), 考走=int(x.group(3)), 考总=int(x.group(4)),
                         最高=float(x.group(2)))
    return 出


def 无奖励():
    出 = {}
    t = 读("日志_无奖励对照.log")
    for m in re.finditer(r"### 种子 (\d+)", t):
        s = int(m.group(1))
        段 = t[m.end(): m.end() + 700]
        e = re.search(r"长出 (\d+) 根新连接", 段)
        x = re.search(r"考试：只放声音\s+挪 ([\d.]+) 米\s+最高站到 ([\d.]+) 米\s+走路拍了 (\d+)/(\d+)", 段)
        if e and x:
            出[s] = dict(新连接=int(e.group(1)), 考走=int(x.group(3)), 考总=int(x.group(4)),
                         最高=float(x.group(2)))
    return 出


def main():
    环 = 有环()
    无 = 无奖励()
    if not 环:
        print("还没有发育实验的日志，先跑 实验_发育_声音叫走路_多种子.py")
        return
    种子 = sorted(环)
    print("有环版读到 %d 个种子；无奖励对照读到 %d 个" % (len(种子), len(无)))

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    图, ax = plt.subplots(1, 3, figsize=(14.2, 4.4))

    # ---- (a) 三个阶段的"走路动作亮着几拍" ----
    a = ax[0]
    阶段 = ["1 出生\n趴着·只放声音", "2 天生\n站着·看红", "4 考试\n趴着·只放声音"]
    xs = [0, 1, 2.45]
    for k, s in enumerate(种子):
        d = 环[s]
        比 = [d["出生走"] / d["出生总"], d["天生走"] / d["天生总"], d["考走"] / d["考总"]]
        xx = [x + (k - 2) * 0.045 for x in xs]
        a.plot(xx, 比, "-", color=色[k % 5], alpha=0.45, lw=1.0)
        a.plot(xx, 比, "o", color=色[k % 5], ms=6, label=str(s))
    a.annotate("", xy=(2.35, 0.75), xytext=(1.1, 0.75),
               arrowprops=dict(arrowstyle="-|>", color="0.35", lw=1.4))
    a.text(1.72, 0.79, "教学 200 拍\n红 + 声音 + 多巴胺", ha="center", fontsize=8, color="0.25")
    a.set_xticks(xs)
    a.set_xticklabels(阶段, fontsize=8)
    a.set_xlim(-0.35, 2.85)
    a.set_ylim(-0.06, 1.14)
    a.set_ylabel("走路动作亮着的比例")
    a.set_title("(a) 教之前不会，教之后会（同一个脑）", fontsize=10)
    a.legend(fontsize=7, ncol=2, title="建网种子", title_fontsize=7, loc="center left")
    a.grid(alpha=0.25)
    a.axhline(0, color="0.85", lw=0.8)

    # ---- (b) 教学写进多少根新连接 ----
    b = ax[1]
    ns = len(种子)
    mus = sorted(无)
    xs1 = [i - 0.19 for i in range(ns)]
    xs2 = [i + 0.19 for i in range(len(mus))]
    b.bar(xs1, [环[s]["新连接"] / 1e5 for s in 种子], width=0.36, color="tab:blue",
          label="点亮多巴胺")
    b.bar(xs2, [无[s]["新连接"] / 1e5 for s in mus], width=0.36, color="tab:red",
          label="不点亮多巴胺")
    for x in xs2:
        b.text(x, 0.12, "0", ha="center", fontsize=9, color="tab:red", fontweight="bold")
    b.set_xticks(range(max(ns, len(mus))))
    b.set_xticklabels([str(s) for s in sorted(set(种子) | set(无))], fontsize=7)
    b.set_ylabel("教学 200 拍写进的兴奋连接（万根）")
    b.set_ylim(0, max([环[s]["新连接"] for s in 种子]) / 1e5 * 1.22)
    b.set_title("(b) 不点亮多巴胺，就一个字节都不写", fontsize=10)
    b.legend(fontsize=8)
    b.grid(alpha=0.25, axis="y")

    # ---- (c) 考试最后站到多高 ----
    c = ax[2]
    for k, s in enumerate(种子):
        c.plot([0.82], [环[s]["最高"]], "o", color="tab:blue", ms=8)
        c.annotate(str(s), (0.82, 环[s]["最高"]), fontsize=7,
                   xytext=(6, -3), textcoords="offset points", color="0.3")
    for k, s in enumerate(mus):
        c.plot([1.18], [无[s]["最高"]], "s", color="tab:red", ms=8)
        c.annotate(str(s), (1.18, 无[s]["最高"]), fontsize=7,
                   xytext=(6, -3), textcoords="offset points", color="0.3")
    c.axhline(0.099, color="0.55", lw=1.0, ls=":")
    c.text(0.60, 0.104, "趴着的高度 0.099 米", fontsize=7.5, color="0.4")
    c.axhline(0.385, color="0.55", lw=1.0, ls=":")
    c.text(1.30, 0.389, "站着的高度\n0.385 米", fontsize=7.5, color="0.4", va="center")
    c.set_xticks([0.82, 1.18])
    c.set_xticklabels(["点亮多巴胺\n（教过）", "不点亮多巴胺\n（对照）"], fontsize=8)
    c.set_xlim(0.55, 1.78)
    c.set_ylim(0.05, 0.46)
    c.set_ylabel("考试里最高站到（米）")
    c.set_title("(c) 教过的会自己站起来，对照的一直趴着", fontsize=10)
    c.grid(alpha=0.25, axis="y")

    图.tight_layout()
    p = 出图目录 / "figure10_development.png"
    图.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(图)
    print("写好", p)


if __name__ == "__main__":
    main()