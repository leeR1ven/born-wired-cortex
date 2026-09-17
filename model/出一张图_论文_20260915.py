# -*- coding: utf-8 -*-
"""出一张图_论文_20260915.py —— 把新版论文用的 7 张图从日志里读出来画成 PNG。

每一张图的数据来源见函数名后面的注释。数据还没跑完的图会自动跳过（不报错）。

命令：python 出一张图_论文_20260915.py [1,2,3,4,5,6,7]
"""
from __future__ import annotations

import io
import pathlib
import re
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

根 = pathlib.Path(__file__).resolve().parent
出图目录 = 根.parent / "paper" / "figures"
出图目录.mkdir(parents=True, exist_ok=True)


def 读文本(名):
    p = 根 / 名
    if not p.exists():
        p = 根.parent / "logs" / 名
    if not p.exists():
        return ""
    return io.open(p, encoding="utf-8").read()


def 存(图, 名):
    p = 出图目录 / 名
    图.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(图)
    print("  写好", p.name)


# ---------------- 图 1：R1 ----------------
def 图1():
    t = 读文本("日志_闭环多种子.log")
    if "【汇总】" not in t:
        return
    行 = re.findall(r"种子 (\d+)：起手 ([\d.]+) 米 → 站起来末高 ([\d.]+)（最高 ([\d.]+)）；"
                    r"朝红走了 ([-+][\d.]+) 米；什么都不给 → 末高 ([\d.]+)、歪 (\d+) 度", t)
    if not 行:
        return
    行 = [(int(a), float(b), float(c), float(d), float(e), float(f), float(g)) for a, b, c, d, e, f, g in 行]
    种子 = [r[0] for r in 行]
    起手 = [r[1] for r in 行]
    末高 = [r[2] for r in 行]
    走了 = [r[4] for r in 行]
    瘫 = [r[5] for r in 行]
    歪 = [r[6] for r in 行]
    x = np.arange(len(行))
    图, 轴 = plt.subplots(1, 4, figsize=(15.6, 3.5))
    轴[0].bar(x - 0.2, 起手, 0.4, label="start", color="#bbbbbb")
    轴[0].bar(x + 0.2, 末高, 0.4, label="end", color="#2b7bba")
    轴[0].axhline(0.20, ls="--", c="k", lw=0.8)
    轴[0].text(0.02, 0.21, "success threshold 0.20 m", fontsize=7, transform=轴[0].get_yaxis_transform())
    轴[0].set_title("(a) prone -> stand up", fontsize=9); 轴[0].set_ylabel("trunk height (m)")
    轴[0].set_xticks(x); 轴[0].set_xticklabels(种子, fontsize=7, rotation=45); 轴[0].legend(fontsize=7)
    轴[1].bar(x, 走了, 0.55, color="#2b7bba")
    轴[1].axhline(0.30, ls="--", c="k", lw=0.8)
    轴[1].set_title("(b) red ahead -> walk", fontsize=9); 轴[1].set_ylabel("forward displacement (m)")
    轴[1].set_xticks(x); 轴[1].set_xticklabels(种子, fontsize=7, rotation=45)
    轴[2].bar(x, 瘫, 0.55, color="#b03a2e")
    轴[2].axhline(0.20, ls="--", c="k", lw=0.8)
    轴[2].set_title("(c) no sensation -> collapse", fontsize=9); 轴[2].set_ylabel("trunk height (m)")
    轴[2].set_xticks(x); 轴[2].set_xticklabels(["%d\n(%.0f deg tilt)" % (s, w) for s, w in zip(种子, 歪)],
                                               fontsize=6, rotation=45)
    # (d) 白脑对照：同一批种子、同一套背景连接、同一个身体，只把本能表清空
    b = 读文本("日志_白脑_新.log")
    b行 = re.findall(r"(白脑|完整脑)\s+种子 (\d+)（本能连接 (\d+) 根）：起手 ([\d.]+) 米 → "
                     r"站起来末高 ([\d.]+)；朝红走了 ([-+][\d.]+) 米；什么都不给 → 末高 ([\d.]+)、歪 (\d+) 度", b)
    if b行:
        全高 = [float(r[4]) for r in b行 if r[0] == "完整脑"]
        全走 = [float(r[5]) for r in b行 if r[0] == "完整脑"]
        白高 = [float(r[4]) for r in b行 if r[0] == "白脑"]
        白走 = [float(r[5]) for r in b行 if r[0] == "白脑"]
        左 = [0 - 0.17, 1 - 0.17]
        右 = [0 + 0.17, 1 + 0.17]
        轴[3].bar(左, [np.median(全高), np.median(全走)], 0.32, color="#2b7bba", label="innate wiring")
        轴[3].bar(右, [np.median(白高), np.median(白走)], 0.32, color="#b03a2e", label="background only")
        for i, v in enumerate(全高):
            轴[3].scatter([左[0]] * len(全高), 全高, c="k", s=14, zorder=3)
        for i, v in enumerate(白高):
            轴[3].scatter([右[0]] * len(白高), 白高, c="k", s=14, zorder=3)
        轴[3].scatter([左[1]] * len(全走), 全走, c="k", s=14, zorder=3)
        轴[3].scatter([右[1]] * len(白走), 白走, c="k", s=14, zorder=3)
        轴[3].axhline(0.0, c="k", lw=0.6)
        轴[3].axhline(0.20, ls="--", c="k", lw=0.8)
        轴[3].set_xticks([0, 1])
        轴[3].set_xticklabels(["stand up\n(final height, m)", "approach red\n(displacement, m)"], fontsize=7)
        轴[3].set_title("(d) same brain, instinct table cleared", fontsize=9)
        轴[3].set_ylabel("metres")
        轴[3].legend(fontsize=7)
    图.suptitle("Figure 2  Innate wiring alone closes the sensorimotor loop (5 network seeds)  "
                "--  (d) is the within-seed control", fontsize=10)
    存(图, "figure2_closed_loop.png")


# ---------------- 图 2：R2 ----------------
def 图2():
    t = 读文本("日志_前额叶多种子_新A.log") + 读文本("日志_前额叶多种子_新B.log")
    if "【汇总】" not in t:
        return
    行 = re.findall(r"种子 (\d+)  (①|②|③|④|⑤)[^\n]*?走路点火\s+(\d+)/(\d+) 拍 \| 首次点火 ([^|]*?)\| 路程 ([\d.]+) 米 \| 前额叶平均亮\s+(\d+) 个", t)
    if not 行:
        return
    幕名 = {"①": "red, prefrontal\nintact", "②": "red, prefrontal\nsilenced",
            "③": "black\nscreen", "④": "blue screen\n(persists)",
            "⑤": "blue screen\n(recomputed)"}
    组 = {}
    for 序号, 幕, 火, 总, 首, 路程, 前 in 行:
        组.setdefault(幕, []).append((int(火), int(总), float(路程), int(前)))
    键 = [k for k in ["①", "②", "③", "④", "⑤"] if k in 组]
    名 = [幕名[k] for k in 键]
    点火 = [np.array([a for a, b, c, d in 组[k]]) for k in 键]
    前额 = [np.mean([d for a, b, c, d in 组[k]]) for k in 键]
    x = np.arange(len(名))
    图, 轴 = plt.subplots(1, 2, figsize=(11.5, 3.6))
    for i, v in enumerate(点火):
        轴[0].bar(i, v.mean(), 0.55, color="#2b7bba")
        轴[0].scatter([i] * len(v), v, c="k", s=16, zorder=3)
        轴[0].text(i, v.mean() + 6, "%.0f/%d" % (v.mean(), 200), ha="center", fontsize=8)
    轴[0].set_xticks(x); 轴[0].set_xticklabels(名, fontsize=7.5)
    轴[0].set_ylabel("ticks with walking action lit (of 200)")
    轴[0].set_title("(a) silencing the prefrontal population abolishes walking;\nblue crosses the threshold only if the code persists", fontsize=8.5)
    轴[1].bar(x, 前额, 0.55, color="#7d3c98")
    轴[1].set_xticks(x); 轴[1].set_xticklabels(名, fontsize=7.5)
    轴[1].set_ylabel("mean active prefrontal cells")
    轴[1].set_title("(b) the ablation is specific:\nthe blue screen still drives prefrontal cells", fontsize=8.5)
    图.suptitle("Figure 3  Causal intervention (dots = individual network seeds)", fontsize=10)
    存(图, "figure3_prefrontal_ablation.png")


# ---------------- 图 3：R3 ----------------
def 图3():
    t = 读文本("日志_分阶段3.log")
    if "【汇总：每一格" not in t:
        return
    i = t.index("【汇总：每一格")
    块 = t[i:]
    行 = []
    for m in re.finditer(r"^(\S+)\s+\|\s+(\d+/\d+)\s+\|\s+(\d+/\d+)\s+\|\s+(\d+/\d+)\s+\|\s+(\d+/\d+)\s+\|\s+(\d+/\d+)\s*$",
                       块, re.M):
        行.append((m.group(1), [m.group(k) for k in range(2, 7)]))
    if not 行:
        return
    英 = {"A_只有运动库": "A  motor repertoire", "B_加平衡起身": "B  + balance/righting",
           "C_加看见红走": "C  + approach red", "D_加听觉": "D  + auditory",
           "E_加眼睛注视": "E  + gaze"}
    名 = [英.get(a, a) for a, b in 行]
    M = np.array([[int(c.split("/")[0]) / int(c.split("/")[1]) for c in b] for a, b in 行])
    列名 = ["stand up", "walk to red", "react to sound", "stop on tone", "gaze tracking"]
    图, 轴 = plt.subplots(figsize=(7.2, 3.2))
    图_ = 轴.imshow(M, cmap="Blues", vmin=0, vmax=1)
    轴.set_xticks(range(len(列名))); 轴.set_xticklabels(列名, fontsize=8, rotation=12)
    轴.set_yticks(range(len(名))); 轴.set_yticklabels(名, fontsize=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            轴.text(j, i, 行[i][1][j], ha="center", va="center", fontsize=8,
                   color="w" if M[i, j] > 0.6 else "k")
    图.colorbar(图_, ax=轴, shrink=0.8, label="fraction of seeds passing")
    轴.set_title("Figure 4  The repertoire grows as instinct groups are added (3 seeds per stage)", fontsize=10)
    存(图, "figure4_staged_repertoire.png")


# ---------------- 图 4：R4 ----------------
def 图4():
    t = 读文本("日志_渐变正式2.log")
    行 = re.findall(r"步长\s+(\d+)°，(开学习|关学习)，种子 (\d+) → 停在哪\s+(\d+)°", t)
    if not 行:
        return
    D = {}
    for 步, 学, 种, 停 in 行:
        D.setdefault((int(步), 学), []).append((int(种), int(停)))
    步长们 = sorted(set(k[0] for k in D))
    图, 轴 = plt.subplots(1, 2, figsize=(11.6, 4.1), gridspec_kw={"width_ratios": [1, 1.25]})
    w = 0.35
    for k, 学 in enumerate(["开学习", "关学习"]):
        值 = [D.get((b, 学), []) and np.mean([v for _, v in D[(b, 学)]]) or np.nan for b in 步长们]
        轴[0].bar(np.arange(len(步长们)) + (k - 0.5) * w, 值, w,
                  color="#2b7bba" if 学 == "开学习" else "#b03a2e",
                  label="plasticity ON" if 学 == "开学习" else "plasticity OFF")
        for i, b in enumerate(步长们):
            for _, v in D.get((b, 学), []):
                轴[0].scatter([i + (k - 0.5) * w], [v], c="k", s=16, zorder=3)
    轴[0].set_xticks(range(len(步长们))); 轴[0].set_xticklabels(["%d deg" % b for b in 步长们])
    轴[0].set_ylabel("hue at which the sweep stopped (deg)")
    轴[0].set_xlabel("hue step size"); 轴[0].legend(fontsize=8)
    轴[0].set_ylim(0, 180)
    轴[0].set_title("(a) how far the colour may drift", fontsize=9)

    # (b) 漂移前 vs 漂移后：同一颗脑子，哪些色相还能点着走路
    b2 = 读文本("日志_渐变边界.log")
    m = re.search(r"【合并（前后各一次，同一个脑）】(.*)$", b2, re.S)
    if m:
        对 = re.findall(r"(\d+)°：漂移前 ([\d、]+)；漂移后 ([\d、]+)", m.group(1))
        黑 = re.search(r"黑屏：漂移前 ([\d、]+)；漂移后 ([\d、]+)", m.group(1))
        if 对:
            def 通过率(z):
                v = [int(x) for x in z.split("、")]
                return sum(1 for a in v if a >= 6) / float(len(v)), len(v)
            x = np.arange(len(对))
            前 = [通过率(a)[0] for _, a, _ in 对]
            后 = [通过率(b)[0] for _, _, b in 对]
            n = 通过率(对[0][1])[1]
            轴[1].plot(x, 前, "o--", color="#b03a2e", ms=6, lw=1.5, label="before any drift")
            轴[1].plot(x, 后, "s-", color="#2b7bba", ms=6, lw=1.8, label="after drifting to 96 deg")
            轴[1].set_xticks(x); 轴[1].set_xticklabels(["%s" % a for a, _, _ in 对], fontsize=8)
            轴[1].set_xlabel("hue of the full-field colour (deg)")
            轴[1].set_ylabel("seeds that still start walking")
            轴[1].set_ylim(-0.40, 1.18)
            轴[1].axhline(1.0, ls=":", c="k", lw=0.8)
            轴[1].legend(fontsize=7.5, loc="lower center", ncol=2, framealpha=0.95,
                         borderpad=0.35, handlelength=1.6, columnspacing=1.0)
            if 黑:
                r0 = 通过率(黑.group(1))[0]
                r1 = 通过率(黑.group(2))[0]
                轴[1].scatter([len(对)], [r0], marker="o", c="#b03a2e", s=36, zorder=3)
                轴[1].scatter([len(对)], [r1], marker="s", c="#2b7bba", s=36, zorder=3)
                轴[1].set_xticks(list(x) + [len(对)])
                轴[1].set_xticklabels(["%s" % a for a, _, _ in 对] + ["black"], fontsize=8)
            轴[1].set_title("(b) the same brains, before and after the drift", fontsize=9)
    图.suptitle("Figure 5  Local plasticity moves the boundary, and moves it its own way", fontsize=10.5)
    图.tight_layout(rect=[0, 0, 1, 0.90])
    存(图, "figure5_gradient_recognition.png")


# ---------------- 图 5：R5 ----------------
def 图5():
    t = 读文本("日志_方位分辨力.log")
    图, 轴 = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1, 1.1]})
    有1 = False
    m = re.search(r"逐列看[^\n]*\n[^\n]*\n-+\n(.*)$", t, re.S)
    if m:
        行们 = [l for l in m.group(1).strip().split("\n") if l.strip()]
        数据 = []
        for l in 行们:
            格 = l.split("|")
            k = int(格[0].strip())
            值 = [float(v) if v.strip() != "—" else np.nan for v in 格[1].split()]
            数据.append((k, 值))
        x = np.arange(16)
        for k, 值 in 数据:
            轴[0].plot(x, 值, "o-", ms=5, lw=1.4,
                      label="layer %d" % k, color={1: "#2b7bba", 2: "#48a14d", 3: "#b03a2e"}.get(k, "k"))
        轴[0].set_xticks(x); 轴[0].set_xlabel("visual column (direction)")
        轴[0].set_ylabel("cross-column sharing of the direction name")
        轴[0].legend(fontsize=8); 轴[0].set_ylim(-0.05, 0.85)
        轴[0].set_title("(a) direction names: layer 1-2 exclusive, layer 3 blurred")
        有1 = True
    t2 = 读文本("日志_前几层.log")
    if "第3层" in t2:
        行 = re.findall(r"^\s*(\S+)\s*\|\s*(\S+)\s*\|\s*(\d+)\s+(\d+)\s+(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*$", t2, re.M)
        if 行:
            英文 = {"左3": "L3", "左2": "L2", "左1": "L1", "正中": "C",
                    "右1": "R1", "右2": "R2", "右3": "R3"}
            名 = [英文.get(r[0], r[0]) for r in 行]
            x = np.arange(len(名))
            轴[1].plot(x, [int(r[4]) for r in 行], "s-", color="#b03a2e", label="layer 3 (compressed)")
            轴[1].plot(x, [int(r[2]) for r in 行], "o-", color="#2b7bba", label="layer 1 (uncompressed)")
            轴[1].plot(x, [int(r[6]) for r in 行], "^--", color="#777777", label="input layer")
            轴[1].set_xticks(x); 轴[1].set_xticklabels(名, fontsize=8)
            轴[1].set_ylabel("newly active cells for a 0.31-cell ball")
            轴[1].legend(fontsize=8)
            轴[1].set_title("(b) a tiny object: layer 1 localises, layer 3 loses it")
            有1 = True
    if 有1:
        图.suptitle("Figure 6  The cost of hierarchical compression", fontsize=10)
        存(图, "figure6_compression_cost.png")


# ---------------- 图 6：R6（现场跑一遍，把轨迹记下来） ----------------
def 图6():
    import 实验_眼睛跟随 as 跟随
    图, 轴 = plt.subplots(1, 3, figsize=(12.5, 3.4))
    场景 = [("小横着飘", dict(球方位角=-25.0, 球半径=0.05, 球横速=0.5, 秒数=5.0),
             "(a) object crossing the field"),
            ("小突然出现", dict(球方位角=20.0, 球半径=0.05, 出现时刻=2.0, 秒数=5.0),
             "(b) object appearing at t = 2 s"),
            ("空场", dict(球方位角=0.0, 球半径=0.05, 出现时刻=999.0, 秒数=3.0),
             "(c) empty scene (control)")]
    留档 = []
    for i, (事, 参, 标题) in enumerate(场景):
        记 = 跟随.跑(事, 说=False, **参)
        留档.append((事, 参, 标题, 记))
        t = [r[0] for r in 记]
        球 = [r[1] for r in 记]
        眼 = [r[3] for r in 记]
        轴[i].plot(t, 眼, "-", color="#2b7bba", label="eye azimuth")
        轴[i].plot(t, 球, "-", color="#b03a2e", label="object in view")
        轴[i].axhline(0, ls=":", c="k", lw=0.8)
        轴[i].set_xlabel("time (s)"); 轴[i].set_ylabel("angle (deg)")
        轴[i].set_title(标题, fontsize=9); 轴[i].legend(fontsize=7)
    图.suptitle("Figure 7  Gaze tracking emerges from static reflexes (no tracking rule exists)", fontsize=10)
    存(图, "figure7_emergent_gaze.png")
    # 把这三条轨迹落盘：论文里 Figure 7 引用的 日志_眼睛跟随.log 就是它
    with io.open(根.parent / "logs" / "日志_眼睛跟随.log", "w", encoding="utf-8", newline="\n") as f:
        f.write("眼睛跟随（R6）：三幕轨迹，2b7bba 是眼的方位角，b03a2e 是球在视野里的方位角\n")
        f.write("这三条是 出一张图_论文_20260915.py 图6() 现场跑出来的，画 Figure 7 时写盘\n")
        for 事, 参, 标题, 记 in 留档:
            f.write("\n== %s  [%s %s] ==\n" % (标题, 事, 参))
            f.write("  秒     球方位角     眼方位角\n")
            for r in 记:
                f.write("  %5.2f  %9.2f  %9.2f\n" % (r[0], r[1], r[3]))
    print("  写好 日志_眼睛跟随.log")


# ---------------- 图 8：R8 ----------------
def 图8():
    import csv
    t = 读文本("日志_切断回响_有环.log")
    行 = re.findall(r"种子 (\d+)\s+(正常脑|无回响脑)\s*：起身末高 ([\d.]+)[^\n]*?"
                   r"见红走\s*([-+][\d.]+) 米、走路点火\s*(\d+)/(\d+) 拍 \| 盯球平均偏离 ([\d.]+) 度", t)
    图, 轴 = plt.subplots(1, 3, figsize=(13.4, 3.6))
    有 = False
    if 行:
        组 = {"正常脑": [], "无回响脑": []}
        for 种, 名, 高, 走, 火, 总, 偏 in 行:
            组[名].append((float(高), float(火) / float(总), float(偏)))
        名们 = ["stand up\nfrom prone", "walk to red\n(action lit)", "gaze tracking\n(mean error)"]
        判 = [lambda v: v[0] > 0.20, lambda v: v[1] > 0.0, lambda v: v[2] <= 12.0]
        标 = [lambda v: "%.3f m" % v[0], lambda v: "%.0f/200" % (v[1] * 200), lambda v: "%.1f deg" % v[2]]
        for i2 in range(3):
            for k, 名 in enumerate(["正常脑", "无回响脑"]):
                v = 组[名]
                过 = sum(1 for x in v if 判[i2](x))
                轴[0].bar(i2 + (k - 0.5) * 0.36, 过 / len(v), 0.36,
                         color="#2b7bba" if k == 0 else "#b03a2e",
                         label=("intact cortex" if k == 0 else "recurrent step removed") if i2 == 0 else None)
                中 = float(np.median([标[i2](x) for x in v]) if False else 0)
                _x = i2 + (k - 0.5) * 0.36 + (-0.10 if (i2 == 2 and k == 0) else
                                               (0.10 if (i2 == 2 and k == 1) else 0.0))
                轴[0].text(_x, 过 / len(v) + 0.03,
                          "%d/%d\n%s" % (过, len(v),
                                        "%.3f m" % float(np.median([x[0] for x in v])) if i2 == 0 else
                                        ("%.0f/200" % (float(np.median([x[1] for x in v])) * 200) if i2 == 1
                                         else "%.1f deg" % float(np.median([x[2] for x in v])))),
                          ha="center", fontsize=6.8)
        轴[0].set_xticks(range(3)); 轴[0].set_xticklabels(名们, fontsize=8)
        轴[0].set_ylim(0, 1.32); 轴[0].set_ylabel("fraction of seeds passing")
        轴[0].legend(fontsize=7.5, loc="upper right")
        轴[0].set_title("(a) reflexes survive, actions do not", fontsize=9)
        有 = True
    p = 根 / "日志_想还在吗_轨迹_有环.csv"
    if p.exists():
        行们 = list(csv.DictReader(io.open(p, encoding="utf-8")))
        幕名 = []
        for r in 行们:
            if r["幕名"] not in 幕名:
                幕名.append(r["幕名"])
        要画 = [m for m in 幕名 if m.startswith("B")] + [m for m in 幕名 if m.startswith("C")]
        英文幕 = {"B": "red removed at 0.5 s", "C": "red removed, action cleared at 1.0 s"}
        for j2, 名 in enumerate(要画[:2]):
            k = 1 + j2
            if k > 2:
                break
            秒 = sorted(set(float(r["秒"]) for r in 行们 if r["幕名"] == 名))
            def 曲线(列, 归一列=None):
                if 归一列 is None:
                    return [float(np.mean([int(r[列]) for r in 行们
                                          if r["幕名"] == 名 and float(r["秒"]) == s2])) for s2 in 秒]
                return [float(np.mean([int(r[列]) / max(1, int(r[归一列])) for r in 行们
                                      if r["幕名"] == 名 and float(r["秒"]) == s2])) for s2 in 秒]
            轴[k].plot(秒, 曲线("走路"), "-", color="#2b7bba", lw=2, label="walking action lit")
            轴[k].plot(秒, 曲线("视觉红数", "视觉红块"), "-", color="#b03a2e", lw=1.6,
                      label="visual code for red")
            轴[k].plot(秒, 曲线("前额叶红数", "前额叶红块"), "--", color="#7d3c98", lw=1.6,
                      label="prefrontal code")
            轴[k].axvline(0.5, ls=":", c="k", lw=0.9)
            轴[k].text(0.52, 1.05, "red removed", fontsize=6.5)
            if 名.startswith("C"):
                轴[k].axvline(1.0, ls=":", c="k", lw=0.9)
                轴[k].text(1.02, 1.10, "action cleared", fontsize=6.5)
            轴[k].set_ylim(-0.06, 1.22); 轴[k].set_xlabel("time (s)")
            轴[k].set_ylabel("fraction active")
            轴[k].set_title("(%s) %s" % ("bc"[j2], 英文幕.get(名[0], 名)), fontsize=8.5)
            轴[k].legend(fontsize=6.5, loc="center left")
            有 = True
    if 有:
        图.suptitle("Figure 9  Where the behaviour lives: reflexes are wired, actions are cortical trajectories", fontsize=10)
        存(图, "figure9_recurrence.png")


# ---------------- 图 7：R7 ----------------
def 图7():
    t = 读文本("日志_整体偏移_正前方.log")
    m = re.findall(r"^\s+(-?\d+) 列 \|\s+(\d+) \|\s+(\d+)/8 \|\s+([-+][\d.]+) \|\s+([-+][\d.]+)\s*$", t, re.M)
    图, 轴 = plt.subplots(1, 2, figsize=(11, 3.6))
    有 = False
    if m:
        Δ = [int(a) for a, b, c, d, e in m]
        球 = [float(e) for a, b, c, d, e in m]
        轴[0].plot(Δ, 球, "o-", color="#2b7bba", ms=6)
        轴[0].plot(Δ, [d * 6.25 for d in Δ], "k--", lw=1, label="1 cell per cell of shift")
        轴[0].set_xlabel("shift applied to every direction name (columns)")
        轴[0].set_ylabel("object resting position rel. to gaze (deg)")
        轴[0].legend(fontsize=8)
        轴[0].set_title("(a) a shifted innate map biases behaviour", fontsize=9)
        有 = True
    t2 = 读文本("日志_存活曲线.log")
    if "【存活率总表】" in t2:
        块 = t2[t2.index("【存活率总表】"):]
        m2 = re.findall(r"^(\S+)\s+\|\s+(\d+)%\s+\|\s+(\d+/\d+)\s+\|\s+(\d+/\d+)\s+\|\s+(\d+/\d+)\s*$", 块, re.M)
        if m2:
            方式们 = []
            for a, b_, c_, d_, e_ in m2:
                if a not in 方式们:
                    方式们.append(a)
            英文名 = {"随机乱接": "random", "挪一格": "one column over"}
            for 方式, 色 in zip(方式们, ["#2b7bba", "#b03a2e"]):
                行 = [(int(b_), c_, d_, e_) for a, b_, c_, d_, e_ in m2 if a == 方式]
                行.sort()
                x = [p for p, c_, d_, e_ in 行]
                曲线 = [
                    ("walk to red", [int(c_.split("/")[0]) / int(c_.split("/")[1]) for p, c_, d_, e_ in 行], "o-"),
                    ("react to sound", [int(d_.split("/")[0]) / int(d_.split("/")[1]) for p, c_, d_, e_ in 行], "s--"),
                    ("gaze tracking", [int(e_.split("/")[0]) / int(e_.split("/")[1]) for p, c_, d_, e_ in 行], "^:"),
                ]
                for 标, 值, 样 in 曲线:
                    轴[1].plot(x, 值, 样, ms=5, color=色,
                              label="%s / %s" % (英文名.get(方式, 方式), 标))
            轴[1].set_xlabel("fraction of instinct table re-wired (%)")
            轴[1].set_ylabel("fraction of draws still working")
            轴[1].legend(fontsize=7); 轴[1].set_ylim(-0.05, 1.05)
            轴[1].set_title("(b) dose-response: re-wiring the innate table", fontsize=9)
            有 = True
    if 有:
        图.suptitle("Figure 8  Blurred innate wiring", fontsize=10)
        存(图, "figure8_blurred_wiring.png")


要跑 = [int(a) for a in (sys.argv[1].split(",") if len(sys.argv) > 1 else "1,2,3,4,5,6,7,8")]
表 = {1: 图1, 2: 图2, 3: 图3, 4: 图4, 5: 图5, 6: 图6, 7: 图7, 8: 图8}
for k in 要跑:
    if k not in 表:
        continue
    print("画第 %d 张 ..." % k, flush=True)
    try:
        表[k]()
    except Exception as e:
        import traceback
        print("  第 %d 张画不出来：%s" % (k, e))
        traceback.print_exc()
print("目录：", 出图目录)