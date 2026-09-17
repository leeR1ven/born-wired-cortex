# -*- coding: utf-8 -*-
"""实验_方位分辨力_按层.py —— 每一层到底把"东西在哪一边"分辨得多清楚？

★ 为什么量这个：眼睛转不转得过去，靠的是"第 c 列有东西"这个名字**只属于 c 那一列**。
  如果某一层里，一个方向亮的细胞在别的方向也亮，那这一层的名字就是糊的 ——
  本能连接写上去也分不出方向。

★ 量法（不建皮层，只跑视觉网前向传播，几秒钟）：
  · 每个方向的名字 = 那一层里"比眼前什么都没有多亮"的那些细胞（和 标定 同一个做法）
  · 糊度 = 这个名字里，有多大比例的细胞**也在别的方向的名字里**（0 = 完全专属，1 = 全糊）
  · 再顺带量"整张图一共变了几格"（每层亮了多少细胞）

命令：python 实验_方位分辨力_按层.py
"""
from __future__ import annotations

import numpy as np

import 本能工具_instincts as 本能
import 视觉前处理_visual_preprocess as 视觉

列数 = int(视觉.大像素列数)
列宽 = 100.0 / 列数
球半径 = 0.05


def 层激活(图, k):
    末 = np.asarray(视觉.网.前向传播(视觉.图像转信号(图)), dtype=bool)
    if k == 3:
        return 末
    return np.asarray(视觉.网.普通层激活[k - 1], dtype=bool).copy()


列中心 = [-50.0 + 列宽 * (c + 0.5) for c in range(列数)]
视觉.网.前向传播(视觉.图像转信号(本能._空画面()))
空 = {k: 层激活(本能._空画面(), k) for k in (1, 2, 3)}

print("小球半径 %.2f 米（%.2f 个格子）；每层的方位分辨力" % (球半径, np.degrees(2 * np.arctan2(球半径, 3.0)) / 列宽))
print()
print("%6s | %8s | %10s | %12s | %s" % ("层", "亮了多少", "有位子的列", "糊度(中位)", "糊度(最差那列)"))
print("-" * 74)
汇总 = {}
for k in (1, 2, 3):
    名 = {}
    for c in range(列数):
        图 = 本能._球在(np.radians(列中心[c]), 半径=球半径)
        名[c] = set(np.nonzero(层激活(图, k) & ~空[k])[0].tolist())
    糊 = {}
    for c in range(列数):
        if not 名[c]:
            continue
        别人 = set()
        for d in range(列数):
            if d != c:
                别人 |= 名[d]
        糊[c] = len(名[c] & 别人) / len(名[c])
    有 = sorted(糊)
    汇总[k] = (名, 糊)
    print("%6d | %8d | %10d | %12s | %s"
          % (k, int(空[k].sum()),
             len(有),
             "%.2f" % float(np.median(list(糊.values()))) if 糊 else "—",
             "、".join("%d列%.2f" % (c, 糊[c]) for c in sorted(糊, key=lambda x: -糊[x])[:3])))

print()
print("逐列看（糊度：0 = 这个名字只在它自己那一列亮，1 = 别的列也全亮）")
print("%6s | %s" % ("层", "  ".join("%5s" % ("%d列" % c) for c in range(列数))))
print("-" * 74)
for k in (1, 2, 3):
    格 = []
    for c in range(列数):
        v = 汇总[k][1].get(c)
        格.append("%5s" % ("—" if v is None else "%.2f" % v))
    print("%6d | %s" % (k, "  ".join(格)))