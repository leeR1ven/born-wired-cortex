# -*- coding: utf-8 -*-
"""实验_放大_走路边界.py —— 放大画面以后，"看见红就往前走"那条边界还在不在、在哪。

做法：**直接复用** 实验_渐变_边界.py 里的 测一遍()（一个字没改），
但只测"漂移前"那一列 —— 也就是这颗脑子出生、只看了 40 拍纯红的时候，
每个色相还能不能点着走路（给 8 拍，走路那串时刻亮几拍，>=6 算认得出）。

为什么单独做一个：R4 那张表的"漂移前"这一列是**不需要跑几千拍漂移**就有的，
所以放大以后的对照可以很便宜地做出来（见论文 R11 说的"要重新标定 + 复跑"）。

命令：python 实验_放大_走路边界.py 20260914,20260915
      （不写种子 = 论文用的那五个）
"""
from __future__ import annotations

import sys
import time

import numpy as np

import 本能工具_instincts as 本能
import 视觉前处理_visual_preprocess as 视觉
import 实验_渐变_边界 as 边界
import 实验_颜色渐变 as 渐变
import 身体_go2 as 身体
import 主循环_完整的一拍 as 主循环

种子们 = [20260914, 20260915, 20260916, 20260917, 20260918]
色相们 = 边界.色相们
探头 = 边界.探头
判 = 6          # 8 拍里亮 >= 6 拍才算认得出


def 主():
    if len(sys.argv) > 1 and sys.argv[1].strip():
        种子们[:] = [int(a) for a in sys.argv[1].split(",")]
    print("=" * 104)
    print("画面 %d x %d（一格 %.2f 度宽）；种子：%s"
          % (视觉.大像素行数, 视觉.大像素列数,
             100.0 / 视觉.大像素列数, "、".join(str(s) for s in 种子们)))
    print("=" * 104)
    出 = []
    for s in 种子们:
        起 = time.time()
        本能.建网种子 = s
        脑 = 主循环.脑(学=True, 说=False)
        身 = 身体.身体()
        身.摆成站姿()
        身.走(0.5, None)
        时刻 = 渐变.拿时刻(脑)
        # 和 跑一趟() 的开头一模一样：先看 40 拍纯红（这一步是真会学的）
        渐变.跑几拍(脑, 身, 渐变.彩色块(0), 40, 时刻, 渐变.空统计())
        前 = 边界.测一遍(脑, 身, 时刻)
        出.append((s, 前))
        print("  种子 %d：%s   （%.0f 秒）"
              % (s, "、".join("%s=%d/%d" % ("黑" if k == "黑" else str(k), 前[k], 探头)
                             for k in 色相们 + ["黑"]), time.time() - 起), flush=True)
    print()
    print("=" * 104)
    print("【总表】点着几拍 / %d 拍" % 探头)
    头 = "  色相 |" + "".join("%9s" % ("种子%d" % s) for s, _ in 出) + " | 能点着的种子"
    print(头)
    print("-" * len(头))
    for k in 色相们 + ["黑"]:
        列 = [前[k] for _, 前 in 出]
        名 = "%6s" % ("黑屏" if k == "黑" else "%d°" % k)
        print("%s |" % 名 + "".join("%6d/%-3d" % (a, 探头) for a in 列)
              + " |  %d/%d" % (sum(a >= 判 for a in 列), len(列)))
    print()
    print("（判断标准和 R4 一样：8 拍里亮 >=%d 拍，才算这个色相还能点着走路）" % 判)
    return 出


if __name__ == "__main__":
    出 = 主()