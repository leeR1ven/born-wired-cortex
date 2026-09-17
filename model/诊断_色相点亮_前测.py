# -*- coding: utf-8 -*-
"""诊断_色相点亮_前测.py —— 不漂移，只问「哪些色相能点着走路」，一个种子一个种子地报。

和 实验_渐变_边界.py 的「漂移前」那一趟完全一样（同样 40 拍热身、同样 8 拍探头、同样学=True），
只是**不跑漂移**，所以快得多。配置由环境变量 前额叶不保留 决定：
    不设      = 前额叶照常（带环、会累加）
    设成 1    = 前额叶每拍重算（R1~R8 那一档）

命令：python 诊断_色相点亮_前测.py [种子,种子,...]
"""
from __future__ import annotations

import os
import sys

import 本能工具_instincts as 本能
import 实验_渐变_边界 as 边界
import 实验_颜色渐变 as 渐变
import 身体_go2 as 身体
import 主循环_完整的一拍 as 主循环

种子们默认 = [20260914, 20260915, 20260916, 20260917, 20260918]


def main():
    种子们 = 种子们默认
    if len(sys.argv) > 1 and sys.argv[1].strip():
        种子们 = [int(a) for a in sys.argv[1].split(",")]
    print("=" * 100)
    print("哪些色相能点着走路（不漂移）：%d 个种子；前额叶 %s"
          % (len(种子们), "每拍重算（不保留）" if os.environ.get("前额叶不保留") == "1" else "照常（带环）")
          + "；学=%s" % ("关" if os.environ.get("不学") == "1" else "开"))
    print("=" * 100, flush=True)
    出 = {}
    for s in 种子们:
        本能.建网种子 = s
        脑 = 主循环.脑(学=(os.environ.get("不学") != "1"), 说=False)
        身 = 身体.身体()
        身.摆成站姿()
        身.走(0.5, None)
        时刻 = 渐变.拿时刻(脑)
        渐变.跑几拍(脑, 身, 渐变.彩色块(0), 40, 时刻, 渐变.空统计())
        r = 边界.测一遍(脑, 身, 时刻)
        出[s] = r
        print("种子 %d：" % s + "  ".join(
            "%s %d/%d" % (("黑" if k == "黑" else "%d°" % k), r[k], 边界.探头)
            for k in (边界.色相们 + ["黑"])), flush=True)
    print("\n【汇总：有几个种子点着（>= %d 拍算点着）】" % 边界.探头)
    for k in (边界.色相们 + ["黑"]):
        数 = sum(1 for s in 出 if 出[s][k] >= 边界.探头 * 0.75)
        print("  %6s : %d/%d 个种子   %s" % (("黑屏" if k == "黑" else "%d°" % k), 数, len(出),
              "、".join(str(出[s][k]) for s in 种子们)))
    return 出


if __name__ == "__main__":
    出 = main()
