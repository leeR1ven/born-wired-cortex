# -*- coding: utf-8 -*-
"""诊断_前额叶自压_窗口.py —— 找"自压"那个窗口有多宽（多个种子）。

一档自压 = 一个脑 + 四种画面各 120 拍。要看的是：
  · 前额叶亮多少个（希望是几百，不是几千，更不是全灭）
  · 稳不稳（每 20 拍取一个数，希望是平的，不是一直涨）
  · 红屏还走不走路

命令：python 诊断_前额叶自压_窗口.py
"""
from __future__ import annotations

import sys

import 诊断_前额叶自压 as 前


def main():
    种子们 = [20260914, 20260915, 20260916]
    档们 = [0.0005, 0.001, 0.0015]
    print("=" * 96)
    print("自压窗口：%d 个种子 × %d 档" % (len(种子们), len(档们)))
    print("=" * 96, flush=True)
    for 自压 in 档们:
        for 种子 in 种子们:
            出 = 前.跑一档(自压, 0.0, 种子, 拍数=120)
            print("\n【自压 %.4g ｜ 种子 %d】" % (自压, 种子))
            for 名, _ in 前.条件:
                采样, 点火, 末 = 出[名]
                print("   %s：前额叶 %s（末了 %d）｜走路点火 %d/120"
                      % (名, " → ".join(str(x) for x in 采样), 末, 点火), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())