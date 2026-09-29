# -*- coding: utf-8 -*-
"""诊断_前额叶自压_惯性.py —— 带惯性的自压，扫一遍找"窗口"。

为什么分开一份：第一版自压是**瞬时**的（这一拍亮几个就压多少），实测在 0.004 就
全灭/全亮来回跳（前额叶整片变成 0）。现在自压带惯性（一阶低通，平滑 0.9），
这一份专门扫带惯性的那几档，找"前额叶只剩几百个、而且稳得住"的那一档。

命令：python 诊断_前额叶自压_惯性.py [种子] [拍数]
"""
from __future__ import annotations

import sys

import 诊断_前额叶自压 as 前

种子们默认 = [20260914]


def main():
    种子们 = 种子们默认
    拍数 = 120
    if len(sys.argv) > 1 and sys.argv[1].strip():
        种子们 = [int(a) for a in sys.argv[1].split(",")]
    if len(sys.argv) > 2 and sys.argv[2].strip():
        拍数 = int(sys.argv[2])
    档们 = [(0.002, 0.0), (0.004, 0.0), (0.008, 0.0), (0.02, 0.0)]
    print("=" * 96)
    print("前额叶自压（带惯性，平滑 0.9）：%d 个种子，每档四种画面各 %d 拍" % (len(种子们), 拍数))
    print("=" * 96, flush=True)
    for 种子 in 种子们:
        for 自压, 目标 in 档们:
            出 = 前.跑一档(自压, 目标, 种子, 拍数=拍数)
            print("\n【种子 %d ｜ 前额叶自压 %.4g  目标 %.4g】" % (种子, 自压, 目标))
            for 名, _ in 前.条件:
                采样, 点火, 末 = 出[名]
                print("   %s：前额叶 %s（末了 %d）｜走路点火 %d/%d"
                      % (名, " → ".join(str(x) for x in 采样), 末, 点火, 拍数), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())