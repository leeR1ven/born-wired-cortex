# -*- coding: utf-8 -*-
"""实验_视觉线性4_半径.py —— 第一层该看多宽（颜色区分度）

★ 为什么试半径

   输入环上的排布是：一个大像素 = R 的 10 对 + G 的 10 对 + B 的 10 对 = 30 对 = 60 个神经元。
   半径 10 时，一个细胞只看到**一个通道**里的一小段 → 它量的是"这里多亮"，
   量不出"这里红多还是蓝多"（红和蓝都被它当成"亮"）。
   半径 30 时，一个细胞正好看到**整个大像素的三个通道** → 它才有机会算
   "红亮、蓝不亮"这件事。这正是用户要的"保证区分度"。

命令：python 实验_视觉线性4_半径.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import 视觉前处理_visual_preprocess as V
from 实验_视觉线性 import 试验网, 探针, 中红亮度, 档数, 每层数, 打分


def 建(半径, 门槛, 散布=0.6, 视野=600, 种子=20260914):
    网 = 试验网(归一="平方和", 采样数=0, 半径=半径, 输出感受野=视野, 阈值初值=门槛)
    rng = np.random.default_rng(种子 + 7)
    网.阈值 = [np.clip(t + rng.uniform(-散布, 散布, 每层数), 0.01, None) for t in 网.阈值]
    return 网


def 报(标题, 网):
    出 = {名: np.asarray(网.前向传播(V.图像转信号(g)), dtype=bool) for 名, g in 探针.items()}
    基准 = 出["中有红"]
    n = int(基准.sum())
    渐 = []
    for 档 in range(10, -1, -1):
        o = np.asarray(网.前向传播(V.图像转信号(中红亮度(档))), dtype=bool)
        渐.append(float((o & 基准).sum()) / max(n, 1))
    单, 贴, 跨 = 打分(渐)
    亮 = {名: 100 * float(出[名].sum()) / 每层数 for 名 in 探针}
    误 = {名: 100 * float((出[名] & 基准).sum()) / max(n, 1)
          for 名 in ("左有红", "右有红", "中蓝", "全绿", "白", "黑")}
    print(f"[{标题}] 中有红 {n}（{100 * n / 每层数:.0f}%）| 亮几个 "
          + " ".join(f"{名}{亮[名]:.0f}%" for 名 in ("黑", "中有红", "左有红", "中蓝", "全绿", "全红", "白")))
    print(f"   误亮「中有红」： " + " ".join(f"{名}{误[名]:.0f}%" for 名 in 误))
    print(f"   红渐淡: " + " ".join("%3d" % round(100 * v) for v in 渐)
          + f"   -> 单调{单:.0%} 直线{贴:.2f}")
    return 贴, 单, 误, n


def main():
    for 半径 in (10, 20, 30, 40):
        for 门槛 in (0.3, 0.4, 0.5):
            报(f"半径{半径} 门槛{门槛} 散布0.6", 建(半径, 门槛))


if __name__ == "__main__":
    main()