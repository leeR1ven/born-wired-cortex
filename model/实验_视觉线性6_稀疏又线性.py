# -*- coding: utf-8 -*-
"""实验_视觉线性6_稀疏又线性.py —— 找"又稀疏（细胞少）又连续（不一起灭）"的那一点

   门槛中心 调"亮几个"（越大越稀疏）；门槛散布 调"一个个灭还是一起灭"（越大越连续）。
   两个一起调，才能又少又连续。
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import 视觉前处理_visual_preprocess as V
from 实验_视觉线性 import 试验网, 探针, 中红亮度, 档数, 每层数, 打分


def 建(中心, 散布, 采样数=64, 视野=600, 半径=10):
    网 = 试验网(归一="平方和", 采样数=采样数, 半径=半径, 输出感受野=视野, 阈值初值=0.0)
    rng = np.random.default_rng(20260914 + 7)
    网.阈值 = [np.clip(中心 + 散布 * rng.uniform(-1, 1, 每层数), 0.02, None) for _ in range(网.层数 - 1)]
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
    误 = {名: 100 * float((出[名] & 基准).sum()) / max(n, 1) for 名 in ("左有红", "右有红", "中蓝", "全绿")}
    print(f"[{标题}] 中有红 {n}（{100 * n / 每层数:.1f}%）| 亮几个 黑{亮['黑']:.0f}% 中红{亮['中有红']:.0f}%"
          f" 中蓝{亮['中蓝']:.0f}% 全绿{亮['全绿']:.0f}% 白{亮['白']:.0f}%"
          f" | 误亮 中蓝{误['中蓝']:.0f}% 全绿{误['全绿']:.0f}%")
    print(f"   红渐淡: " + " ".join("%3d" % round(100 * v) for v in 渐) + f"  -> 单调{单:.0%} 直线{贴:.2f}")
    return 贴, 单, n


if __name__ == "__main__":
    for 中心 in (0.4, 0.5, 0.6, 0.8):
        for 散布 in (0.6, 0.9, 1.2):
            报(f"门槛{中心}±{散布}", 建(中心, 散布))