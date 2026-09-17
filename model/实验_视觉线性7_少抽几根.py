# -*- coding: utf-8 -*-
"""实验_视觉线性7_少抽几根.py —— 抽得少 -> 细胞之间差别大 -> 又能稀疏又能连续

   每个细胞只从视野里抽几根来连：抽得越少，每个细胞"看上"的东西越不一样，
   它们的驱动值就铺得越开 —— 高门槛卡下去还剩得少，但灭的时候仍然是一个个灭。

命令：python 实验_视觉线性7_少抽几根.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import 视觉前处理_visual_preprocess as V
from 实验_视觉线性 import 试验网, 探针, 中红亮度, 档数, 每层数, 打分


def 建(中心, 散布, 采样数, 视野=600, 半径=10):
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
          f" 中蓝{亮['中蓝']:.0f}% 白{亮['白']:.0f}% | 误亮 左{误['左有红']:.0f}% 右{误['右有红']:.0f}%"
          f" 中蓝{误['中蓝']:.0f}% 全绿{误['全绿']:.0f}%")
    print(f"   红渐淡: " + " ".join("%3d" % round(100 * v) for v in 渐) + f"  -> 单调{单:.0%} 直线{贴:.2f}")
    return 贴, 单, n


if __name__ == "__main__":
    好 = []
    for 采样数 in (8, 16, 32):
        for 中心 in (0.6, 0.9, 1.2):
            贴, 单, n = 报(f"抽{采样数}根 门槛{中心}±1.0", 建(中心, 1.0, 采样数))
            好.append((贴, 单, n, 采样数, 中心))
    print("\n=== 排序 ===")
    for 贴, 单, n, 采样数, 中心 in sorted(好, reverse=True):
        print(f"   抽{采样数}根 门槛{中心}±1.0 中有红{100*n/每层数:.1f}% 直线{贴:.2f} 单调{单:.0%}")