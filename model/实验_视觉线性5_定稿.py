# -*- coding: utf-8 -*-
"""实验_视觉线性5_定稿.py —— 最后比较几种"门槛散布"的写法，挑一个定下来。

  · 加性散布：门槛 = 中心 ± 散布（均匀）——简单，但中心低时会出现负数，被压到 0 就一大批细胞永远亮着
  · 乘性散布：门槛 = 中心 × (0.2 ~ 1.8 之间的随机数)——每个细胞的门槛都正、分布铺得开

命令：python 实验_视觉线性5_定稿.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import 视觉前处理_visual_preprocess as V
from 实验_视觉线性 import 试验网, 探针, 中红亮度, 档数, 每层数, 打分


def 建(门槛, 散布=0.0, 乘性=None, 采样数=64, 视野=600, 半径=10, 种子=20260914):
    网 = 试验网(归一="平方和", 采样数=采样数, 半径=半径, 输出感受野=视野, 阈值初值=门槛)
    rng = np.random.default_rng(种子 + 7)
    if 乘性 is not None:
        网.阈值 = [t * rng.uniform(*乘性, size=每层数) for t in 网.阈值]
    elif 散布 > 0:
        网.阈值 = [np.clip(t + rng.uniform(-散布, 散布, 每层数), 0.02, None) for t in 网.阈值]
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


if __name__ == "__main__":
    报("加性 0.4±0.6（上一轮最好的那个）", 建(0.4, 散布=0.6))
    for 中心 in (0.5, 0.7, 0.9, 1.2):
        报(f"乘性 {中心}×(0.2~1.8)", 建(中心, 乘性=(0.2, 1.8)))
    for 中心 in (0.7, 0.9, 1.2):
        报(f"乘性 {中心}×(0.4~1.6)", 建(中心, 乘性=(0.4, 1.6)))