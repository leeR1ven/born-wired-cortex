# -*- coding: utf-8 -*-
"""实验_视觉线性3_门槛散布.py —— 神经元的放电门槛本来就该各不相同

★ 为什么要试这个

   上面两轮已经确定：把权重按**平方和**归一（而不是按总和 → 取平均）之后，
   红一点点暗下去，视皮层就一点点暗下去（单调、连续），不再"忽高忽低"。
   但曲线还是偏陡：一暗到某个档位就整片灭掉。

   原因：现在**每一层所有神经元用同一个门槛**。同一层的细胞"偏好的亮度"全都
   卡在同一个地方（温度码 ± 是对称的，零点和门槛一卡就卡在中间那档），
   所以它们是一起翻的，不是一个个翻的。

   真实的神经元本来就**每个细胞门槛不一样**（兴奋性不同）。
   门槛一出生就各不相同 → 每个细胞"偏好的亮度"散布在整个范围里
   → 亮度一路降下来，细胞就一个一个灭，而不是一起灭。

命令：python 实验_视觉线性3_门槛散布.py
"""
from __future__ import annotations

import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import 视觉前处理_visual_preprocess as V
from 实验_视觉线性 import 试验网, 探针, 中红亮度, 档数, 每层数, 打分

红左列, 红右列 = 5 * (1920 // 16), 11 * (1920 // 16)


def 建(阈值中心, 散布, 采样数=64, 视野=600, 种子=20260914):
    网 = 试验网(归一="平方和", 采样数=采样数, 输出感受野=视野, 阈值初值=阈值中心)
    if 散布 > 0:
        rng = np.random.default_rng(种子 + 7)
        网.阈值 = [np.clip(t + rng.uniform(-散布, 散布, 每层数), 0.01, None)
                 for t in 网.阈值]
    return 网


def 量(网):
    t0 = time.perf_counter()
    出 = {名: np.asarray(网.前向传播(V.图像转信号(g)), dtype=bool) for 名, g in 探针.items()}
    秒 = (time.perf_counter() - t0) / len(探针)
    基准 = 出["中有红"]
    n = int(基准.sum())
    渐 = []
    for 档 in range(10, -1, -1):
        o = np.asarray(网.前向传播(V.图像转信号(中红亮度(档))), dtype=bool)
        渐.append(float((o & 基准).sum()) / max(n, 1))
    return 出, 渐, n, 秒


def 报(标题, 网):
    出, 渐, n, 秒 = 量(网)
    单, 贴, 跨 = 打分(渐)
    亮 = {名: 100 * float(出[名].sum()) / 每层数 for 名 in 探针}
    基准 = 出["中有红"]
    误 = {名: 100 * float((出[名] & 基准).sum()) / max(n, 1) for 名 in ("左有红", "右有红", "中蓝", "全红")}
    print(f"[{标题}] 中有红 {n}（{100 * n / 每层数:.0f}%）| 亮几个 "
          + " ".join(f"{名}{亮[名]:.0f}%" for 名 in ("黑", "中有红", "左有红", "右有红", "中蓝", "全红"))
          + f" | 一帧{1000 * 秒:.0f}ms")
    print(f"   误亮中有红： " + " ".join(f"{名}{误[名]:.0f}%" for 名 in 误))
    print(f"   红渐淡: " + " ".join("%3d" % round(100 * v) for v in 渐)
          + f"   -> 单调{单:.0%} 直线{贴:.2f} 跨度{跨:.0%}")
    return 贴, 单, 渐, n


def main():
    print("=== 固定门槛（现在这样）vs 每格神经元的门槛各不相同 ===")
    好 = []
    for 中心 in (0.25, 0.4, 0.6, 0.8):
        for 散布 in (0.0, 0.2, 0.4, 0.6):
            贴, 单, 渐, n = 报(f"门槛{中心} 散布{散布}", 建(中心, 散布))
            if 单 >= 0.99:
                好.append((贴, 中心, 散布, 渐, n))
    print("\n=== 排前几名（单调 100% 的）===")
    for 贴, 中心, 散布, 渐, n in sorted(好, reverse=True)[:6]:
        print(f"   门槛{中心} 散布{散布}  直线{贴:.2f}  中有红占{100 * n / 每层数:.0f}%  "
              + " ".join("%3d" % round(100 * v) for v in 渐))


if __name__ == "__main__":
    main()