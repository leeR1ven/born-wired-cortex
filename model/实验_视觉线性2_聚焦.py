# -*- coding: utf-8 -*-
"""实验_视觉线性.py —— 「外界画面线性地变，视觉皮层是不是也线性地跟着变？」（聚焦版）

量三件事：
  1) 红一点点调暗（档 10→0），"中有红"那组细胞被点亮的比例跟不跟着降（要单调、要像直线）
  2) 亮几个稳不稳（黑=0，看得见的东西都差不多多）
  3) 认不认得出"红在哪"（"左有红/右有红"会不会把"中有红"那组误亮太多）

命令：python 实验_视觉线性.py
"""
from __future__ import annotations

import pathlib
import sys
import time

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import 视觉前处理_visual_preprocess as V
from 实验_视觉线性 import 试验网, 图, 探针, 中红亮度, 档数, 每层数

红左列, 红右列 = 5 * (1920 // 16), 11 * (1920 // 16)


def 评估(网):
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


def 打分(渐):
    arr = np.asarray(渐)
    s = np.linspace(1.0, 0.0, 档数)
    return (float(np.mean(np.diff(arr) <= 0.02)),
            float(1.0 - np.mean((arr - s) ** 2) / np.var(s)),
            float(arr[0] - arr[-1]))


def 报(标题, 网):
    出, 渐, n, 秒 = 评估(网)
    单, 贴, 跨 = 打分(渐)
    基准 = 出["中有红"]
    误 = {名: (100 * float((出[名] & 基准).sum()) / max(n, 1))
          for 名 in ("全红", "左有红", "右有红", "中蓝", "黑")}
    亮 = {名: 100 * float(出[名].sum()) / 每层数 for 名 in 探针}
    print(f"[{标题}] 中有红 {n} 个（{100 * n / 每层数:.0f}%），一帧 {1000 * 秒:.0f}ms")
    print(f"   亮几个： " + " ".join(f"{名}{亮[名]:.0f}%" for 名 in ("黑", "中有红", "左有红", "右有红", "全红", "中蓝", "白")))
    print(f"   会把「中有红」误亮多少： " + " ".join(f"{名}{误[名]:.0f}%" for 名 in 误))
    print(f"   红渐淡 视比例(%): " + " ".join("%3d" % round(100 * v) for v in 渐))
    print(f"   -> 单调 {单:.0%}，像一条直线 {贴:.2f}，跨度 {跨:.0%}")
    return 单, 贴, 跨, 渐, n, 秒, 误


def main():
    print(f"视觉输入：24x16 x RGB x 10对 = {每层数} 神经元/层\n")
    print("=== 生产用的那张网（总和归一 + 取前几名已去掉 + 阈值0.05）===")
    报(f"现网 阈值{V.阈值初值}", V.网)

    print("\n=== 平方和归一，抽 64 个输入连线：视野 x 门槛 ===")
    for 视野 in (60, 200, 600):
        for 阈值 in (0.2, 0.25, 0.3):
            网 = 试验网(归一="平方和", 采样数=64, 输出感受野=视野, 阈值初值=阈值)
            报(f"视野{视野} 阈值{阈值}", 网)


if __name__ == "__main__":
    main()