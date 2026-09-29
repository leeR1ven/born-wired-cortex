# -*- coding: utf-8 -*-
"""诊断_出生抑制权重扫一扫.py —— 修完抑制接线后本能变弱，扫一个旋钮看能不能把余量找回来

只动「抑制出生权重」（随机抑制边的出生权重，0.15）。它**不碰**本能表写的那些抑制
（那些是按 |力度|/抑制强度 算的，与这个数无关），所以是一个干净的单一旋钮。

用法：cd model; python 诊断_出生抑制权重扫一扫.py
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np
import 皮层连接_cortex_links as C
import 实验_闭环前提_多种子 as E

种子们 = [20260914, 20260915, 20260916, 20260917, 20260918]
原始 = C.抑制出生权重
print("=" * 96)
print("扫「抑制出生权重」：站起来 / 朝红走 / 什么都不给")
print("=" * 96)
for v in (0.15, 0.12, 0.10, 0.08, 0.06):
    C.抑制出生权重 = v
    站 = []; 走 = []; 末 = []
    for s in 种子们:
        r = E.跑一趟(s)
        站.append(r["站起来末高"]); 走.append(r["朝红走了"]); 末.append(r["瘫掉末高"])
    站 = np.array(站); 走 = np.array(走); 末 = np.array(末)
    print(f"抑制出生权重 {v:.2f} | 站起来 {int((站 > 0.20).sum())}/{len(站)}"
          f"（中位 {np.median(站):.3f} 最低 {站.min():.3f}）"
          f" | 朝红走 {int((走 > 0.30).sum())}/{len(走)}（中位 {np.median(走):+.2f} 最少 {走.min():+.2f}）"
          f" | 瘫掉 {int((末 < 0.20).sum())}/{len(末)}", flush=True)
C.抑制出生权重 = 原始