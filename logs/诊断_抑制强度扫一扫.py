# -*- coding: utf-8 -*-
"""诊断_抑制强度扫一扫.py —— 修完抑制接线后随机抑制边变多，用「抑制强度」这一个旋钮重新配平

为什么这个旋钮是干净的：本能表写的抑制，进网时的权重是 |力度| / 抑制强度，
而它出电流时又乘回 抑制强度 —— 两头抵消。所以 抑制强度 只缩放**随机抑制边**。

用法：cd model; python -X utf8 诊断_抑制强度扫一扫.py
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np
import 皮层连接_cortex_links as C
import 实验_闭环前提_多种子 as E

种子们 = [20260914, 20260915, 20260916, 20260917, 20260918]
原始 = C.抑制强度
print("抑制强度扫描：站起来 / 朝红走 / 什么都不给要不要瘫掉")
for v in (1.5, 1.2, 1.0, 0.8, 0.6):
    C.抑制强度 = v
    站 = []; 走 = []; 末 = []
    for s in 种子们:
        r = E.跑一趟(s)
        站.append(r["站起来末高"]); 走.append(r["朝红走了"]); 末.append(r["瘫掉末高"])
    站 = np.array(站); 走 = np.array(走); 末 = np.array(末)
    print(f"SUPPRESS {v:.2f} | stand {int((站 > 0.20).sum())}/{len(站)} med {np.median(站):.3f} min {站.min():.3f}"
          f" | walk {int((走 > 0.30).sum())}/{len(走)} med {np.median(走):+.2f} min {走.min():+.2f}"
          f" | limp {int((末 < 0.20).sum())}/{len(末)}", flush=True)
C.抑制强度 = 原始