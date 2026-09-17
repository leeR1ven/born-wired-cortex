import pathlib
# -*- coding: utf-8 -*-
"""E0 两档对照：每一拍 12 块肌肉的发力，看从哪一拍开始分家、差在哪块肌肉。"""
import os, sys, pathlib
import numpy as np
根 = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(根))

import 本能工具_instincts as 本能
import 主循环_完整的一拍 as 主循环
import 身体_go2 as 身体

图高, 图宽 = 主循环.图高, 主循环.图宽
红图 = 主循环.红图("中")
种子 = int(sys.argv[1]) if len(sys.argv) > 1 else 20260915
肌肉名 = list(身体.肌肉名)

def 采一档(不保留):
    本能.建网种子 = 种子
    脑 = 主循环.脑(学=False, 说=False, 前额叶不保留=不保留)
    身 = 身体.身体()
    声 = np.asarray(本能.听觉例子["低频响"](), dtype=np.float64)
    身.摆成站姿(); 身.走(0.5, None)
    脑.亮[:] = False; 脑.上一拍[:] = False
    力s = []; 高s = []; 歪s = []
    for k in range(1, 121):
        脑.一拍(图=红图, 声=声, 身=身)
        力 = np.asarray(脑.肌肉发力(), dtype=float)
        身.步进(力)
        力s.append(力); 高s.append(身.机身高度()); 歪s.append(身.机身歪了())
    return np.array(力s), np.array(高s), np.array(歪s)

A, 高A, 歪A = 采一档(False)
B, 高B, 歪B = 采一档(True)
print("肌肉顺序：%s" % "、".join(肌肉名), flush=True)
print("拍   歪(带环) 歪(无环)  力差最大的一块(差值)   总力(带环) 总力(无环)", flush=True)
for k in range(0, 120, 2):
    d = A[k] - B[k]
    j = int(np.argmax(np.abs(d)))
    print("%3d   %5.0f   %5.0f    %-8s %+.2f          %5.2f     %5.2f"
          % (k + 1, 歪A[k], 歪B[k], 肌肉名[j], d[j], A[k].sum(), B[k].sum()), flush=True)
print("", flush=True)
for k in (10, 30, 60, 90, 120):
    print("第 %d 拍 带环：%s" % (k, " ".join("%s%.2f" % (n, v) for n, v in zip(肌肉名, A[k - 1]))), flush=True)
    print("       无环：%s" % (" ".join("%s%.2f" % (n, v) for n, v in zip(肌肉名, B[k - 1]))), flush=True)