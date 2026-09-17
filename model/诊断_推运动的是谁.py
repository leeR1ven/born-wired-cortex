import pathlib
# -*- coding: utf-8 -*-
"""E0：力量从第几拍分家；那一拍运动区收到的电流里，哪一路多出来了。"""
import os, sys, pathlib
import numpy as np
根 = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(根))

import 本能工具_instincts as 本能
import 主循环_完整的一拍 as 主循环
import 运动输出区_motor_output as 运动
import 身体_go2 as 身体

图高, 图宽 = 主循环.图高, 主循环.图宽
红图 = 主循环.红图("中")
种子 = int(sys.argv[1]) if len(sys.argv) > 1 else 20260915
区们 = ["本体感觉", "平衡感觉", "机身状态", "运动记忆", "视觉运动", "前额叶", "听觉", "视觉", "运动"]

def 采(不保留, 拍数=48):
    本能.建网种子 = 种子
    脑 = 主循环.脑(学=False, 说=False, 前额叶不保留=不保留)
    身 = 身体.身体()
    网 = 脑.网
    声 = np.asarray(本能.听觉例子["低频响"](), dtype=np.float64)
    运起 = 网.区起点["运动"]; 终 = 运起 + 运动.腿区宽度
    号 = [网.区名.index(n) for n in 区们]
    行 = []
    for k in range(1, 拍数 + 1):
        亮 = 脑.上一拍
        s = np.zeros(len(网.区名))
        if 亮.any():
            边, _ = 网._出边切片(np.nonzero(亮)[0])
            if 边.size:
                目 = 网._兴奋目[边]
                m = (目 >= 运起) & (目 < 终)
                if m.any():
                    s = np.bincount(网.区号[网._兴奋源[边][m]],
                                    weights=网._兴奋权[边][m].astype(np.float64),
                                    minlength=len(网.区名))
        脑.一拍(图=红图, 声=声, 身=身)
        力 = np.asarray(脑.肌肉发力(), dtype=float)
        身.步进(力)
        行.append((k, np.array([s[i] for i in 号]), 力.copy()))
    return 行

A = 采(False); B = 采(True)
print("拍 总力(带环) 总力(无环) 差最大的肌肉 差值 | " + " ".join("%8s" % n[:4] for n in 区们), flush=True)
for (k, sa, fa), (_, sb, fb) in zip(A, B):
    if k % 3:
        continue
    d = fa - fb
    j = int(np.argmax(np.abs(d)))
    print("%3d   %5.2f    %5.2f   %-9s %+5.2f | " % (k, fa.sum(), fb.sum(), 身体.肌肉名[j], d[j])
          + " ".join("%+8.2f" % v for v in (sa - sb)), flush=True)