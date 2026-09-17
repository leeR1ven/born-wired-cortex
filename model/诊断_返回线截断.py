import pathlib
# -*- coding: utf-8 -*-
"""E0 对照：把天生互惠返回线 ①不动 ②只掐掉"送回运动区"那一段 ③整条掐掉 ④环也关掉。
看哪一档不摔。"""
import os, sys, pathlib
import numpy as np
根 = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(根))

import 本能工具_instincts as 本能
import 前额叶区_prefrontal as 前额
import 视觉前处理_visual_preprocess as 视觉
import 听觉前处理_auditory_preprocess as 听觉
import 主循环_完整的一拍 as 主循环
import 身体_go2 as 身体

图高, 图宽 = 主循环.图高, 主循环.图宽
红图 = 主循环.红图("中")
种子 = int(sys.argv[1]) if len(sys.argv) > 1 else 20260915
视宽 = 视觉.网.输出层数量
听宽 = 听觉.网.输出层数量
挡 = [0]          # 0=不挡 1=只挡运动段 2=全挡
原返回 = type(前额.返回线).返回

def 包装(self, 输出激活, 抑制量=0.0):
    r = 原返回(self, 输出激活, 抑制量)
    if 挡[0] == 2:
        return np.zeros_like(r)
    if 挡[0] == 1:
        r = r.copy()
        r[视宽 + 听宽:] = False
    return r
type(前额.返回线).返回 = 包装

def 跑(名, 不保留, 挡法):
    挡[0] = 挡法
    本能.建网种子 = 种子
    脑 = 主循环.脑(学=False, 说=False, 前额叶不保留=不保留)
    身 = 身体.身体()
    声 = np.asarray(本能.听觉例子["低频响"](), dtype=np.float64)
    身.摆成站姿(); 身.走(0.5, None)
    脑.亮[:] = False; 脑.上一拍[:] = False
    上一 = 身.数据.qpos[:2].copy(); 路程 = 0.0; 翻在 = None
    for k in range(1, 201):
        脑.一拍(图=红图, 声=声, 身=身)
        身.步进(脑.肌肉发力())
        p = 身.数据.qpos[:2]
        路程 += float(np.hypot(p[0] - 上一[0], p[1] - 上一[1])); 上一 = p.copy()
        歪 = 身.机身歪了()
        if 翻在 is None and 歪 > 75:
            翻在 = k
    print("  %-26s 挪 %5.2f 米  %-14s 末高 %.3f 末歪 %3.0f"
          % (名, 路程, ("第 %d 拍翻" % 翻在) if 翻在 else "一次没翻",
             身.机身高度(), 身.机身歪了()), flush=True)

print("### E0 对照（种子 %d）" % 种子, flush=True)
跑("① 原样（带环）", False, 0)
跑("② 只掐掉返回线送运动区", False, 1)
跑("③ 整条返回线掐掉", False, 2)
跑("④ 环关掉（参考）", True, 0)