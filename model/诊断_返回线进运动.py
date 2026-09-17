import pathlib
# -*- coding: utf-8 -*-
"""量一量「天生互惠返回线」每一拍往运动区里 OR 进多少个细胞，以及它跟着前额叶涨不涨。"""
import os, sys, pathlib
import numpy as np
根 = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(根))

import 本能工具_instincts as 本能
import 前额叶区_prefrontal as 前额
import 视觉前处理_visual_preprocess as 视觉
import 听觉前处理_auditory_preprocess as 听觉
import 运动输出区_motor_output as 运动
import 主循环_完整的一拍 as 主循环
import 身体_go2 as 身体

图高, 图宽 = 主循环.图高, 主循环.图宽
红图 = 主循环.红图("中")
种子 = int(sys.argv[1]) if len(sys.argv) > 1 else 20260915
视宽 = 视觉.网.输出层数量
听宽 = 听觉.网.输出层数量
记录 = {'回运动': 0, '回全部': 0}
原返回 = type(前额.返回线).返回

def 包装(self, 输出激活, 抑制量=0.0):
    r = 原返回(self, 输出激活, 抑制量)
    记录['回全部'] = int(r.sum())
    记录['回运动'] = int(r[视宽 + 听宽:].sum())
    return r
type(前额.返回线).返回 = 包装

def 跑(名, 不保留, 自压=None):
    for k in ("前额叶不保留", "前额叶自压"):
        os.environ.pop(k, None)
    if 自压:
        os.environ["前额叶自压"] = 自压
    本能.建网种子 = 种子
    脑 = 主循环.脑(学=False, 说=False, 前额叶不保留=不保留)
    身 = 身体.身体()
    网 = 脑.网
    声 = np.asarray(本能.听觉例子["低频响"](), dtype=np.float64)
    前起, 前宽 = 网.区起点["前额叶"], 网.区宽["前额叶"]
    运起, 运宽 = 网.区起点["运动"], 网.区宽["运动"]
    运记起 = 网.区起点["运动记忆"]
    走块 = 运记起 + 运动记忆.时刻们("前进1.2")
    身.摆成站姿(); 身.走(0.5, None)
    脑.亮[:] = False; 脑.上一拍[:] = False
    print("### %s（种子 %d）" % (名, 种子), flush=True)
    print("   拍    高    歪  前额亮  回全部 回运动  运动亮  走链", flush=True)
    翻在 = None
    for k in range(1, 201):
        脑.一拍(图=红图, 声=声, 身=身)
        身.步进(脑.肌肉发力())
        歪 = 身.机身歪了()
        if 翻在 is None and 歪 > 75:
            翻在 = k
        if k % 10 == 0:
            print("  %3d  %5.3f %5.0f  %6d  %6d %6d  %6d  %4d"
                  % (k, 身.机身高度(), 歪, int(脑.亮[前起:前起 + 前宽].sum()),
                     记录['回全部'], 记录['回运动'], int(脑.亮[运起:运起 + 运宽].sum()),
                     int(脑.亮[走块].sum())), flush=True)
    print("  -> %s\n" % (("第 %d 拍翻" % 翻在) if 翻在 else "没翻"), flush=True)

import 运动记忆区_motor_memory as 运动记忆
跑("① 带环（会摔）", False)
跑("② 环关掉（不摔）", True)
跑("③ 带环 + 自压 0.002", False, "0.002")