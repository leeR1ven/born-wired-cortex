import pathlib
# -*- coding: utf-8 -*-
"""E0（站着+红+声音，不学）：前额叶那股一直在涨的活动，电流都打到哪个区去了。"""
import os, sys, pathlib
import numpy as np
根 = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(根))

import 本能工具_instincts as 本能
import 主循环_完整的一拍 as 主循环
import 运动记忆区_motor_memory as 运动记忆
import 身体_go2 as 身体

图高, 图宽 = 主循环.图高, 主循环.图宽
红图 = 主循环.红图("中")
种子 = int(sys.argv[1]) if len(sys.argv) > 1 else 20260915

def 跑(名, 不保留):
    本能.建网种子 = 种子
    脑 = 主循环.脑(学=False, 说=False, 前额叶不保留=不保留)
    身 = 身体.身体()
    网 = 脑.网
    声 = np.asarray(本能.听觉例子["低频响"](), dtype=np.float64)
    前起, 前宽 = 网.区起点["前额叶"], 网.区宽["前额叶"]
    运记起 = 网.区起点["运动记忆"]
    走块 = 运记起 + 运动记忆.时刻们("前进1.2")
    身.摆成站姿(); 身.走(0.5, None)
    脑.亮[:] = False; 脑.上一拍[:] = False

    print("### %s（种子 %d）" % (名, 种子), flush=True)
    print("  拍  高     歪  前额亮   " + " ".join("%6s" % n[:6] for n in ["视觉", "听觉", "运动", "运动记忆", "本体", "平衡", "机身", "视运"]), flush=True)
    上一 = 身.数据.qpos[:2].copy()
    for k in range(1, 201):
        脑.一拍(图=红图, 声=声, 身=身)
        身.步进(脑.肌肉发力())
        上一 = 身.数据.qpos[:2].copy()
        if k % 10 == 0:
            亮 = 脑.上一拍
            前亮 = np.nonzero(亮[前起:前起 + 前宽])[0]
            if 前亮.size:
                边, _ = 网._出边切片(前亮 + 前起)
                目 = 网._兴奋目[边]
                权 = 网._兴奋权[边].astype(np.float64)
                号 = 网.区号[目]
                s = np.bincount(号, weights=权, minlength=len(网.区名))
            else:
                s = np.zeros(len(网.区名))
            要 = {"视觉": "视觉", "听觉": "听觉", "运动": "运动", "运动记忆": "运动记忆",
                 "本体感觉": "本体感觉", "平衡感觉": "平衡感觉", "机身状态": "机身状态", "视觉运动": "视觉运动"}
            print("  %3d %5.3f %4.0f %6d   " % (k, 身.机身高度(), 身.机身歪了(), 前亮.size)
                  + " ".join("%6.1f" % s[网.区名.index(v)] for v in 要.values()) +
                  "   走链%d" % int(脑.亮[走块].sum()), flush=True)

跑("带环（会摔的那一档）", False)
跑("环关掉（不摔的那一档）", True)