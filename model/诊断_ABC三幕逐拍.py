import pathlib
# -*- coding: utf-8 -*-
"""诊断：A/B/C 三幕逐拍对比 —— 站起来链、走路链、运动区、力矩。"""
import os, sys, pathlib
import numpy as np
根 = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(根))
os.environ.setdefault("前额叶不保留", "1")

import 本能工具_instincts as 本能
import 主循环_完整的一拍 as 主循环
import 运动记忆区_motor_memory as 运动记忆
import 身体_go2 as 身体

本能.建网种子 = 20260914
脑 = 主循环.脑(学=True, 说=False)
网 = 脑.网
身 = 身体.身体()
运记起 = 网.区起点["运动记忆"]
站块 = 运记起 + np.asarray(运动记忆.时刻们("站起来"), dtype=np.int64)
走块 = 运记起 + np.asarray(运动记忆.时刻们("前进1.2"), dtype=np.int64)
前一块 = 脑.段["前额叶"]
黑图 = np.zeros((主循环.图高, 主循环.图宽, 3), dtype=np.uint8)
红图 = 主循环.红图("中")
帧秒 = 主循环.帧秒
print("站起来块 %d 个细胞，走路块 %d 个细胞" % (len(站块), len(走块)), flush=True)

for 名, 起点, 图 in (("A 趴着无图", "趴", None), ("B 站着看红", "站", 红图), ("C 站着黑屏", "站", 黑图)):
    身.摆成趴姿() if 起点 == "趴" else 身.摆成站姿()
    身.走(0.5, None)
    脑.亮[:] = False; 脑.上一拍[:] = False
    print("=== %s ===" % 名, flush=True)
    for k in range(int(3.0/帧秒)):
        出 = 脑.一拍(图=图, 声=None, 身=身)
        力 = 脑.肌肉发力()
        身.步进(力)
        if k % 4 == 0:
            亮 = lambda a: int(np.count_nonzero(出[网.区起点[a]:网.区起点[a]+网.区宽[a]]))
            print("  %3d %.2f 末高%.3f 歪%3.0f | 站起来链%3d 走路链%3d | 前额%4d 视觉%5d 听觉%5d 运动%3d 运记%4d | 力%s"
                  % (k, k*帧秒, 身.机身高度(), 身.机身歪了(),
                     int(np.count_nonzero(出[站块])), int(np.count_nonzero(出[走块])),
                     int(np.count_nonzero(出[前一块])), 亮("视觉"), 亮("听觉"), 亮("运动"), 亮("运动记忆"),
                     np.round(np.asarray(力)[:3], 2)), flush=True)