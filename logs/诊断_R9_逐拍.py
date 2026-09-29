import pathlib
# -*- coding: utf-8 -*-
"""诊断：R9 考试段是"突然摔"还是"慢慢歪"。逐拍记大脑各区亮几个、走路链走到第几拍。"""
import os, sys, pathlib
import numpy as np
根 = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(根))
os.environ.pop("前额叶不保留", None)
os.environ.pop("前额叶自压", None)
os.environ.pop("用不上衰减率", None)

import 本能工具_instincts as 本能
import 主循环_完整的一拍 as 主循环
import 运动记忆区_motor_memory as 运动记忆
import 身体_go2 as 身体

图高, 图宽 = 主循环.图高, 主循环.图宽
黑图 = np.zeros((图高, 图宽, 3), dtype=np.uint8)
红图 = 主循环.红图("中")

种子 = int(sys.argv[1]) if len(sys.argv) > 1 else 20260915
本能.建网种子 = 种子
脑 = 主循环.脑(学=True, 说=False)
身 = 身体.身体()
网 = 脑.网
运记起 = 网.区起点["运动记忆"]
走块 = 运记起 + 运动记忆.时刻们("前进1.2")
声 = np.asarray(本能.听觉例子["低频响"](), dtype=np.float64)
区名们 = ["视觉", "听觉", "前额叶", "运动", "运动记忆", "本体感觉", "平衡感觉", "机身状态", "视觉运动"]

def 区亮(名):
    i, w = 网.区起点[名], 网.区宽[名]
    return int(脑.亮[i:i + w].sum())

def 段(名, 拍数, 图, 有声, 多巴胺=False, 逐拍=False):
    脑.多巴胺亮 = 多巴胺
    上一 = 身.数据.qpos[:2].copy()
    路程 = 0.0
    翻在 = None
    行 = []
    for k in range(1, 拍数 + 1):
        脑.一拍(图=图, 声=(声 if 有声 else None), 身=身)
        身.步进(脑.肌肉发力())
        p = 身.数据.qpos[:2]
        路程 += float(np.hypot(p[0] - 上一[0], p[1] - 上一[1]))
        上一 = p.copy()
        歪 = 身.机身歪了()
        if 翻在 is None and 歪 > 75:
            翻在 = k
        拍链 = int(脑.亮[走块].sum())
        首位 = int(np.nonzero(脑.亮[走块])[0][0]) if 拍链 else -1
        if 逐拍:
            行.append((k, 身.机身高度(), 歪, 路程, 拍链, 首位,
                        float(np.abs(脑.肌肉发力()).sum()),
                        [区亮(n) for n in 区名们]))
    return 行, dict(路程=路程, 翻在=翻在, 歪=身.机身歪了(), 高=身.机身高度())

身.摆成趴姿(); 身.走(0.5, None); 脑.亮[:] = False
段("出生", 30, 黑图, True)
身.摆成站姿(); 身.走(0.5, None); 脑.亮[:] = False
段("天生", 30, 红图, False)
身.摆成站姿(); 身.走(0.5, None); 脑.亮[:] = False
段("教学", 200, 红图, True, 多巴胺=True)
身.摆成趴姿(); 身.走(0.5, None); 脑.亮[:] = False
行, 汇总 = 段("考试", 200, 黑图, True, 逐拍=True)

表头 = "拍  高     歪   路程  走链 首位  力      " + " ".join("%s" % n for n in 区名们)
print("### 种子 %d 考试段逐拍（%s）" % (种子, 表头), flush=True)
for r in 行:
    if r[0] % 2 == 0 or r[0] < 40:
        print("%3d %5.3f %5.0f %5.2f %3d %4d %6.1f  " % (r[0], r[1], r[2], r[3], r[4], r[5], r[6])
              + " ".join("%6d" % v for v in r[7]), flush=True)
print("", flush=True)
print("考试段：挪 %.2f 米、%s、末高 %.3f、末歪 %.0f" % (
    汇总["路程"], ("第 %d 拍翻倒" % 汇总["翻在"]) if 汇总["翻在"] else "一次没翻", 汇总["高"], 汇总["歪"]), flush=True)