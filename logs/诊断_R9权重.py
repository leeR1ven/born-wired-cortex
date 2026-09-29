import pathlib
# -*- coding: utf-8 -*-
"""诊断 R9：考试段到底什么在长、摔是谁干的。"""
import os, sys, pathlib
import numpy as np
根 = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(根))
os.environ.pop("前额叶不保留", None)          # R9 是"带环"那一档

import 本能工具_instincts as 本能
import 主循环_完整的一拍 as 主循环
import 运动记忆区_motor_memory as 运动记忆
import 身体_go2 as 身体

图高, 图宽 = 主循环.图高, 主循环.图宽
黑图 = np.zeros((图高, 图宽, 3), dtype=np.uint8)
红图 = 主循环.红图("中")
种子 = int(sys.argv[1]) if len(sys.argv) > 1 else 20260915

本能.建网种子 = 种子
名字 = 本能.读名字()
脑 = 主循环.脑(学=True, 说=False)
身 = 身体.身体()
网 = 脑.网
运记起 = 网.区起点["运动记忆"]
走块 = 运记起 + 运动记忆.时刻们("前进1.2")
声 = np.asarray(本能.听觉例子["低频响"](), dtype=np.float64)
前起, 前宽 = 网.区起点["前额叶"], 网.区宽["前额叶"]

# 细胞 -> 区名
区名 = {}
for k, 起 in 网.区起点.items():
    区名[k] = (起, 起 + 网.区宽[k])
def 谁(i):
    for k, (a, b) in 区名.items():
        if a <= i < b:
            return k
    return "?"

出生非固条数 = int((~网._兴奋固化).sum())
print("出生：非固化兴奋连接 %d 条（总共 %d 条）" % (出生非固条数, 网._兴奋源.size), flush=True)

def 权重摘要():
    非固 = ~网._兴奋固化
    w = 网._兴奋权[非固]
    return (int(非固.sum()), float(w.sum()), float(w.max()) if w.size else 0.0,
            int((w > 0.02).sum()), int((w > 0.1).sum()))

def 最重的几根(k=5):
    非固 = np.nonzero(~网._兴奋固化)[0]
    if 非固.size == 0:
        return []
    w = 网._兴奋权[非固]
    o = np.argsort(w)[::-1][:k]
    return [(谁(int(网._兴奋源[非固[i]])), 谁(int(网._兴奋目[非固[i]])), float(w[i]))
            for i in o]

def 段(名, 拍数, 图, 有声, 多巴胺=False):
    脑.多巴胺亮 = 多巴胺
    print("=== %s ===" % 名, flush=True)
    for k in range(1, 拍数 + 1):
        脑.一拍(图=图, 声=(声 if 有声 else None), 身=身)
        身.步进(脑.肌肉发力())
        if k % 20 == 0 or (名 == "考试" and k == 拍数):
            n, 总, 最大, 过2, 过10 = 权重摘要()
            print("  %s 拍%3d 末高%.3f 歪%3.0f | 走链%d 前额%5d | 可塑%d条 总重%.1f 最大%.3f >0.02:%d >0.1:%d"
                  % (名, k, 身.机身高度(), 身.机身歪了(),
                     int(脑.亮[走块].sum()), int(脑.亮[前起:前起+前宽].sum()),
                     n, 总, 最大, 过2, 过10), flush=True)
            for a, b, w in 最重的几根(3):
                print("       最重：%s → %s  %.3f" % (a, b, w), flush=True)

身.摆成趴姿(); 身.走(0.5, None); 脑.亮[:] = False
段("出生", 30, 黑图, True)
身.摆成站姿(); 身.走(0.5, None); 脑.亮[:] = False
段("天生", 30, 红图, False)
身.摆成站姿(); 身.走(0.5, None); 脑.亮[:] = False
段("教学", 200, 红图, True, 多巴胺=True)
身.摆成趴姿(); 身.走(0.5, None); 脑.亮[:] = False
段("考试", 200, 黑图, True)