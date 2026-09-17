import pathlib
# -*- coding: utf-8 -*-
"""试验：把"前额叶动态整体抑制"打开到几个档位，跑真正的 R9（含考试段），
   看能不能把"走久了摔"修掉，同时不毁掉学会的能力。"""
import os, sys, pathlib, importlib
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
帧秒 = 主循环.帧秒

def 跑一个种子(种子, k):
    if k is None:
        os.environ.pop("前额叶自压", None)
    else:
        os.environ["前额叶自压"] = str(k)
    本能.建网种子 = 种子
    脑 = 主循环.脑(学=True, 说=False)
    身 = 身体.身体()
    网 = 脑.网
    运记起 = 网.区起点["运动记忆"]
    走块 = 运记起 + 运动记忆.时刻们("前进1.2")
    声 = np.asarray(本能.听觉例子["低频响"](), dtype=np.float64)
    前起, 前宽 = 网.区起点["前额叶"], 网.区宽["前额叶"]

    def 段(拍数, 图, 有声, 多巴胺=False):
        脑.多巴胺亮 = 多巴胺
        起 = 身.数据.qpos[:2].copy()
        路程 = 0.0; 走拍 = 0; 翻在 = None; 前额峰 = 0
        for k2 in range(1, 拍数 + 1):
            脑.一拍(图=图, 声=(声 if 有声 else None), 身=身)
            身.步进(脑.肌肉发力())
            if 脑.亮[走块].sum():
                走拍 += 1
            p = 身.数据.qpos[:2]
            路程 += float(np.hypot(p[0]-起[0], p[1]-起[1]))
            前额峰 = max(前额峰, int(脑.亮[前起:前起+前宽].sum()))
            if 翻在 is None and 身.机身歪了() > 75:
                翻在 = k2
        return dict(路程=路程, 走拍=走拍, 翻在=翻在, 末高=身.机身高度(),
                    歪=身.机身歪了(), 前额峰=前额峰)

    身.摆成趴姿(); 身.走(0.5, None); 脑.亮[:] = False
    出生 = 段(30, 黑图, True)
    身.摆成站姿(); 身.走(0.5, None); 脑.亮[:] = False
    天生 = 段(30, 红图, False)
    身.摆成站姿(); 身.走(0.5, None); 脑.亮[:] = False
    教学 = 段(200, 红图, True, 多巴胺=True)
    身.摆成趴姿(); 身.走(0.5, None); 脑.亮[:] = False
    考试 = 段(200, 黑图, True)
    return 出生, 天生, 教学, 考试

for k in (None, 0.001, 0.0015, 0.002):
    print("=========== 前额叶自压 = %s ===========" % ("关（老行为）" if k is None else k), flush=True)
    for 种子 in (20260915, 20260916, 20260914):
        出生, 天生, 教学, 考试 = 跑一个种子(种子, k)
        f = lambda r: "走路 %3d/200、挪 %.2f 米、%s、末高 %.3f、歪 %3.0f、前额峰 %d" % (
            r["走拍"], r["路程"],
            ("第 %d 拍翻倒" % r["翻在"]) if r["翻在"] else "一次没翻",
            r["末高"], r["歪"], r["前额峰"])
        print("  种子 %d" % 种子, flush=True)
        print("    出生：", f(出生), flush=True)
        print("    天生：", f(天生), flush=True)
        print("    教学：", f(教学), flush=True)
        print("    考试：", f(考试), flush=True)