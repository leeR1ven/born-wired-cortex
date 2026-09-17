import pathlib
# -*- coding: utf-8 -*-
"""试验：新机制「用不上的连接会被削弱」对 R9 有什么用。"""
import os, sys, pathlib
import numpy as np
根 = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(根))
os.environ.pop("前额叶不保留", None)
os.environ.pop("前额叶自压", None)

import 本能工具_instincts as 本能
import 主循环_完整的一拍 as 主循环
import 运动记忆区_motor_memory as 运动记忆
import 身体_go2 as 身体

图高, 图宽 = 主循环.图高, 主循环.图宽
黑图 = np.zeros((图高, 图宽, 3), dtype=np.uint8)
红图 = 主循环.红图("中")

def 跑一个种子(种子, 削):
    if 削 is None:
        os.environ.pop("用不上衰减率", None)
    else:
        os.environ["用不上衰减率"] = str(削)
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
        w0 = 网._兴奋权.astype(np.float64).copy()
        路程 = 0.0; 走拍 = 0; 翻在 = None
        for k in range(1, 拍数 + 1):
            脑.一拍(图=图, 声=(声 if 有声 else None), 身=身)
            身.步进(脑.肌肉发力())
            if 脑.亮[走块].sum():
                走拍 += 1
            p = 身.数据.qpos[:2]
            路程 += float(np.hypot(p[0]-起[0], p[1]-起[1]))
            if 翻在 is None and 身.机身歪了() > 75:
                翻在 = k
        非固 = ~网._兴奋固化
        差 = 网._兴奋权.astype(np.float64) - w0
        差[~非固] = 0.0
        return dict(拍数=拍数, 走拍=走拍, 路程=路程, 翻在=翻在, 末高=身.机身高度(),
                    歪=身.机身歪了(), 总增=float(差.sum()), 最大增=float(差.max()),
                    最大减=float(差.min()))

    身.摆成趴姿(); 身.走(0.5, None); 脑.亮[:] = False
    出生 = 段(30, 黑图, True)
    身.摆成站姿(); 身.走(0.5, None); 脑.亮[:] = False
    天生 = 段(30, 红图, False)
    身.摆成站姿(); 身.走(0.5, None); 脑.亮[:] = False
    教学 = 段(200, 红图, True, 多巴胺=True)
    身.摆成趴姿(); 身.走(0.5, None); 脑.亮[:] = False
    考试 = 段(200, 黑图, True)
    return 出生, 天生, 教学, 考试

for 削 in (0.02, 0.05, 0.1):
    print("=========== 用不上衰减率 = %s ===========" % 削, flush=True)
    for 种子 in (20260915, 20260916):
        出生, 天生, 教学, 考试 = 跑一个种子(种子, 削)
        f = lambda r: "走路 %3d/%d、挪 %6.2f 米、%s、末高 %.3f、歪 %3.0f | 权重增 %+.0f 最大增 %+.3f 最大减 %.3f" % (
            r["走拍"], r["拍数"], r["路程"],
            ("第 %d 拍翻倒" % r["翻在"]) if r["翻在"] else "一次没翻",
            r["末高"], r["歪"], r["总增"], r["最大增"], r["最大减"])
        print("  种子 %d" % 种子, flush=True)
        for 名, r in (("出生", 出生), ("天生", 天生), ("教学", 教学), ("考试", 考试)):
            print("    %s：" % 名, f(r), flush=True)