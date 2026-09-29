import pathlib
# -*- coding: utf-8 -*-
"""把「前额叶 → 运动记忆」这条天生规则打个折，跑真正的 R9 四段，看还学不学得会、还摔不摔。"""
import os, sys, pathlib
import numpy as np
根 = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(根))
os.environ.pop("前额叶不保留", None)

import 本能工具_instincts as 本能
import 主循环_完整的一拍 as 主循环
import 运动记忆区_motor_memory as 运动记忆
import 身体_go2 as 身体

图高, 图宽 = 主循环.图高, 主循环.图宽
黑图 = np.zeros((图高, 图宽, 3), dtype=np.uint8)
红图 = 主循环.红图("中")

def 打折(网, 折):
    if 折 == 1.0:
        return 0
    前起, 前宽 = 网.区起点["前额叶"], 网.区宽["前额叶"]
    记起, 记宽 = 网.区起点["运动记忆"], 网.区宽["运动记忆"]
    m = ((网._兴奋源 >= 前起) & (网._兴奋源 < 前起 + 前宽)
         & (网._兴奋目 >= 记起) & (网._兴奋目 < 记起 + 记宽))
    网._兴奋权[m] = (网._兴奋权[m].astype(np.float64) * 折).astype(np.float32)
    return int(m.sum())

def 跑一个种子(种子, 折):
    本能.建网种子 = 种子
    脑 = 主循环.脑(学=True, 说=False)
    身 = 身体.身体()
    网 = 脑.网
    n = 打折(网, 折)
    运记起 = 网.区起点["运动记忆"]
    走块 = 运记起 + 运动记忆.时刻们("前进1.2")
    声 = np.asarray(本能.听觉例子["低频响"](), dtype=np.float64)
    前起, 前宽 = 网.区起点["前额叶"], 网.区宽["前额叶"]

    def 段(拍数, 图, 有声, 多巴胺=False):
        脑.多巴胺亮 = 多巴胺
        起 = 身.数据.qpos[:2].copy(); 路程 = 0.0; 走拍 = 0; 翻在 = None
        for k in range(1, 拍数 + 1):
            脑.一拍(图=图, 声=(声 if 有声 else None), 身=身)
            身.步进(脑.肌肉发力())
            if 脑.亮[走块].any():
                走拍 += 1
            p = 身.数据.qpos[:2]
            路程 += float(np.hypot(p[0] - 起[0], p[1] - 起[1]))
            if 翻在 is None and 身.机身歪了() > 75:
                翻在 = k
        return dict(拍数=拍数, 走拍=走拍, 路程=路程, 翻在=翻在, 末高=身.机身高度(),
                    歪=身.机身歪了(), 前额=int(脑.亮[前起:前起 + 前宽].sum()))

    身.摆成趴姿(); 身.走(0.5, None); 脑.亮[:] = False; 脑.上一拍[:] = False
    出生 = 段(30, 黑图, True)
    身.摆成站姿(); 身.走(0.5, None); 脑.亮[:] = False; 脑.上一拍[:] = False
    天生 = 段(30, 红图, False)
    身.摆成站姿(); 身.走(0.5, None); 脑.亮[:] = False; 脑.上一拍[:] = False
    教学 = 段(200, 红图, True, 多巴胺=True)
    身.摆成趴姿(); 身.走(0.5, None); 脑.亮[:] = False; 脑.上一拍[:] = False
    考试 = 段(200, 黑图, True)
    return 出生, 天生, 教学, 考试, n

for 折 in (1.0, 0.5):
    print("============ 前额叶→运动记忆 折 = %s ============" % 折, flush=True)
    for 种子 in (20260915, 20260918, 20260914, 20260916, 20260917):
        出生, 天生, 教学, 考试, n = 跑一个种子(种子, 折)
        f = lambda r: "走链 %3d/%d、挪 %6.3f 米、%s、末高 %.3f、歪 %4.0f、前额 %d" % (
            r["走拍"], r["拍数"], r["路程"],
            ("第 %d 拍翻倒" % r["翻在"]) if r["翻在"] else "一次没翻",
            r["末高"], r["歪"], r["前额"])
        print("  种子 %d（打折边 %d 根）" % (种子, n), flush=True)
        print("    出生：", f(出生), flush=True)
        print("    天生：", f(天生), flush=True)
        print("    教学：", f(教学), flush=True)
        print("    考试：", f(考试), flush=True)