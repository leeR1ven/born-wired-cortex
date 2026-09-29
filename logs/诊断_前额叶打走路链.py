import pathlib
# -*- coding: utf-8 -*-
"""量：前额叶每一拍往"走路链"那 15 个神经元上打多少电流。
再试：把"前额叶 → 运动记忆"这条天生连接的强度打个折，摔不摔。"""
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

def 跑(名, 折, 打电流=False, 拍数=200):
    本能.建网种子 = 种子
    脑 = 主循环.脑(学=False, 说=False)
    身 = 身体.身体()
    网 = 脑.网
    声 = np.asarray(本能.听觉例子["低频响"](), dtype=np.float64)
    前起, 前宽 = 网.区起点["前额叶"], 网.区宽["前额叶"]
    运记起, 运记宽 = 网.区起点["运动记忆"], 网.区宽["运动记忆"]
    链 = 运记起 + 运动记忆.时刻们("前进1.2")
    if 折 != 1.0:
        m = ((网._兴奋源 >= 前起) & (网._兴奋源 < 前起 + 前宽)
             & (网._兴奋目 >= 运记起) & (网._兴奋目 < 运记起 + 运记宽))
        网._兴奋权[m] = (网._兴奋权[m].astype(np.float64) * 折).astype(np.float32)
        print("  [%s] 前额叶→运动记忆 的边 %d 根，强度 x%.2f" % (名, int(m.sum()), 折), flush=True)
    身.摆成站姿(); 身.走(0.5, None)
    脑.亮[:] = False; 脑.上一拍[:] = False
    上一 = 身.数据.qpos[:2].copy(); 路程 = 0.0; 翻在 = None
    for k in range(1, 拍数 + 1):
        if 打电流:
            亮 = 脑.上一拍
            v = 0.0
            if 亮[前起:前起 + 前宽].any():
                边, _ = 网._出边切片(np.nonzero(亮[前起:前起 + 前宽])[0] + 前起)
                if 边.size:
                    目 = 网._兴奋目[边]
                    q = (目 >= 链[0]) if False else np.isin(目, 链)
                    if q.any():
                        v = float(网._兴奋权[边][q].astype(np.float64).sum())
        else:
            v = 0.0
        脑.一拍(图=红图, 声=声, 身=身)
        if 打电流 and k % 20 == 0:
            print("     第 %3d 拍：前额叶打到走路链上的电流 = %.3f（走路链亮 %d 个）"
                  % (k, v, int(脑.亮[链].sum())), flush=True)
        身.步进(脑.肌肉发力())
        p = 身.数据.qpos[:2]
        路程 += float(np.hypot(p[0] - 上一[0], p[1] - 上一[1])); 上一 = p.copy()
        歪 = 身.机身歪了()
        if 翻在 is None and 歪 > 75:
            翻在 = k
    print("  %-30s 挪 %5.2f 米  %-14s 末高 %.3f 末歪 %3.0f"
          % (名, 路程, ("第 %d 拍翻" % 翻在) if 翻在 else "一次没翻", 身.机身高度(), 身.机身歪了()), flush=True)
    return 翻在

print("### 一、先看前额叶往走路链上打多少电流（种子 %d）" % 种子, flush=True)
跑("原样", 1.0, 打电流=True, 拍数=120)
print("\n### 二、把 前额叶→运动记忆 打折，看摔不摔（200 拍）", flush=True)
for 折 in (1.0, 0.5, 0.25, 0.0):
    跑("折 %s" % 折, 折)