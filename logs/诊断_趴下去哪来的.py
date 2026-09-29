# -*- coding: utf-8 -*-
import numpy as np
import 身体_go2 as 身体
import 运动输出区_motor_output as 运动
import 本能工具_instincts as 本能
import 感觉区_共同 as 感觉
import 前额叶区_prefrontal as 前额
import 运动记忆区_motor_memory as 记

网, 条 = 本能.装(说=False)
段名 = {}
for n, seg in 记.时刻段.items():
    for t in seg:
        段名[t] = n
起头 = {n: seg[0] for n, seg in 记.时刻段.items()}
趴头 = 起头["趴下去"]
趴下头 = 起头["坐下"]
print("趴下去 头一拍在本区下标 =", 趴头, "；坐下 =", 趴下头)
运起 = 网.区起点["运动"]; 运宽 = 网.区宽["运动"]
记起 = 网.区起点["运动记忆"]; 记宽 = 网.区宽["运动记忆"]
前起 = 网.区起点["前额叶"]; 前宽 = 网.区宽["前额叶"]

身 = 身体.身体(); 身.摆成趴姿(); 身.走(0.5, None)
亮 = np.zeros(网.总数, dtype=bool)
for k in range(120):
    for 区名, 区 in 感觉.三个感觉区():
        c = 网.区起点[区名]
        亮[c:c + 区.输出层数量] = 区.状态转激活(区.读身体(身))
    各路 = {nm: 亮[网.区段(nm)] for nm, _, _, _ in 前额.前额网.输入路}
    前 = np.asarray(前额.前额网.前向传播(各路), dtype=bool)
    亮[前起:前起 + 前宽] = 前
    回 = 前额.返回线.返回(前)
    回记 = 回["运动记忆"]
    回记名 = sorted({段名.get(int(i), "?" + str(int(i))) for i in np.nonzero(回记)[0]})
    前入 = 各路["运动记忆"]           # 前额叶这一拍从运动记忆收到什么（已经扩散过）
    入名 = sorted({段名.get(int(i), "?" + str(int(i))) for i in np.nonzero(前入)[0]})
    for nm, _, _, _ in 前额.前额网.输入路:
        s = 回[nm]
        if s.any():
            亮[网.区段(nm)] |= s
    出 = 网.步进(亮)
    发力 = 运动.解码发力(出[运起:运起 + 运动.运动区宽度])
    身.步进(发力)
    亮 = 出
    记忆亮 = np.nonzero(出[记起:记起 + 记宽])[0]
    名 = sorted({段名.get(int(i), "?" + str(int(i))) for i in 记忆亮})
    趴亮 = 趴头 in set((记忆亮).tolist())
    if 50 <= k <= 70 or k % 20 == 0:
        print("%3d 高%.3f | 前额叶亮%4d | 回运动记忆[%s] | 前额叶收到运动记忆[%s] | 运动记忆亮[%s] 趴下去头=%s"
              % (k, 身.机身高度(), int(前.sum()), ",".join(回记名)[:40], ",".join(入名)[:40],
                 ",".join(名)[:40], 趴亮))
