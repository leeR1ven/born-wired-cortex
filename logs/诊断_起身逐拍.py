# -*- coding: utf-8 -*-
import numpy as np
import 身体_go2 as 身体
import 运动输出区_motor_output as 运动
import 本能工具_instincts as 本能
import 感觉区_共同 as 感觉
import 前额叶区_prefrontal as 前额
import 运动记忆区_motor_memory as 记
import 平衡感觉区_balance as 平衡
import 本体感觉区_proprioception as 本体

网, 条 = 本能.装(说=False)
身 = 身体.身体()
身.摆成趴姿(); 身.走(0.5, None)
运起 = 网.区起点["运动"]; 运宽 = 网.区宽["运动"]
记起 = 网.区起点["运动记忆"]; 记宽 = 网.区宽["运动记忆"]
段名 = {}
for n, seg in 记.时刻段.items():
    for t in seg:
        段名[t] = n
亮 = np.zeros(网.总数, dtype=bool)
for k in range(260):
    for 区名, 区 in 感觉.三个感觉区():
        c = 网.区起点[区名]
        亮[c:c + 区.输出层数量] = 区.状态转激活(区.读身体(身))
    各路 = {nm: 亮[网.区段(nm)] for nm, _, _, _ in 前额.前额网.输入路}
    前 = np.asarray(前额.前额网.前向传播(各路), dtype=bool)
    前起 = 网.区起点["前额叶"]
    亮[前起:前起 + 网.区宽["前额叶"]] = 前
    回 = 前额.返回线.返回(前)
    for nm, _, _, _ in 前额.前额网.输入路:
        s = 回[nm]
        if s.any():
            亮[网.区段(nm)] |= s
    出 = 网.步进(亮)
    发力 = 运动.解码发力(出[运起:运起 + 运动.运动区宽度])
    身.步进(发力)
    亮 = 出
    if k % 10 == 0 or 55 <= k <= 90:
        记忆亮 = np.nonzero(出[记起:记起 + 记宽])[0]
        所亮 = sorted({段名.get(int(i), "?" + str(i)) for i in 记忆亮})
        运亮 = int(出[运起:运起 + 运宽].sum())
        print("%3d 高%.3f 歪%3.0f 记忆亮%d %s | 运动亮%d 力均值%.2f 最大%.2f"
              % (k, 身.机身高度(), 身.机身歪了(), len(记忆亮),
                 "/".join(所亮[:3]), 运亮, float(np.mean(np.abs(发力))), float(np.max(np.abs(发力)))))
