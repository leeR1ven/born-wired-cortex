# -*- coding: utf-8 -*-
import numpy as np
import 身体_go2 as 身体
import 运动输出区_motor_output as 运动
import 本能工具_instincts as 本能
import 感觉区_共同 as 感觉
import 前额叶区_prefrontal as 前额
import 运动记忆区_motor_memory as 记
import 视觉前处理_visual_preprocess as 视觉

网, 条 = 本能.装(说=False)
段名 = {}
for n, seg in 记.时刻段.items():
    for t in seg:
        段名[t] = n
运起 = 网.区起点["运动"]; 运宽 = 网.区宽["运动"]
记起 = 网.区起点["运动记忆"]; 记宽 = 网.区宽["运动记忆"]
前起 = 网.区起点["前额叶"]; 前宽 = 网.区宽["前额叶"]


def 跑(身, 秒, 视觉激活=None, 用感觉=True, 每=10, 标签=""):
    亮 = np.zeros(网.总数, dtype=bool)
    起 = 身.数据.qpos[:2].copy()
    for k in range(int(秒 / 0.02)):
        if 用感觉:
            for 区名, 区 in 感觉.三个感觉区():
                c = 网.区起点[区名]
                亮[c:c + 区.输出层数量] = 区.状态转激活(区.读身体(身))
        本能.灌视觉(亮, 网, 第4层=视觉激活)
        各路 = {nm: 亮[网.区段(nm)] for nm, _, _, _ in 前额.前额网.输入路}
        前 = np.asarray(前额.前额网.前向传播(各路), dtype=bool)
        亮[前起:前起 + 前宽] = 前
        回 = 前额.返回线.返回(前)
        for nm, _, _, _ in 前额.前额网.输入路:
            s = 回[nm]
            if s.any():
                亮[网.区段(nm)] |= s
        出 = 网.步进(亮)
        发力 = 运动.解码发力(出[运起:运起 + 运动.运动区宽度])
        身.步进(发力)
        亮 = 出
        if k % 每 == 0:
            记忆亮 = np.nonzero(出[记起:记起 + 记宽])[0]
            名 = sorted({段名.get(int(i), "?" + str(int(i))) for i in 记忆亮})
            print("  %s %3d 高%.3f 歪%3.0f 记忆[%s] 力%.2f~%.2f"
                  % (标签, k, 身.机身高度(), 身.机身歪了(), ",".join(名),
                     float(np.min(发力)), float(np.max(发力))))
    return dict(末高=身.机身高度(), 歪=身.机身歪了(),
                前进=float(身.数据.qpos[0] - 起[0]))


print("=== [1] 趴着，什么都不给 ===")
身 = 身体.身体(); 身.摆成趴姿(); 身.走(0.5, None)
print(" 起手高 %.3f" % 身.机身高度())
r = 跑(身, 3.0, 标签="起身"); print(" -> 末高 %.3f 歪 %.0f" % (r["末高"], r["歪"]))

print("=== [2] 站着，眼前一片红 ===")
身 = 身体.身体(); 身.摆成站姿(); 身.走(0.5, None)
图 = np.zeros((1080, 1920, 3), dtype=np.uint8); 图[:, 640:1280, 0] = 255
红 = np.asarray(视觉.网.前向传播(视觉.图像转信号(图)), dtype=bool)
r = 跑(身, 4.0, 视觉激活=红, 标签="走")
print(" -> 前进 %.2f 米 末高 %.3f 歪 %.0f" % (r["前进"], r["末高"], r["歪"]))
