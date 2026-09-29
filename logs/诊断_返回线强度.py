# -*- coding: utf-8 -*-
import json, io
import numpy as np
import 身体_go2 as 身体
import 运动输出区_motor_output as 运动
import 本能工具_instincts as 本能
import 感觉区_共同 as 感觉
import 前额叶区_prefrontal as 前额
import 运动记忆区_motor_memory as 记
import 视觉前处理_visual_preprocess as 视觉

网, 条 = 本能.装(说=False)
名字 = 本能.读名字()
段名 = {}
for n, seg in 记.时刻段.items():
    for t in seg:
        段名[t] = n
记起 = 网.区起点["运动记忆"]; 记宽 = 网.区宽["运动记忆"]
前起 = 网.区起点["前额叶"]; 前宽 = 网.区宽["前额叶"]


def 报(标签, 前):
    前 = np.asarray(前, dtype=bool)
    回 = 前额.返回线.返回(前)
    峰 = 前额.返回线.上次峰值
    记回 = 回["运动记忆"]
    序号 = np.nonzero(记回)[0]
    名 = sorted({段名.get(int(i), "?" + str(int(i))) for i in 序号})
    print("%-14s 前额叶亮%5d | 回运动记忆 %2d 个 %s | 峰值 %s"
          % (标签, int(前.sum()), len(序号), ",".join(名)[:60],
             " ".join("%s%.3f" % (k, v) for k, v in 峰.items())))


print("=== 甲：前额叶整片不亮 ===")
报("全灭", np.zeros(前宽, dtype=bool))

print("=== 乙：前额叶里「在想走路」那 12 个细胞亮着 ===")
亮 = np.zeros(前宽, dtype=bool)
亮[np.array(名字["前额叶"]["在想走路"], dtype=int)] = True
报("在想走路", 亮)

print("=== 丙：前额叶里「看着红」整块亮着（在想红）===")
亮 = np.zeros(前宽, dtype=bool)
亮[np.array(名字["前额叶"]["看着红"], dtype=int)] = True
报("看着红", 亮)

print("=== 丁：趴着自己站起来，跑到第 70 拍时前额叶的真实状态 ===")
身 = 身体.身体(); 身.摆成趴姿(); 身.走(0.5, None)
亮 = np.zeros(网.总数, dtype=bool)
for k in range(70):
    for 区名, 区 in 感觉.三个感觉区():
        c = 网.区起点[区名]
        亮[c:c + 区.输出层数量] = 区.状态转激活(区.读身体(身))
    各路 = {nm: 亮[网.区段(nm)] for nm, _, _, _ in 前额.前额网.输入路}
    前 = np.asarray(前额.前额网.前向传播(各路), dtype=bool)
    亮[前起:前起 + 前宽] = 前
    回 = 前额.返回线.返回(前)
    for nm, _, _, _ in 前额.前额网.输入路:
        s = 回[nm]
        if s.any():
            亮[网.区段(nm)] |= s
    出 = 网.步进(亮)
    身.步进(运动.解码发力(出[网.区起点["运动"]:网.区起点["运动"] + 运动.运动区宽度]))
    亮 = 出
    if k == 60:
        报("起身60拍", 前)

print("=== 戊：站着看红 60 拍时前额叶的真实状态 ===")
身 = 身体.身体(); 身.摆成站姿(); 身.走(0.5, None)
图 = np.zeros((1080, 1920, 3), dtype=np.uint8); 图[:, 640:1280, 0] = 255
红 = np.asarray(视觉.网.前向传播(视觉.图像转信号(图)), dtype=bool)
亮 = np.zeros(网.总数, dtype=bool)
for k in range(60):
    for 区名, 区 in 感觉.三个感觉区():
        c = 网.区起点[区名]
        亮[c:c + 区.输出层数量] = 区.状态转激活(区.读身体(身))
    本能.灌视觉(亮, 网, 第4层=红)
    各路 = {nm: 亮[网.区段(nm)] for nm, _, _, _ in 前额.前额网.输入路}
    前 = np.asarray(前额.前额网.前向传播(各路), dtype=bool)
    亮[前起:前起 + 前宽] = 前
    回 = 前额.返回线.返回(前)
    for nm, _, _, _ in 前额.前额网.输入路:
        s = 回[nm]
        if s.any():
            亮[网.区段(nm)] |= s
    出 = 网.步进(亮)
    身.步进(运动.解码发力(出[网.区起点["运动"]:网.区起点["运动"] + 运动.运动区宽度]))
    亮 = 出
报("看红60拍", 前)
