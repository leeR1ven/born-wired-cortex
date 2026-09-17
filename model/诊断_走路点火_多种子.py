# -*- coding: utf-8 -*-
"""诊断_走路点火_多种子.py —— 「前进1.2」那个头细胞，五个种子里每个收到多少兴奋、多少抑制。"""
from __future__ import annotations

import numpy as np

import 本能工具_instincts as 本能
import 运动记忆区_motor_memory as 运动记忆
import 实验_颜色渐变 as 渐变
import 身体_go2 as 身体
import 主循环_完整的一拍 as 主循环

种子们 = [20260914, 20260915, 20260916, 20260917, 20260918]
名 = 本能.读名字()
print("%10s | %8s | %8s | %8s | %8s | %s" % ("种子", "视觉那半", "前额叶那半", "兴奋合计", "抑制", "净"))
print("-" * 74)
for s in 种子们:
    本能.建网种子 = s
    脑 = 主循环.脑(学=False, 说=False)
    网 = 脑.网
    运记起 = 网.区起点["运动记忆"]
    目标 = 运记起 + int(名["运动记忆"]["前进1.2"][0])
    掩 = 网._兴奋目 == 目标
    源 = 网._兴奋源[掩]
    权 = 网._兴奋权[掩].astype(np.float64)
    块 = {}
    for 区, 表 in 名.items():
        起 = 网.区起点[区]
        for k2, idx in 表.items():
            块["%s:%s" % (区, k2)] = 起 + np.asarray(idx, dtype=np.int64)
    身 = 身体.身体()
    身.摆成站姿()
    身.走(0.5, None)
    时刻 = 渐变.拿时刻(脑)
    # 走 40 拍纯红，取"视觉和前额叶都亮着"的那些拍
    总_兴, 总_抑 = [], []
    最差 = None
    for k in range(40):
        脑.一拍(图=渐变.彩色块(0), 身=身)
        身.步进(脑.肌肉发力())
        亮曲 = 脑.上一拍
        贡 = 权 * 亮曲[源]
        抑 = float(网._抑制电流(亮曲)[目标])
        总_兴.append(float(贡.sum()))
        总_抑.append(抑)
        if 最差 is None or 贡.sum() - 抑 < 最差:
            最差 = 贡.sum() - 抑
        视半 = float(贡[np.isin(源, 块["视觉:中有红"])].sum())
        前半 = float(贡[np.isin(源, 块["前额叶:看着红"])].sum())
    print("%10d | %8.3f | %8.3f | %8.3f | %8.3f | %+.3f（最差 %+.3f）"
          % (s, 视半, 前半, float(np.mean(总_兴)), float(np.mean(总_抑)),
             float(np.mean(总_兴)) - float(np.mean(总_抑)), 最差))