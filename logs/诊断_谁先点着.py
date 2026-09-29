# -*- coding: utf-8 -*-
"""诊断_谁先点着.py —— 走起来的那一刻，到底是「前进1.2」里哪个时刻细胞先亮的？它收到多少电流？"""
from __future__ import annotations

import numpy as np

import 本能工具_instincts as 本能
import 运动记忆区_motor_memory as 运动记忆
import 实验_颜色渐变 as 渐变
import 身体_go2 as 身体
import 主循环_完整的一拍 as 主循环

种子们 = [20260914, 20260918]
名 = 本能.读名字()
for s in 种子们:
    本能.建网种子 = s
    脑 = 主循环.脑(学=False, 说=False)
    网 = 脑.网
    运记起 = 网.区起点["运动记忆"]
    时刻 = 运记起 + np.asarray(运动记忆.时刻们("前进1.2"), dtype=np.int64)
    掩 = np.isin(网._兴奋目, 时刻)
    源, 权 = 网._兴奋源[掩], 网._兴奋权[掩].astype(np.float64)
    身 = 身体.身体()
    身.摆成站姿()
    身.走(0.5, None)
    print("=" * 78)
    print("种子 %d：前进1.2 一共 %d 个时刻；打进它们的连接 %d 根" % (s, len(时刻), 源.size))
    print("%4s | %-28s | %-28s" % ("拍", "亮着的时刻（下标）", "每个时刻收到的兴奋-抑制"))
    for k in range(6):
        脑.一拍(图=渐变.彩色块(0), 身=身)
        身.步进(脑.肌肉发力())
        亮曲 = 脑.上一拍
        们 = np.nonzero(亮曲[时刻])[0]
        兴 = np.zeros(len(时刻))
        for i, t in enumerate(时刻):
            m = 网._兴奋目 == t
            兴[i] = float(网._兴奋权[m].astype(np.float64)[亮曲[网._兴奋源[m]]].sum())
        抑 = 网._抑制电流(亮曲)[时刻]
        净 = 兴 - 抑
        print("%4d | %-28s | 最好 %+.3f（第 %d 个）"
              % (k, "、".join(str(int(i)) for i in 们[:6]) or "—",
                 float(净.max()), int(净.argmax())))