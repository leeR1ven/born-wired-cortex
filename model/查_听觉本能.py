# -*- coding: utf-8 -*-
"""查_听觉本能.py —— 给一声「响」，看 运动记忆:原地踏步 那串时刻到底走没走起来。"""
from __future__ import annotations

import numpy as np

import 主循环_完整的一拍 as 主循环
import 本能工具_instincts as 本能
import 运动记忆区_motor_memory as 运动记忆

脑 = 主循环.脑(学=False, 说=False)
网 = 脑.网
运记起 = 网.区起点["运动记忆"]
时刻 = 运记起 + np.asarray(运动记忆.时刻们("原地踏步"), dtype=np.int64)
谱 = 本能.听觉例子["响"]()
原位 = {int(t): i for i, t in enumerate(时刻)}
脑.亮[:] = False
脑.上一拍[:] = False
print("%4s | %6s | %s" % ("拍", "亮了几个时刻", "哪几个"))
for k in range(12):
    脑.一拍(声=谱)
    亮曲 = 脑.上一拍
    们 = np.nonzero(亮曲[时刻])[0]
    print("%4d | %10d | %s" % (k, 们.size, "、".join(str(int(i)) for i in 们[:8])))