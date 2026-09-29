# -*- coding: utf-8 -*-
"""诊断_踏步点火.py —— 给一声「响」，看 运动记忆:原地踏步 那个头细胞收到的电流从哪来。"""
from __future__ import annotations

import sys

import numpy as np

import 主循环_完整的一拍 as 主循环
import 本能工具_instincts as 本能
import 运动记忆区_motor_memory as 运动记忆

脑 = 主循环.脑(学=False, 说=False)
网 = 脑.网
名字 = 本能.读名字()
运记起 = 网.区起点["运动记忆"]
谱 = 本能.听觉例子["响"]()
时刻 = 运记起 + np.asarray(运动记忆.时刻们("原地踏步"), dtype=np.int64)
目标 = int(时刻[0])
print("原地踏步一共 %d 个时刻，头一拍细胞 = %d" % (len(时刻), 目标))

掩 = 网._兴奋目 == 目标
源 = 网._兴奋源[掩]
权 = 网._兴奋权[掩].astype(np.float64)
区名 = np.empty(网.总数, dtype=object)
for 名, 起 in 网.区起点.items():
    区名[起:起 + 网.区宽[名]] = 名
块 = {}
for 区, 表 in 名字.items():
    起 = 网.区起点[区]
    for 名, idx in 表.items():
        块["%s:%s" % (区, 名)] = 起 + np.asarray(idx, dtype=np.int64)

print("打进这个细胞的兴奋连接 %d 根，来自这些名字块：" % 源.size)
for 名, idx in 块.items():
    m = np.isin(源, idx)
    if m.any():
        print("   %-22s %3d 根，权重合计 %.3f" % (名, int(m.sum()), float(权[m].sum())))

print()
print("%4s | %8s | %8s | %8s | %10s" % ("拍", "听觉亮", "响名亮", "兴奋", "抑制"))
脑.亮[:] = False
脑.上一拍[:] = False
听起, 听宽 = 脑.听起, 脑.听宽
for k in range(8):
    脑.一拍(声=谱)
    亮曲 = 脑.上一拍
    贡 = 权 * 亮曲[源]
    抑 = float(网._抑制电流(亮曲)[目标])
    听亮 = int(亮曲[听起:听起 + 听宽].sum())
    响亮 = float(亮曲[块["听觉:响"]].mean())
    print("%4d | %8d | %8.2f | %8.3f | %8.3f" % (k, 听亮, 响亮, float(贡.sum()), 抑))