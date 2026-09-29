# -*- coding: utf-8 -*-
"""诊断_机身速度.py —— 站着的时候，机身"前后速度"到底是多少？
用来判断 机身状态:在往前走 这条名字的档位是不是定得太低（站着抖一下就算"在往前走"）。
"""
from __future__ import annotations

import numpy as np

import 主循环_完整的一拍 as 主循环
import 身体_go2 as 身体
import 本能工具_instincts as 本能

帧秒 = 主循环.帧秒
图高, 图宽 = 主循环.图高, 主循环.图宽


def 红带():
    图 = np.zeros((图高, 图宽, 3), dtype=np.uint8)
    图[:, 图宽 // 3: 2 * 图宽 // 3, 0] = 255
    return 图


脑 = 主循环.脑(学=False, 说=False, 前额叶按灭=True)
名字 = 本能.读名字()
起 = 脑.网.区起点["机身状态"]
块 = 起 + np.asarray(名字["机身状态"]["在往前走"], dtype=np.int64)

身 = 身体.身体()
身.摆成站姿()
身.走(0.5, None)
脑.要学 = False
脑.前额叶按灭 = True
图 = 红带()
v = []
for k in range(200):
    脑.一拍(图=图, 身=身)
    身.步进(脑.肌肉发力())
    速 = float(身.数据.qvel[0])
    v.append((k, 速, 身.数据.qpos[2], int(脑.上一拍[块].sum())))
    if k < 45 or k % 10 == 0:
        print("拍%3d  前后速度 %+6.3f 米/秒  → 折成特征值 %+5.2f  机身 %.3f 米  「在往前走」亮 %d/%d"
              % (k, 速, np.clip(速 / 2.0, -1, 1), 身.数据.qpos[2], 脑.上一拍[块].sum(), len(块)))

走着 = [a for a in v if a[3] > 0]
站着 = [a for a in v if a[3] == 0]
print()
print("「在往前走」亮着的时候：前后速度 %+0.3f ~ %+0.3f 米/秒（%d 拍）"
      % (min(a[1] for a in 走着), max(a[1] for a in 走着), len(走着)))
print("「在往前走」不亮的时候：前后速度 %+0.3f ~ %+0.3f 米/秒（%d 拍）"
      % (min(a[1] for a in 站着), max(a[1] for a in 站着), len(站着)))
