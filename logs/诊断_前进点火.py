# -*- coding: utf-8 -*-
"""诊断_前进点火.py —— 「前进1.2」那个头细胞，每一拍收到的电流按来源拆开。

用户 2026-09-15 要的：把前额叶按灭，看到红却不该走起来（验证_前额叶.py 的第②项）。
实测它还走（164/200 拍），所以把电流摊开看看到底是谁把它点着的。
"""
from __future__ import annotations

import sys

import numpy as np

import 主循环_完整的一拍 as 主循环
import 身体_go2 as 身体
import 本能工具_instincts as 本能
import 运动记忆区_motor_memory as 运动记忆

帧秒 = 主循环.帧秒
图高, 图宽 = 主循环.图高, 主循环.图宽
按灭 = "按灭" in sys.argv[1:]


def 红带():
    图 = np.zeros((图高, 图宽, 3), dtype=np.uint8)
    图[:, 图宽 // 3: 2 * 图宽 // 3, 0] = 255
    return 图


脑 = 主循环.脑(学=False, 说=False, 前额叶按灭=按灭)
网 = 脑.网
名字 = 本能.读名字()
运记起 = 网.区起点["运动记忆"]
目标 = 运记起 + int(名字["运动记忆"]["前进1.2"][0])

掩 = 网._兴奋目 == 目标
源 = 网._兴奋源[掩]
权 = 网._兴奋权[掩].astype(np.float64)
区名 = np.empty(网.总数, dtype=object)
for 名, 起 in 网.区起点.items():
    区名[起:起 + 网.区宽[名]] = 名

# 名字块 -> 绝对下标
块 = {}
for 区, 表 in 名字.items():
    起 = 网.区起点[区]
    for 名, idx in 表.items():
        块["%s:%s" % (区, 名)] = 起 + np.asarray(idx, dtype=np.int64)

关心 = ["视觉:中有红", "视觉:左有红", "视觉:右有红",
        "机身状态:在往前走", "本体感觉:趴着", "本体感觉:站着", "前额叶:看着红"]


def 亮成(亮拍):
    出 = []
    for 名 in 关心:
        if 名 in 块:
            f = 亮拍[块[名]].mean()
            if f > 0.05:
                出.append("%s %.2f" % (名.split(":")[1], f))
    return "、".join(出) if 出 else "—"


def 电流拆(亮拍):
    贡献 = 权 * 亮拍[源]
    出 = {}
    for i in np.nonzero(贡献 > 0)[0]:
        k = 区名[源[i]]
        出[k] = 出.get(k, 0.0) + float(贡献[i])
    全部 = 网._抑制电流(亮拍)
    return 出, float(全部[目标])


身 = 身体.身体()
身.摆成站姿()
身.走(0.5, None)
脑.要学 = False
脑.前额叶按灭 = 按灭
脑.亮[:] = False
脑.上一拍[:] = False
图 = 红带()
时刻 = 运记起 + np.asarray(运动记忆.时刻们("前进1.2"), dtype=np.int64)

print("前额叶按灭 =", 按灭, "｜目标细胞 =", 目标)
print("%4s | %5s | %6s | %6s | %s" % ("拍", "点火", "兴奋", "抑制", "电流从哪来"))
for k in range(200):
    脑.一拍(图=图, 身=身)
    身.步进(脑.肌肉发力())
    亮曲 = 脑.上一拍          # 这一拍真正参与运算的
    出, 抑 = 电流拆(亮曲)
    点 = bool(亮曲[目标])
    总 = sum(出.values()) - 抑
    if k < 60 or k % 20 == 0:
        排 = sorted(出.items(), key=lambda x: -x[1])[:4]
        print("%4d | %5s | %6.2f | %6.2f | %s   ｜ 亮着：%s"
              % (k, "亮" if 点 else "  ", sum(出.values()), 抑,
                 "、".join("%s %.2f" % (a, b) for a, b in 排), 亮成(亮曲)))
