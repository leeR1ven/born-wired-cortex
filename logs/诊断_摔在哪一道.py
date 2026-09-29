# -*- coding: utf-8 -*-
"""诊断_摔在哪一道.py —— 同一串动作，摔是摔在下面哪一道？

  (a) 精确发力直接推身体           = 身体本身走不走得动
  (b) 发力只留 10 档（皮层就是这么表示的） = 量化掉多少
  (c) 走一遍皮层网（验收就是这么做的）     = 皮层再掉多少
  (d) 换一个起步拍（离站姿最近的那个）     = 起步选得好不好

命令：python 诊断_摔在哪一道.py
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

import 身体_go2 as 身体
import 运动输出区_motor_output as 运动
import 本能工具_instincts as 本能

根 = pathlib.Path(__file__).resolve().parent
帧秒 = 1.0 / 50.0
库 = json.loads((根 / "动作库.json").read_text(encoding="utf-8"))
验收 = json.loads((根 / "验收_动作库.json").read_text(encoding="utf-8"))
名字 = 本能.读名字()
网, _ = 本能.装(说=False)
前起 = 网.区起点["运动记忆"]
运起 = 网.区起点["运动"]

手编 = {"站姿", "趴姿", "坐", "蹲", "站高", "抬左前腿", "抬右前腿", "趴下", "趴姿低",
       "蹲坐", "站起来", "趴下去", "坐下", "抬腿", "低头"}
摔的 = [k for k in 验收 if k not in 手编 and 验收[k]["撑住秒"] < 3.9]
print("现在摔的有 %d 个（%.0f%%），挑 6 个来看" % (len(摔的), 100 * len(摔的) / (len(库) - len(手编))))
挑 = 摔的[:6]


def 推身体(发力, 秒=4.0):
    身 = 身体.身体()
    身.摆成站姿()
    for i in range(int(秒 / 帧秒)):
        身.步进(np.clip(发力[i % len(发力)], 0, 1))
        if 身.机身歪了() > 75:
            break
    return i * 帧秒


def 走皮层(名, 秒=4.0):
    身 = 身体.身体()
    身.摆成站姿()
    亮 = np.zeros(网.总数, dtype=bool)
    亮[前起 + np.array(名字["运动记忆"][名], dtype=int)] = True
    for i in range(int(秒 / 帧秒)):
        出 = 网.步进(亮)
        身.步进(运动.解码发力(出[运起:运起 + 运动.运动区宽度]))
        亮 = 出
        if 身.机身歪了() > 75:
            break
    return i * 帧秒


for 名 in 挑:
    拍 = np.array(库[名]["拍"], dtype=float)
    if len(拍) == 1:
        continue
    a = 推身体(拍)
    b = 推身体(np.round(拍 * 10) / 10)
    c = 走皮层(名)
    print("%-24s %2d拍  精确 %.1f 秒 | 只留10档 %.1f 秒 | 走皮层 %.1f 秒" % (名, len(拍), a, b, c))