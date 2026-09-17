# -*- coding: utf-8 -*-
"""实验_放大_色相红块.py —— 放大画面以后，R4 那两半（视觉:中有红 / 前额叶:看着红）还挑不挑颜色。

和 诊断_色相_红块.py 算的是同一件事、同一套权重（视觉 ×0.80、前额叶 ×0.85、阈值 1.0），
只是**按画面宽度自动变**，而且写出来的日志带后缀，不会盖掉论文那一条。

命令：python 实验_放大_色相红块.py
"""
from __future__ import annotations
import pathlib

import io
import os

import numpy as np

import 本能工具_instincts as 本能
import 视觉前处理_visual_preprocess as 视觉
import 实验_颜色渐变 as 渐变
import 前额叶区_prefrontal as 前额

权重视觉 = 0.80
权重前额 = 0.85
阈值 = 1.0
后缀 = os.environ.get("AGI数据后缀", "")

本能.建网种子 = 20260914
网, 条 = 本能.装(说=False)
名字 = 本能.读名字()
视红 = np.array(名字["视觉"]["中有红"], dtype=np.int64)
前红 = np.array(名字["前额叶"]["看着红"], dtype=np.int64)
空听 = np.zeros(网.区宽["听觉"], dtype=bool)
空运 = np.zeros(网.区宽["运动"], dtype=bool)

出 = io.StringIO()
出.write("画面 %d x %d；「视觉:中有红」%d 个细胞、「前额叶:看着红」%d 个细胞"
         "（种子 20260914，本能表 %d 条）\n"
         % (视觉.大像素行数, 视觉.大像素列数, 视红.size, 前红.size, len(本能.读本能表())))
出.write("走路的两半：视觉 ×%.2f、前额叶 ×%.2f，阈值 %.2f\n\n" % (权重视觉, 权重前额, 阈值))
出.write("  色相 |   视觉红亮/总数 | 前额叶红亮/总数 |  视觉给的 | 前额叶给的 |   净电流 | 过阈值? | rgb\n")
出.write("-" * 112 + "\n")
for h in (0, 24, 48, 72, 96, 120, 144, 168, 180, 204, 240, 270, 300, 330):
    图 = 渐变.彩色块(h)
    out = np.asarray(视觉.网.前向传播(视觉.图像转信号(图)), dtype=bool)
    n视 = int(out[视红].sum())
    前 = np.asarray(前额.前额网.前向传播(out, 空听, 空运), dtype=bool)
    n前 = int(前[前红].sum())
    视半 = 权重视觉 * n视 / float(视红.size)
    前半 = 权重前额 * n前 / float(前红.size)
    出.write("%6d | %5d / %4d   | %5d / %4d    | %8.2f | %9.2f | %8.2f | %6s | %s\n"
             % (h, n视, 视红.size, n前, 前红.size, 视半, 前半, 视半 + 前半,
                "是" if 视半 + 前半 >= 阈值 else "否", tuple(int(x) for x in 图[500, 960])))
出.write("\n（黑屏两半都是 0，所以没列进来；阈值只是把两半加起来跟 1.0 比，"
         "实际能不能走要看 实验_放大_走路边界.py 的实测）\n")
文本 = 出.getvalue()
print(文本)
io.open(str(pathlib.Path(__file__).resolve().parent.parent / "logs" / "日志_色相红块%s.log") % 后缀, "w", encoding="utf-8", newline="\n").write(文本)