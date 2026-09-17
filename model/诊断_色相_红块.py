import pathlib
# -*- coding: utf-8 -*-
"""诊断_色相_红块.py —— 「红」这个名字块到底有多挑颜色？

★ 为什么要有这个：
  论文 R2 说「看见红 → 往前走」是**两半相加**过阈值：
     视觉:中有红 给 0.80，前额叶:看着红 给 0.85，阈值 1.0。
  那么「哪种颜色还能点着走路」就应该由「两半之和」预测。
  这个脚本把每一个色相下这两块各亮多少、加起来多少，一条一条量出来。

★ 它同时回答一个容易让人误会的问题：
  「红」这个名字块**并不是**只有红色才亮 —— 青色也能点亮它 42%。
  名字块是**标定**出来的（红图减空图的那批细胞），不是「红色这个概念」。

命令：python 诊断_色相_红块.py
"""
import io

import numpy as np

import 本能工具_instincts as 本能
import 实验_颜色渐变 as 渐变
import 视觉前处理_visual_preprocess as 视觉
import 前额叶区_prefrontal as 前额

权重视觉 = 0.80
权重前额 = 0.85
阈值 = 1.0

本能.建网种子 = 20260914
网, 条 = 本能.装(说=False)
名字 = 本能.读名字()
视起 = 网.区起点["视觉"]
前起 = 网.区起点["前额叶"]
视红 = np.array(名字["视觉"]["中有红"], dtype=np.int64)
前红 = np.array(名字["前额叶"]["看着红"], dtype=np.int64)
空听 = np.zeros(网.区宽["听觉"], dtype=bool)
空运 = np.zeros(网.区宽["运动"], dtype=bool)

行 = []
for h in (0, 24, 48, 72, 96, 120, 144, 168, 180, 204, 240, 270, 300, 330):
    图 = 渐变.彩色块(h)
    out = np.asarray(视觉.网.前向传播(视觉.图像转信号(图)), dtype=bool)
    n视 = int(out[视红].sum())
    前 = np.asarray(前额.前额网.前向传播(out, 空听, 空运), dtype=bool)
    n前 = int(前[前红].sum())
    视半 = 权重视觉 * n视 / float(视红.size)
    前半 = 权重前额 * n前 / float(前红.size)
    行.append((h, n视, 视红.size, n前, 前红.size, 视半, 前半, 视半 + 前半,
               tuple(int(x) for x in 图[500, 960])))

出 = io.StringIO()
出.write("「视觉:中有红」%d 个细胞、「前额叶:看着红」%d 个细胞（种子 20260914，本能表 585 条）\n"
         % (视红.size, 前红.size))
出.write("走路的两半：视觉 ×%.2f、前额叶 ×%.2f，阈值 %.2f\n\n" % (权重视觉, 权重前额, 阈值))
出.write("  色相 |   视觉红亮/总数 | 前额叶红亮/总数 |  视觉给的 | 前额叶给的 |   净电流 | 过阈值? | rgb\n")
出.write("-" * 112 + "\n")
for h, n视, 视总, n前, 前总, 视半, 前半, 和, rgb in 行:
    出.write("%6d | %5d / %4d   | %5d / %4d    | %8.2f | %9.2f | %8.2f | %6s | %s\n"
             % (h, n视, 视总, n前, 前总, 视半, 前半, 和, "是" if 和 >= 阈值 else "否", rgb))
出.write("\n（过阈值 是/否 只是把两半加起来跟 1.0 比 —— 是静态的一拍，实际走路要多拍才转起来，\n")
出.write("  所以「刚好压在 1.0 上下」的那些色相（96°、180°）要看 R4 的实测）\n")
文本 = 出.getvalue()
print(文本)
io.open(str(pathlib.Path(__file__).resolve().parent.parent / "logs" / '日志_色相红块.log'), 'w', encoding='utf-8', newline='\n').write(文本)