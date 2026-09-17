import pathlib
# -*- coding: utf-8 -*-
"""诊断_整拍耗时.py —— 真正的一拍（世界出画面 + 皮层 + 前额叶 + 学习）要多少毫秒。

和 规模试验_测量.py 的区别：那个只量"皮层网走一步"，这个量**闭环里真实的一拍**。
"""
import os, sys, time
import numpy as np
os.environ.setdefault("前额叶不保留", "1")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); os.chdir(str(pathlib.Path(__file__).resolve().parent))
import 主循环_完整的一拍 as 主
import 身体_go2 as 身体
import 世界_world as 世界


def 量(要学, 拍数=150):
    脑 = 主.脑(学=要学, 说=False)
    身 = 身体.身体(); 身.摆成站姿()
    世 = 世界.世界()
    世.物体.append((1.2, 0.0, 0.30, 0.25, (255, 0, 0)))     # 正前方一个红块
    时 = dict(看=0.0, 脑=0.0, 身=0.0)
    for k in range(20):                                     # 先跑热
        图 = 世.看(); 脑.一拍(图=图, 身=身); 身.步进(脑.肌肉发力())
    for k in range(拍数):
        t0 = time.perf_counter()
        图 = 世.看()
        t1 = time.perf_counter()
        脑.一拍(图=图, 身=身)
        t2 = time.perf_counter()
        身.步进(脑.肌肉发力())
        t3 = time.perf_counter()
        时["看"] += t1 - t0; 时["脑"] += t2 - t1; 时["身"] += t3 - t2
    总 = sum(时.values()) / 拍数 * 1000
    return 总, {k: v / 拍数 * 1000 for k, v in 时.items()}, 脑


print("24x16 闭环里真实的一拍（世界 24x16 出画面）")
for 要学 in (False, True):
    总, 明细, 脑 = 量(要学)
    print("  学习%s：一拍 %.0f 毫秒（看 %.0f + 皮层 %.0f + 身体 %.0f）；末了亮着 %d 个细胞"
          % ("开" if 要学 else "关", 总, 明细["看"], 明细["脑"], 明细["身"], int(脑.亮.sum())))