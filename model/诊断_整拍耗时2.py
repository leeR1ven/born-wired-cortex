import pathlib
# -*- coding: utf-8 -*-
"""诊断_整拍耗时2.py —— 真实闭环里"带学习"那一拍，时间到底花在哪（含已经学出来的新连接）。"""
import cProfile, io, os, pstats, sys, time
import numpy as np
os.environ.setdefault("前额叶不保留", "1")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); os.chdir(str(pathlib.Path(__file__).resolve().parent))
import 主循环_完整的一拍 as 主
import 身体_go2 as 身体
import 世界_world as 世界

脑 = 主.脑(学=True, 说=False)
网 = 脑.网
身 = 身体.身体(); 身.摆成站姿()
世 = 世界.世界()
世.物体.append((1.2, 0.0, 0.30, 0.25, (255, 0, 0)))
for k in range(120):
    图 = 世.看(); 脑.一拍(图=图, 身=身); 身.步进(脑.肌肉发力())
print("跑热 120 拍之后：临时表 %d 个源 / %d 根连接；候选表 %d 对；亮着 %d 个细胞"
      % (len(网._待出), sum(len(v) for v in 网._待出.values()), len(网._候选), int(脑.亮.sum())))
训练前 = sum(len(v) for v in 网._待出.values())
t0 = time.perf_counter()
for k in range(30):
    图 = 世.看(); 脑.一拍(图=图, 身=身); 身.步进(脑.肌肉发力())
print("再走 30 拍：每拍 %.0f 毫秒；临时表长到 %d 根"
      % ((time.perf_counter() - t0) / 30 * 1000, sum(len(v) for v in 网._待出.values())))
pr = cProfile.Profile(); pr.enable()
for k in range(30):
    脑.一拍(图=图, 身=身)
pr.disable()
s = io.StringIO()
pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(10)
print(s.getvalue())