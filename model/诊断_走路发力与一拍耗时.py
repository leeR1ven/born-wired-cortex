import pathlib
# -*- coding: utf-8 -*-
"""诊断_走路发力与一拍耗时.py —— 两件事一次问清：
   (1) 我们脑子里走路时 12 块肌肉实际发多大力（和抄来的那条比，差在哪）
   (2) 一拍到底慢在哪（带学习 / 不带学习，各占多少）
"""
import json, io, os, sys, time, cProfile, pstats
import numpy as np
os.environ.setdefault("前额叶不保留", "1")      # 论文 R1-R8 那一档
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
os.chdir(str(pathlib.Path(__file__).resolve().parent))
import 主循环_完整的一拍 as 主
import 运动记忆区_motor_memory as 运动记忆
import 身体_go2 as 身体

print("建一颗脑子（24x16 默认）……", flush=True)
t0 = time.perf_counter()
脑 = 主.脑(学=True, 说=False)
print("建好，%.1f 秒；皮层 %d 个神经元" % (time.perf_counter() - t0, 脑.网.总数), flush=True)
身 = 身体.身体()
身.摆成站姿()

# ---------- (1) 走路：红在正前方，走 200 拍（4 秒）----------
图 = 主.红图("中")
发历史 = []
位历史 = []
运记起 = 脑.网.区起点["运动记忆"]
走时刻 = 运记起 + np.asarray(运动记忆.时刻们("前进1.2"), dtype=np.int64)
点火 = 0
t0 = time.perf_counter()
for k in range(200):
    脑.一拍(图=图, 身=身)
    发力 = 脑.肌肉发力()
    身.步进(发力)
    发历史.append(发力.copy())
    位历史.append((身.数据.qpos[0], 身.数据.qpos[1]))
    if 脑.上一拍[走时刻].any():
        点火 += 1
带学习每拍 = (time.perf_counter() - t0) / 200
位历史 = np.array(位历史)
前进 = float(位历史[-1, 0] - 位历史[0, 0])
print("")
print("【1】走路（红在正前方，200 拍 = 4 秒）")
print("  往前走 %.2f 米；走路那 15 个时刻点火 %d/200 拍" % (前进, 点火))
print("  大脑发出去的发力：最小 %.2f 最大 %.2f 平均 %.2f" %
      (min(a.min() for a in 发历史), max(a.max() for a in 发历史),
       float(np.mean([a.mean() for a in 发历史]))))

库 = json.loads(io.open(str(pathlib.Path(__file__).resolve().parent / "动作库_训练好的.json"), encoding="utf-8").read())
条 = None
for k, v in 库.items():
    if v.get("指令") == [1.2, 0.0, 0.0]:
        条 = (k, v); break
if 条 is None:
    for k, v in 库.items():
        if abs(v["指令"][0] - 1.2) < 1e-9 and abs(v["指令"][1]) < 1e-9 and abs(v["指令"][2]) < 1e-9:
            条 = (k, v); break
名, v = 条
库发 = np.array(v["发力"], dtype=float)
print("  抄来的「%s」：%d 拍一个周期 %.3f 秒，库里的发力 最小 %.2f 最大 %.2f 平均 %.2f"
      % (名, v["一拍个数"], v["周期秒"], 库发.min(), 库发.max(), float(库发.mean())))
print("  老师的记录：3 秒走 %.2f 米（%.2f 米/秒）；我们这 4 秒 %.2f 米（%.2f 米/秒）"
      % (v["走动米"], v["走动米"] / 3.0, 前进, 前进 / 4.0))
# 逐块肌肉比一比（我们走路中间那 60 拍的平均 vs 库里的平均）
中 = np.array([a for a in 发历史[70:130]])
print("  逐块肌肉（我们 70~130 拍的平均） vs （抄来的平均）：")
for i in range(12):
    print("     %-8s 我们 %.2f  抄来 %.2f" % (身体.肌肉名[i], 中[:, i].mean(), 库发[:, i].mean()))

# ---------- (2) 一拍耗时 ----------
print("")
print("【2】一拍要多久")
print("  上面那次：带学习 %.0f 毫秒/拍" % (带学习每拍 * 1000))
print("  （身体一拍 20 毫秒）")
脑2 = 主.脑(学=False, 说=False)
身2 = 身体.身体(); 身2.摆成站姿()
for _ in range(10):
    脑2.一拍(图=图, 身=身2)
t0 = time.perf_counter()
for _ in range(100):
    脑2.一拍(图=图, 身=身2)
print("  另一颗脑子不带学习：%.1f 毫秒/拍" % ((time.perf_counter() - t0) / 100 * 1000))

pr = cProfile.Profile()
pr.enable()
for _ in range(30):
    脑.一拍(图=图, 身=身)
pr.disable()
print("")
print("【3】带学习时，时间花在哪（前 12 名，按自身耗时）")
s = io.StringIO()
pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(12)
for 行 in s.getvalue().splitlines():
    print("  " + 行)