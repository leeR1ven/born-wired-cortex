import pathlib
# -*- coding: utf-8 -*-
"""诊断_步态相位.py —— 我们脑子里那条走路，节拍对不对？开环回放能走多快？"""
import os, sys, time
import numpy as np
os.environ.setdefault("前额叶不保留", "1")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); os.chdir(str(pathlib.Path(__file__).resolve().parent))
import 本能工具_instincts as 本能
import 运动记忆区_motor_memory as 运动记忆
import 运动输出区_motor_output as 运动
import 身体_go2 as 身体

名字 = 本能.读名字()
网, 条 = 本能.装(说=False)
运记起 = 网.区起点["运动记忆"]
运起 = 网.区起点["运动"]

# ---------- (1) 开环回放：只点第一拍，让链子自己走 ----------
print("【1】开环回放（只点「前进1.2」第一拍，没有眼睛没有前额叶）")
身 = 身体.身体(); 身.摆成站姿(); 身.走(0.5, None)
亮 = np.zeros(网.总数, dtype=bool)
亮[运记起 + np.array(名字["运动记忆"]["前进1.2"], dtype=int)] = True
起 = 身.数据.qpos[:2].copy()
相位 = []
for k in range(200):
    出 = 网.步进(亮)
    发力 = 运动.解码发力(出[运起:运起 + 运动.运动区宽度])
    身.步进(发力)
    亮 = 出
    活 = np.nonzero(亮[运记起 + 运动记忆.时刻们("前进1.2")])[0]
    相位.append(活.tolist())
末 = 身.数据.qpos[:2]
走 = float(np.hypot(末[0] - 起[0], 末[1] - 起[1]))
print("  4 秒走了 %.2f 米（%.2f 米/秒）；末了机身高 %.3f 米 歪 %.1f 度"
      % (走, 走 / 4.0, 身.机身高度(), 身.机身歪了()))
第一拍 = [p[0] for p in 相位 if len(p) == 1 and p[0] == 0]
print("  15 个时刻里，同时活的个数：%s"
      % sorted({len(p) for p in 相位}))
print("  前 40 拍每拍活的是第几个时刻：")
print("   ", [p[0] if len(p) == 1 else p for p in 相位[:40]])
# 一个周期多少拍：第一次回到时刻0 的间隔
回 = [k for k in range(1, 200) if 0 in 相位[k] and 0 not in 相位[k - 1]]
if len(回) >= 2:
    print("  回到第一个时刻的间隔（拍）：%s  -> 周期约 %.3f 秒"
          % (np.diff(回)[:6].tolist(), float(np.mean(np.diff(回))) * 0.02))

# ---------- (2) 整脑子走：同一条链子，节拍一样吗 ----------
print("")
print("【2】整脑子（红在正前方、带前额叶、不带学习）")
import 主循环_完整的一拍 as 主
脑 = 主.脑(学=False, 说=False)
身2 = 身体.身体(); 身2.摆成站姿()
图 = 主.红图("中")
起2 = None
相位2 = []
for k in range(200):
    脑.一拍(图=图, 身=身2)
    身2.步进(脑.肌肉发力())
    if k == 0:
        起2 = 身2.数据.qpos[:2].copy()
    活 = np.nonzero(脑.上一拍[运记起 + 运动记忆.时刻们("前进1.2")])[0]
    相位2.append(活.tolist())
末2 = 身2.数据.qpos[:2]
走2 = float(np.hypot(末2[0] - 起2[0], 末2[1] - 起2[1]))
print("  4 秒走了 %.2f 米（%.2f 米/秒）；末了机身高 %.3f 米 歪 %.1f 度"
      % (走2, 走2 / 4.0, 身2.机身高度(), 身2.机身歪了()))
print("  同时活的个数：%s" % sorted({len(p) for p in 相位2}))
print("  前 40 拍每拍活的是第几个时刻：")
print("   ", [p[0] if len(p) == 1 else p for p in 相位2[:40]])
回2 = [k for k in range(1, 200) if 0 in 相位2[k] and 0 not in 相位2[k - 1]]
if len(回2) >= 2:
    print("  回到第一个时刻的间隔（拍）：%s  -> 周期约 %.3f 秒"
          % (np.diff(回2)[:6].tolist(), float(np.mean(np.diff(回2))) * 0.02))
else:
    print("  （这 4 秒里没回到第一个时刻；现在活的是第 %s 个"
          % (相位2[-1][0] if len(相位2[-1]) == 1 else 相位2[-1]))