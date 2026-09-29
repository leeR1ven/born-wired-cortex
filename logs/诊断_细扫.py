import pathlib
# -*- coding: utf-8 -*-
"""细扫：把全部 26 个动作都算进去，找回放最稳的一档「节奏 + 肌肉刚柔 + 力矩上限」。"""
import json, pathlib
import numpy as np
import 身体_go2 as 身体

根 = pathlib.Path(__file__).resolve().parent
库 = json.loads((根 / "动作库.json").read_text(encoding="utf-8"))
全部 = list(库.keys())
秒 = 5.0
起手趴 = {"站起来"}

def 跑(名, v, 帧秒, k, d, τ):
    拍 = np.array(v["拍"], dtype=float)
    身体.刚度 = np.full(12, float(k)); 身体.阻尼 = np.full(12, float(d)); 身体.最大力矩 = np.full(12, float(τ))
    身 = 身体.身体()
    身.每帧物理步 = max(1, int(round(帧秒 / 身.模型.opt.timestep)))
    if 名 in 起手趴: 身.摆成趴姿()
    else: 身.摆成站姿()
    身.走(0.5, None)
    起 = 身.数据.qpos[:2].copy()
    n = len(拍)
    for i in range(int(秒 / 帧秒)):
        身.步进(拍[i % n])
        if 身.机身歪了() > 75: break
    else:
        i = int(秒 / 帧秒) - 1
    p = 身.数据.qpos
    return i * 帧秒, float(np.hypot(p[0]-起[0], p[1]-起[1]))

候选 = [(1/30, k, d, 25.0) for k, d in [(10,0.5),(15,0.75),(20,1),(25,1.25),(30,1.5),(40,2)]] + \
       [(1/50, k, d, 25.0) for k, d in [(40,2),(60,3),(80,4),(100,5),(150,8)]]
print("%-22s %8s %8s %8s   %s" % ("节奏/k/d/τ", "平均撑住", "最差", "走动的", "摔得最快的三个"))
结果 = []
for 帧秒, k, d, τ in 候选:
    行 = []
    for 名, v in 库.items():
        t, m = 跑(名, v, 帧秒, k, d, τ)
        行.append((名, t, m))
    平均 = np.mean([t for _, t, _ in 行]); 最差 = min(t for _, t, _ in 行)
    走动的 = sum(1 for _, t, m in 行 if m > 0.25)
    坏 = sorted([x for x in 行 if x[1] < 4.0], key=lambda x: x[1])[:3]
    结果.append((平均, 最差, 帧秒, k, d, τ, 坏))
    标 = "%s k%g d%g τ%g" % ("30Hz" if abs(帧秒-1/30) < 1e-9 else "50Hz", k, d, τ)
    print("%-22s %8.2f %8.2f %8d   %s" % (标, 平均, 最差, 走动的,
          ", ".join("%s %.1f" % (a, t) for a, t, _ in 坏)))
最好 = max(结果)
print("\n最好的一档：%s k%g d%g τ%g  平均 %.2f 秒" % ("30Hz" if abs(最好[2]-1/30)<1e-9 else "50Hz", 最好[3], 最好[4], 最好[5], 最好[0]))