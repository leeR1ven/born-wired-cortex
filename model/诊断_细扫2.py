import pathlib
# -*- coding: utf-8 -*-
"""在 50Hz 上细调「肌肉刚柔 + 力矩上限」，并打出赢家逐个动作的成绩单。"""
import json, pathlib
import numpy as np
import 身体_go2 as 身体

根 = pathlib.Path(__file__).resolve().parent
库 = json.loads((根 / "动作库.json").read_text(encoding="utf-8"))
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
    n = len(拍); i = int(秒 / 帧秒) - 1
    for i in range(int(秒 / 帧秒)):
        身.步进(拍[i % n])
        if 身.机身歪了() > 75: break
    p = 身.数据.qpos
    return i * 帧秒, float(np.hypot(p[0]-起[0], p[1]-起[1]))

行 = []
for k, d in [(60,3),(80,4),(100,5),(120,6)]:
    for τ in (25.0, 45.0):
        明细 = {名: 跑(名, v, 1/50, k, d, τ) for 名, v in 库.items()}
        t = [x[0] for x in 明细.values()]; m = [x[1] for x in 明细.values()]
        走动 = sum(1 for x in m if x > 0.25)
        行.append((float(np.mean(t)), min(t), k, d, τ, 走动, 明细))
        print("50Hz k%g d%g τ%g : 平均 %.2f  最差 %.2f  走动的 %d/26" % (k, d, τ, np.mean(t), min(t), 走动))
最好 = max(行, key=lambda x: (x[6].__len__() and round(x[0],2), x[1]))
print("\n赢家：50Hz k%g d%g τ%g —— 逐个动作成绩单" % (最好[2], 最好[3], 最好[4]))
for 名, (t, m) in 最好[6].items():
    print("   %-10s 撑住 %4.1f 秒   走了 %5.2f 米   %s" % (名, t, m, "✓" if t >= 4.0 else "✗"))