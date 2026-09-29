import pathlib
# -*- coding: utf-8 -*-
"""扫「大脑节奏 × 肌肉刚柔」，看哪一档回放抄下来的步态最稳。"""
import json, pathlib
import numpy as np
import 身体_go2 as 身体

根 = pathlib.Path(__file__).resolve().parent
库 = json.loads((根 / "动作库_训练好的.json").read_text(encoding="utf-8"))
难 = ["慢走", "中走", "左转", "快走", "右横移", "右转", "后退"]
秒 = 5.0

def 跑(名, 帧秒, k, d, τ):
    拍 = np.array(库[名]["发力"], dtype=float)
    身体.刚度 = np.full(12, float(k)); 身体.阻尼 = np.full(12, float(d)); 身体.最大力矩 = np.full(12, float(τ))
    身 = 身体.身体()
    身.每帧物理步 = max(1, int(round(帧秒 / 身.模型.opt.timestep)))
    身.摆成站姿(); 身.走(0.5, None)
    for i in range(int(秒 / 帧秒)):
        身.步进(拍[i % len(拍)])
        if 身.机身歪了() > 75: return i * 帧秒
    return 秒

print("每格 = 7 个难动作的平均撑住秒（5 秒满分）")
print("%-10s" % "节奏", "".join("%16s" % ("k%d d%g" % (k, d)) for k, d in
      [(20,1),(30,1.5),(40,2),(60,3),(80,4),(100,5),(150,8)]))
for 帧秒, 名 in ((1/30, "30Hz"), (1/50, "50Hz")):
    行 = []
    for k, d in [(20,1),(30,1.5),(40,2),(60,3),(80,4),(100,5),(150,8)]:
        平均 = np.mean([跑(a, 帧秒, k, d, 25.0) for a in 难])
        行.append("%16s" % ("%.2f" % 平均))
    print("%-10s" % 名, "".join(行))