import pathlib
# -*- coding: utf-8 -*-
"""快速诊断：同一个抄下来的动作，换不同的「节奏 + 肌肉参数」回放，看还摔不摔。"""
import json, pathlib, sys, time
import numpy as np
import 身体_go2 as 身体

根 = pathlib.Path(__file__).resolve().parent
库 = json.loads((根 / "动作库_训练好的.json").read_text(encoding="utf-8"))

条件 = {
    "A 现在的：30Hz k60 d3 τ25":      dict(帧秒=1/30, k=[60.0]*12, d=[3.0]*12, τ=[25.0]*12),
    "B 只改节奏：50Hz k60 d3 τ25":    dict(帧秒=1/50, k=[60.0]*12, d=[3.0]*12, τ=[25.0]*12),
    "C 只改增益：30Hz 老师的k d τ":   dict(帧秒=1/30, k=[20,20,40]*4, d=[1,1,2]*4, τ=[23.7,23.7,45.43]*4),
    "D 全照老师：50Hz 老师的k d τ":   dict(帧秒=1/50, k=[20,20,40]*4, d=[1,1,2]*4, τ=[23.7,23.7,45.43]*4),
}

动作 = ["慢走", "中走", "右横移", "左转", "边走边左转", "快走", "左横移"]
秒 = 5.0

def 跑一遍(名, 条):
    拍 = np.array(库[名]["发力"], dtype=float)          # 一个周期
    身体.刚度 = np.array(条["k"]); 身体.阻尼 = np.array(条["d"]); 身体.最大力矩 = np.array(条["τ"])
    身 = 身体.身体()
    身.每帧物理步 = max(1, int(round(条["帧秒"] / 身.模型.opt.timestep)))
    身.摆成站姿(); 身.走(0.5, None)
    起 = 身.数据.qpos[:2].copy()
    撑 = 0
    总 = int(秒 / 条["帧秒"])
    for k in range(总):
        身.步进(拍[k % len(拍)])
        撑 = k
        if 身.机身歪了() > 75: break
    p = 身.数据.qpos
    return 撑 * 条["帧秒"], float(np.hypot(p[0]-起[0], p[1]-起[1]))

print("条件 × 动作： 撑住秒 / 走了米   （5 秒满分）")
print("%-28s" % "条件", " ".join("%9s" % a for a in 动作))
for cn, 条 in 条件.items():
    行 = []
    for a in 动作:
        t, m = 跑一遍(a, 条)
        行.append("%5.1f/%.2f" % (t, m))
    print("%-28s" % cn, " ".join("%9s" % x for x in 行))