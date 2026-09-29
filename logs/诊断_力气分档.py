import pathlib
# -*- coding: utf-8 -*-
"""同一个抄下来的步态，只改「力气分几档」回放，看档位够不够。"""
import json, pathlib
import numpy as np
import 身体_go2 as 身体
根 = pathlib.Path(__file__).resolve().parent
库 = json.loads((根 / "动作库.json").read_text(encoding="utf-8"))
测试 = ["前进0.4","前进0.8","前进1.2","后退0.4","后退0.8","左移0.4","右移0.4",
        "左转0.75","右转0.75","前进0.8左移0.4","前进0.8左转0.75","前进1.2右转0.75"]
秒 = 5.0
def 跑(名, 档):
    拍 = np.array(库[名]["拍"], dtype=float)
    拍 = np.rint(拍 * 档) / 档 if 档 else 拍        # 力气只能分 档 挡
    身 = 身体.身体(); 身.摆成站姿(); 身.走(0.5, None)
    起 = 身.数据.qpos[:2].copy(); n = len(拍); i = 0
    for i in range(int(秒 / 身体.每帧秒)):
        身.步进(拍[i % n])
        if 身.机身歪了() > 75: break
    p = 身.数据.qpos
    return i * 身体.每帧秒, float(np.hypot(p[0]-起[0], p[1]-起[1]))
print("%-16s" % "动作", "".join("%14s" % ("%d 档" % d) for d in (10, 20, 25, 50, 0)))
for 名 in 测试:
    行 = ["%5.1f秒/%4.2f米" % 跑(名, d) for d in (10, 20, 25, 50, 0)]
    print("%-16s" % 名, "".join("%14s" % x for x in 行))
print("（0 档 = 不分档，直接用连续值；每格 = 撑住秒 / 走了米）")