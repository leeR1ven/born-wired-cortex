# -*- coding: utf-8 -*-
"""查_站姿和走路重不重.py —— 纯算，不建网。
为什么要查：姿势互斥那 12 条规则用「运动:站姿」当"我在站着"的信号。
如果走路的每一拍也把「运动:站姿」那批肌肉细胞点亮，那这批细胞就不是"站着"的意思，
用它当源去压别人，就会把走路一起压掉。
"""
from __future__ import annotations
import io, json
import numpy as np
import 运动输出区_motor_output as 运动

库 = json.loads(io.open("动作库.json", encoding="utf-8").read())
print("动作库里的动作数：", len(库))

站姿 = np.nonzero(运动.发力转激活(np.asarray(库["站姿"]["拍"][0], dtype=float)))[0]
print("运动:站姿 = %d 个肌肉细胞" % 站姿.size)

for 名 in ["前进1.2", "原地踏步", "坐下", "趴下去", "站起来", "坐", "蹲", "趴姿"]:
    if 名 not in 库:
        continue
    拍 = 库[名]["拍"]
    并 = np.zeros(运动.运动区宽度, dtype=bool)
    for f in 拍:
        并 |= 运动.发力转激活(np.asarray(f, dtype=float))
    交 = np.intersect1d(np.nonzero(并)[0], 站姿)
    print("  %-6s %2d拍  整段用过 %3d 个细胞；其中和「站姿」重合 %3d 个（%5.1f%%）；"
          "第一拍里重合 %d / %d"
          % (名, len(拍), int(并.sum()), 交.size, 100.0 * 交.size / max(站姿.size, 1),
             int(np.intersect1d(np.nonzero(运动.发力转激活(np.asarray(拍[0], dtype=float)))[0], 站姿).size),
             int(运动.发力转激活(np.asarray(拍[0], dtype=float)).sum())))

print()
print("=== 身体感觉区里有哪些【姿势】名字（可以当真在站着的信号）===")
for 文件, 模块名 in [("本体感觉区_proprioception.py", "本体感觉"),
                    ("平衡感觉区_balance.py", "平衡感觉"),
                    ("机身状态区_bodystate.py", "机身状态")]:
    import importlib
    m = importlib.import_module(文件.replace(".py", ""))
    print(" ", 模块名, list(m.情况.keys()))
