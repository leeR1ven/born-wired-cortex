# -*- coding: utf-8 -*-
"""修融合的 bug：整数基因不能取平均。

2026-10-01 实测：融合爹把 eye_change_relay_steps 平均成 8.375 步，控制器直接拒收，
300 只里 107 只没建出来（四批同样规模只 14 只是因为这一条）。平均完必须取整。
"""
import io, sys
P = r"F:\born-wired-cortex\engine_v2\tools\evolve_chase.py"
t = io.open(P, encoding="utf-8").read()

old = '''    genome = {name: float(np.mean(values)) for name, values in columns.items() if values}'''
new = '''    genome = {}
    for name, values in columns.items():
        if not values:
            continue
        mean = float(np.mean(values))
        if name in sc.taskbank.INTEGER_GENES:
            # 整数基因不能取平均：8 只平均出来 8.375 步，控制器当场拒收。
            # 2026-10-01 实测，不取整的话 300 只里有 107 只建不出来。
            mean = float(max(1, int(round(mean))))
        genome[name] = mean'''
if t.count(old) != 1:
    print("!! 找到 %d 处" % t.count(old)); sys.exit(1)
io.open(P, "w", encoding="utf-8", newline="\n").write(t.replace(old, new))
print("ok")