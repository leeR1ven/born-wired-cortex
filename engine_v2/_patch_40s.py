# -*- coding: utf-8 -*-
"""转弯限速调高（消掉慢慢往外飘）+ 每趟 40 秒（够绕场地一圈）。

贴边绕圈时朝向一直在转，转弯限速太低就永远差一点点角度，那一点点的朝外分量累积起来
就是「慢慢往外飘」。调高限速之后绕圈半径就稳住了。
"""
import io

p = r"F:\born-wired-cortex\engine_v2\tools\chase_red_ball.py"
s = io.open(p, encoding="utf-8", newline="").read()
old = "TURN_RATE = 3.0         # 球每秒最多把朝向扳多少（弧度/秒），保证路线是弧线不是折线"
new = "TURN_RATE = 8.0         # 球每秒最多把朝向扳多少（弧度/秒）。太低的话，贴边绕圈时\n                        # 朝向永远差一点点，那点朝外的分量累积起来就是慢慢往外飘"
assert old in s, "no TURN_RATE"
s = s.replace(old, new, 1)
io.open(p, "w", encoding="utf-8", newline="").write(s)

q = r"F:\born-wired-cortex\engine_v2\tools\chase_task.py"
s = io.open(q, encoding="utf-8", newline="").read()
old = "CHASE_SECONDS = 30.           # 每趟考多少秒（球 0.6 米/秒绕场地一圈要 30 秒出头）"
new = "CHASE_SECONDS = 40.           # 每趟考多少秒（球 0.6 米/秒绕场地一圈要 35 秒左右）"
assert old in s, "no CHASE_SECONDS"
s = s.replace(old, new, 1)
s = s.replace("每趟 30 秒", "每趟 40 秒", 1)
io.open(q, "w", encoding="utf-8", newline="").write(s)

r = r"F:\born-wired-cortex\engine_v2\tools\evolve_chase.py"
s = io.open(r, encoding="utf-8", newline="").read()
s = s.replace('default=30., help="追着跑的球那几趟每趟考多少秒"',
              'default=40., help="追着跑的球那几趟每趟考多少秒"', 1)
s = s.replace('job.get("chase_seconds", 30.)', 'job.get("chase_seconds", 40.)', 1)
io.open(r, "w", encoding="utf-8", newline="").write(s)
print("patched TURN_RATE=8, CHASE_SECONDS=40")