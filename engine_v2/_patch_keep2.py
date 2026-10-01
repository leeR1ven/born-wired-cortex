# -*- coding: utf-8 -*-
"""把「太远就慢下来」改成「太远就停下来等」，并把速度上限提到跟狗差不多。

用户 2026-10-01：球始终跟狗保持一定距离，不会太近也不会太远。
  离狗 <= 2.0 米 -> 速度拉到 2 倍（0.6*2 = 1.2 米/秒，跟狗差不多快，狗压不到跟前）
  离狗 >= 4.5 米 -> 速度降到 0（狗不过来它就在那儿等着，不会跑没影）
"""
import io

p = r"F:\born-wired-cortex\engine_v2\tools\chase_red_ball.py"
s = io.open(p, encoding="utf-8", newline="").read()
old = """KEEP_SLOW = .15         # 最慢时留多少速度（别真停成一块石头）
KEEP_FAST = 1.3         # 最近时速度放大多少倍
"""
new = """KEEP_SLOW = .0          # 最慢就是停在那儿等它（别跑没影）
KEEP_FAST = 2.0         # 最近时速度放大多少倍（0.6*2 = 1.2 米/秒，跟狗差不多快）
"""
assert old in s, "no keep cfg"
s = s.replace(old, new, 1)
io.open(p, "w", encoding="utf-8", newline="").write(s)

q = r"F:\born-wired-cortex\engine_v2\tools\chase_task.py"
s = io.open(q, encoding="utf-8", newline="").read()
old = "CHASE_NEAR = 1.0              # 跑完落后不到这么近 = 这一趟「追住了」"
new = "CHASE_NEAR = 2.8              # 跑完落后不到这么近 = 这一趟「追住了」（球会保持距离，追住 ≈ 贴在 2~3 米内）"
assert old in s, "no CHASE_NEAR"
s = s.replace(old, new, 1)
io.open(q, "w", encoding="utf-8", newline="").write(s)

r = r"F:\born-wired-cortex\engine_v2\tools\evolve_chase.py"
s = io.open(r, encoding="utf-8", newline="").read()
old = "FOLLOW_SETTLED = 1.2     # 跑完落后不到这么近 = 这一趟「追住了」"
new = "FOLLOW_SETTLED = 2.8     # 跑完落后不到这么近 = 这一趟「追住了」（跟 chase_task 一个口径）"
assert old in s, "no FOLLOW_SETTLED"
s = s.replace(old, new, 1)
s = s.replace("and float(item.get(\"settled\", 99.)) <= FOLLOW_SETTLED]",
              "and float(item.get(\"settled\", 99.)) <= FOLLOW_SETTLED]", 1)
io.open(r, "w", encoding="utf-8", newline="").write(s)
print("patched: stop when far, catch bar 2.8 m")