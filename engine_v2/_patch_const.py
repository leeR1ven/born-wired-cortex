# -*- coding: utf-8 -*-
"""球改回**固定速度**（用户 2026-10-01）：保持距离那套会让狗永远追不上。

球的规则现在就三条：
  1) 每一拍都背离狗（叠一个缓慢摆动的小偏角，所以路线是随机曲线）
  2) 速度恒定 = flee（默认 0.6 米/秒，比狗慢，狗追得上）
  3) 地图无限大，没有任何场边规则
以后想加难度就把 --flee 往上调（等迭代多了再慢慢提).
"""
import io

p = r"F:\born-wired-cortex\engine_v2\tools\chase_red_ball.py"
s = io.open(p, encoding="utf-8", newline="").read()

old_cfg = """KEEP_NEAR = 2.0         # 球离狗比这还近：全力跑
KEEP_FAR = 4.5          # 球离狗比这还远：慢下来等它（别跑没影了）
KEEP_SLOW = .0          # 最慢就是停在那儿等它（别跑没影）
KEEP_FAST = 2.0         # 最近时速度放大多少倍（0.6*2 = 1.2 米/秒，跟狗差不多快）
"""
assert old_cfg in s, "no keep cfg"
s = s.replace(old_cfg, "", 1)

old_cmt = """            # 但**始终跟狗保持一段距离**（用户 2026-10-01）：太近了拼命跑、太远了慢下来
            # 等它 —— 这样球总在狗前面几米晃，狗一直看得见、也一直追得上。
"""
new_cmt = """            # 速度**恒定**（用户 2026-10-01）：一度试过「近了快跑、远了等它」，但那样狗
            # 永远追不上球。所以现在就是匀速逃，狗只要真会追就追得上。等迭代多了、
            # 狗确实会追了，再把 --flee 往上调给球提速。
"""
assert old_cmt in s, "no comment"
s = s.replace(old_cmt, new_cmt, 1)

old_move = """                speed = flee*float(np.clip((KEEP_FAR - length)/(KEEP_FAR - KEEP_NEAR),
                                           KEEP_SLOW, KEEP_FAST))
                ball = ball + np.array([math.cos(escape_yaw), math.sin(escape_yaw), 0.])*speed*DT
"""
new_move = """                ball = ball + np.array([math.cos(escape_yaw), math.sin(escape_yaw), 0.])*flee*DT
"""
assert old_move in s, "no speed line"
s = s.replace(old_move, new_move, 1)
io.open(p, "w", encoding="utf-8", newline="").write(s)

q = r"F:\born-wired-cortex\engine_v2\tools\chase_task.py"
s = io.open(q, encoding="utf-8", newline="").read()
old = "CHASE_NEAR = 2.8              # 跑完落后不到这么近 = 这一趟「追住了」（球会保持距离，追住 ≈ 贴在 2~3 米内）"
new = "CHASE_NEAR = 1.5              # 跑完落后不到这么近 = 这一趟「追住了」（球匀速逃，追住了就贴在 1 米多）"
assert old in s, "no CHASE_NEAR"
s = s.replace(old, new, 1)
io.open(q, "w", encoding="utf-8", newline="").write(s)

r = r"F:\born-wired-cortex\engine_v2\tools\evolve_chase.py"
s = io.open(r, encoding="utf-8", newline="").read()
old = "FOLLOW_SETTLED = 2.8     # 跑完落后不到这么近 = 这一趟「追住了」（跟 chase_task 一个口径）"
new = "FOLLOW_SETTLED = 1.5     # 跑完落后不到这么近 = 这一趟「追住了」（跟 chase_task 一个口径）"
assert old in s, "no FOLLOW_SETTLED"
s = s.replace(old, new, 1)
io.open(r, "w", encoding="utf-8", newline="").write(s)
print("patched: constant ball speed")