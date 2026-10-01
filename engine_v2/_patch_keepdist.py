# -*- coding: utf-8 -*-
"""球跟狗始终保持一段距离：不会太近，也不会太远（用户 2026-10-01）。

朝向不变（始终背离狗 + 随机弯度），改的是**速度**：
  离狗比 KEEP_NEAR 还近 -> 全力跑（放大 KEEP_FAST 倍）
  离狗比 KEEP_FAR 还远 -> 慢下来等它（只留 KEEP_SLOW 倍）
中间线性过渡。这样球总在狗前面 2~4.5 米晃，狗一直能看见它、也一直追得上。
"""
import io

p = r"F:\born-wired-cortex\engine_v2\tools\chase_red_ball.py"
s = io.open(p, encoding="utf-8", newline="").read()

old_cfg = "TURN_RATE = 8.0         # 球每秒最多把朝向扳多少（弧度/秒）\n"
new_cfg = ("TURN_RATE = 8.0         # 球每秒最多把朝向扳多少（弧度/秒）\n"
           "KEEP_NEAR = 2.0         # 球离狗比这还近：全力跑\n"
           "KEEP_FAR = 4.5          # 球离狗比这还远：慢下来等它（别跑没影了）\n"
           "KEEP_SLOW = .15         # 最慢时留多少速度（别真停成一块石头）\n"
           "KEEP_FAST = 1.3         # 最近时速度放大多少倍\n")
assert old_cfg in s, "no TURN_RATE"
s = s.replace(old_cfg, new_cfg, 1)

old_cmt = """            # 地图是无限大的（用户 2026-10-01）：球不用被圈在场地里，也没有「绕一圈」
            # 这回事。它只管带着这个随机弯度一路背离狗跑 —— 狗不追，它就一路跑远。
"""
new_cmt = """            # 地图是无限大的（用户 2026-10-01）：球不用被圈在场地里，也没有「绕一圈」
            # 这回事。它只管带着这个随机弯度背离狗跑。
            # 但**始终跟狗保持一段距离**（用户 2026-10-01）：太近了拼命跑、太远了慢下来
            # 等它 —— 这样球总在狗前面几米晃，狗一直看得见、也一直追得上。
"""
assert old_cmt in s, "no comment"
s = s.replace(old_cmt, new_cmt, 1)

old_move = """                ball = ball + np.array([math.cos(escape_yaw), math.sin(escape_yaw), 0.])*flee*DT
"""
new_move = """                speed = flee*float(np.clip((KEEP_FAR - length)/(KEEP_FAR - KEEP_NEAR),
                                           KEEP_SLOW, KEEP_FAST))
                ball = ball + np.array([math.cos(escape_yaw), math.sin(escape_yaw), 0.])*speed*DT
"""
assert old_move in s, "no move line"
s = s.replace(old_move, new_move, 1)
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("patched: keep-distance speed rule")