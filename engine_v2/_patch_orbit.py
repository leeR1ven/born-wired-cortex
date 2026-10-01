# -*- coding: utf-8 -*-
"""贴边绕圈的方向要定死，不能每一步重挑。

原来每一步都按「离当前朝向更近的那条切线」重挑方向，球和狗半径差不多时这个选择会
来回翻 180 度，限速之下的实际朝向就变成一个「朝外」的合成方向 —— 球于是往外飘。
现在：第一次贴到场地边时按当前朝向挑一个绕圈方向，之后一路绕着走；回到场地里才解除。
"""
import io

p = r"F:\born-wired-cortex\engine_v2\tools\chase_red_ball.py"
s = io.open(p, encoding="utf-8", newline="").read()

old_cfg = """FIELD_RADIUS = 3.1      # 场地墙在 3.5 米；球跑到这个半径就贴着场地绕圈
"""
new_cfg = """FIELD_RADIUS = 3.0      # 场地墙在 3.5 米；球跑到这个半径就贴着场地绕圈
FIELD_RELEASE = 2.6     # 回到这个半径以内才解除「绕圈」
"""
assert old_cfg in s, "no FIELD_RADIUS"
s = s.replace(old_cfg, new_cfg, 1)

old_state = "    wander = 0.0\n    escape_yaw = None\n    ball_track = []\n"
new_state = "    wander = 0.0\n    escape_yaw = None\n    orbit = 0.          # 贴边绕圈的方向：+1 逆时针、-1 顺时针、0 还没定\n    ball_track = []\n"
assert old_state in s, "no state block"
s = s.replace(old_state, new_state, 1)

old_turn = """                    radius = float(np.hypot(ball[0], ball[1]))
                    if radius > FIELD_RADIUS:
                        # 贴场地边：改走切线（绕着场地跑），哪边转得少就走哪边。
                        tangent = math.atan2(ball[1], ball[0]) + math.pi/2.
                        gap = math.atan2(math.sin(tangent - want), math.cos(tangent - want))
                        want = tangent if abs(gap) <= math.pi/2. else tangent - math.pi
"""
new_turn = """                    radius = float(np.hypot(ball[0], ball[1]))
                    if radius <= FIELD_RELEASE:
                        orbit = 0.
                    elif radius > FIELD_RADIUS:
                        # 贴场地边：改走切线，绕着场地跑。绕圈方向**只在刚贴边时挑一次**
                        # （挑离现在这个朝向更近的那条切线），之后一路绕下去 ——
                        # 每一步都重挑会让方向来回翻，球就往外飘了。
                        base = math.atan2(ball[1], ball[0])
                        if orbit == 0.:
                            ref = want if escape_yaw is None else escape_yaw
                            gap = math.atan2(math.sin(ref - (base + math.pi/2.)),
                                             math.cos(ref - (base + math.pi/2.)))
                            orbit = -1. if abs(gap) > math.pi/2. else 1.
                        want = base + orbit*math.pi/2.
"""
assert old_turn in s, "no turn block"
s = s.replace(old_turn, new_turn, 1)
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("patched orbit direction is sticky")