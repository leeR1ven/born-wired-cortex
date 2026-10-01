# -*- coding: utf-8 -*-
"""球的走法改成「随机曲线 + 主动远离狗」（用户 2026-10-01）。

上一版贴到场地边之后绕着一条固定的圆跑，路线太规整了。现在：快到场边就把朝向
**平滑地**往场中心拧一点（越靠边拧得越多），所以它是拐着弯飘回场地里的，
每一段弧度都随弯度的随机游走而不一样 —— 随机曲线，且始终主动背离狗。
"""
import io

p = r"F:\born-wired-cortex\engine_v2\tools\chase_red_ball.py"
s = io.open(p, encoding="utf-8", newline="").read()

old_cfg = """WANDER_TAU = 1.2        # 弯的「记忆时间」（秒）：越小拐得越急
WANDER_SIGMA = .9       # 弯得有多厉害（弧度/√秒）
WANDER_MAX = .6         # 最多偏离「背对狗」多少弧度（约 34 度）—— 再大就不像「一直逃」了
TURN_RATE = 8.0         # 球每秒最多把朝向扳多少（弧度/秒）。太低的话，贴边绕圈时
                        # 朝向永远差一点点，那点朝外的分量累积起来就是慢慢往外飘
FIELD_RADIUS = 3.0      # 场地墙在 3.5 米；球跑到这个半径就贴着场地绕圈
FIELD_RELEASE = 2.6     # 回到这个半径以内才解除「绕圈」
"""
new_cfg = """WANDER_TAU = 1.0        # 弯的「记忆时间」（秒）：越小拐得越急
WANDER_SIGMA = 1.3      # 弯得有多厉害（弧度/√秒）
WANDER_MAX = 1.15       # 最多偏离「背对狗」多少弧度（约 66 度）—— 再大就不像「一直逃」了
TURN_RATE = 8.0         # 球每秒最多把朝向扳多少（弧度/秒）
FIELD_SOFT = 2.2        # 场地墙在 3.5 米。球跑到这个半径就开始往场中心拐
FIELD_RADIUS = 3.2      # 到这个半径，拐回来的力加到最大
"""
assert old_cfg in s, "no config"
s = s.replace(old_cfg, new_cfg, 1)

old_comment = """            # 快跑到场边（离场中心 FIELD_RADIUS）时，把朝向拧成贴边的切线方向，
            # 于是它绕着场地跑圈 —— 狗要追住它就得抄近路，真正看出会不会追。
"""
new_comment = """            # 快跑到场边（离场中心 FIELD_SOFT）时，把朝向平滑地往场中心拧一点，
            # 越靠边拧得越多 —— 于是它拐着弯飘回场地里。每一段弯度都不一样，
            # 走出来的是一条随机曲线，不是一条固定的圆。
"""
assert old_comment in s, "no comment"
s = s.replace(old_comment, new_comment, 1)

old_state = "    orbit = 0.          # 贴边绕圈的方向：+1 逆时针、-1 顺时针、0 还没定\n"
assert old_state in s, "no orbit state"
s = s.replace(old_state, "", 1)

old_turn = """                    radius = float(np.hypot(ball[0], ball[1]))
                    if radius <= FIELD_RELEASE:
                        orbit = 0.
                    elif radius > FIELD_RADIUS:
                        # 贴场地边：改走切线，绕着场地跑。绕圈方向**只在刚贴边时挑一次**
                        # （挑离现在这个朝向更近的那条切线），之后一路绕下去 ——
                        # 每一步都重挑会让方向来回翻，球就往外飘了。
                        radial = math.atan2(ball[1], ball[0])
                        if orbit == 0.:
                            ref = want if escape_yaw is None else escape_yaw
                            gap = math.atan2(math.sin(ref - (radial + math.pi/2.)),
                                             math.cos(ref - (radial + math.pi/2.)))
                            orbit = -1. if abs(gap) > math.pi/2. else 1.
                        want = radial + orbit*math.pi/2.
"""
new_turn = """                    radius = float(np.hypot(ball[0], ball[1]))
                    if radius > FIELD_SOFT:
                        # 快到场地边了：把「背离狗」这个方向和「朝场中心」那个方向
                        # 按越靠边越大的比重混在一起 —— 它就拐着弯回到场地里。
                        inward = math.atan2(-ball[1], -ball[0])
                        t = min(1., (radius - FIELD_SOFT)/(FIELD_RADIUS - FIELD_SOFT))
                        if t >= 1.:
                            want = inward
                        else:
                            want = math.atan2((1. - t)*math.sin(want) + t*math.sin(inward),
                                              (1. - t)*math.cos(want) + t*math.cos(inward))
"""
assert old_turn in s, "no turn block"
s = s.replace(old_turn, new_turn, 1)
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("patched: random curve + soft boundary")