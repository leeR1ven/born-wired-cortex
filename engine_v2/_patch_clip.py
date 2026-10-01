# -*- coding: utf-8 -*-
"""球的贴边处理：只削掉「朝外」那一部分，别把方向整个换成切线。

换成切线就成了一条固定的圆（用户 2026-10-01：不要刚好绕一圈）。
现在：越靠边，把「朝外」的分量削得越多（最多全削掉），剩下的方向还是由「背离狗」
说了算 —— 于是它贴着场地边拐着弯走，每一段的弯都随狗的位置和随机弯度变化。
"""
import io

p = r"F:\born-wired-cortex\engine_v2\tools\chase_red_ball.py"
s = io.open(p, encoding="utf-8", newline="").read()

old_cfg = """FIELD_SOFT = 2.2        # 场地墙在 3.5 米。球跑到这个半径就开始往场中心拐
FIELD_RADIUS = 3.2      # 到这个半径，拐回来的力加到最大
"""
new_cfg = """FIELD_SOFT = 2.4        # 场地墙在 3.5 米。球跑到这个半径就开始削「朝外」的分量
FIELD_RADIUS = 3.4      # 到这个半径，「朝外」的分量削得一点不剩
"""
assert old_cfg in s, "no config"
s = s.replace(old_cfg, new_cfg, 1)

old_turn = """                    radius = float(np.hypot(ball[0], ball[1]))
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
new_turn = """                    radius = float(np.hypot(ball[0], ball[1]))
                    if radius > FIELD_SOFT:
                        # 快到场地边了：把朝向里「朝外」的那一部分削掉，越靠边削得越多
                        # （最多全削掉）。剩下的方向还是由「背离狗」说了算 ——
                        # 所以它是贴着场地边拐着弯走，不是沿着一条固定的圆跑。
                        t = min(1., (radius - FIELD_SOFT)/(FIELD_RADIUS - FIELD_SOFT))
                        ux, uy = math.cos(want), math.sin(want)
                        ox, oy = ball[0]/radius, ball[1]/radius
                        outward = ux*ox + uy*oy
                        if outward > 0.:
                            ux, uy = ux - t*outward*ox, uy - t*outward*oy
                            norm = math.hypot(ux, uy)
                            if norm > 1e-6:
                                want = math.atan2(uy/norm, ux/norm)
"""
assert old_turn in s, "no turn block"
s = s.replace(old_turn, new_turn, 1)
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("patched: clip outward component only")