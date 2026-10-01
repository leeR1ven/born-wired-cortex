# -*- coding: utf-8 -*-
"""去掉场边那套：地图无限大，球只管**背离狗 + 随机曲线**地逃（用户 2026-10-01）。

上一版快到场边会把「朝外」的分量削掉，于是球被圈在场地里绕。用户说地图搞成无限大就行，
球没必要一直在圈子里跑。所以整段贴边逻辑删掉：朝向 = 「背离狗」+ 缓慢摆动的小偏角，
永远带着背离狗的分量，路线是随机曲线，狗不追它就一路跑远。
"""
import io

p = r"F:\born-wired-cortex\engine_v2\tools\chase_red_ball.py"
s = io.open(p, encoding="utf-8", newline="").read()

old_cfg = """FIELD_SOFT = 2.4        # 场地墙在 3.5 米。球跑到这个半径就开始削「朝外」的分量
FIELD_RADIUS = 3.4      # 到这个半径，按倍数把「朝外」的分量削过头，让它往回拐
FIELD_MARGIN = 1.2      # 削过头的倍数（1.0 = 刚好削平，会卡住往外飘）
"""
assert old_cfg in s, "no field cfg"
s = s.replace(old_cfg, "", 1)

old_cmt = """            # 快跑到场边（离场中心 FIELD_SOFT）时，把朝向平滑地往场中心拧一点，
            # 越靠边拧得越多 —— 于是它拐着弯飘回场地里。每一段弯度都不一样，
            # 走出来的是一条随机曲线，不是一条固定的圆。
"""
new_cmt = """            # 地图是无限大的（用户 2026-10-01）：球不用被圈在场地里，也没有「绕一圈」
            # 这回事。它只管带着这个随机弯度一路背离狗跑 —— 狗不追，它就一路跑远。
"""
assert old_cmt in s, "no comment"
s = s.replace(old_cmt, new_cmt, 1)

old_turn = """                    radius = float(np.hypot(ball[0], ball[1]))
                    if radius > FIELD_SOFT:
                        # 快到场地边了：把朝向里「朝外」的那一部分削掉，越靠边削得越多
                        # （最多全削掉）。剩下的方向还是由「背离狗」说了算 ——
                        # 所以它是贴着场地边拐着弯走，不是沿着一条固定的圆跑。
                        t = min(1., (radius - FIELD_SOFT)/(FIELD_RADIUS - FIELD_SOFT))*FIELD_MARGIN
                        ux, uy = math.cos(want), math.sin(want)
                        ox, oy = ball[0]/radius, ball[1]/radius
                        outward = ux*ox + uy*oy
                        if outward > 0.:
                            # 多削 FIELD_MARGIN 倍：贴边时带一点点朝里的分量，拐着弯回场地。
                            ux, uy = ux - t*outward*ox, uy - t*outward*oy
                            want = math.atan2(uy, ux)
"""
assert old_turn in s, "no turn block"
s = s.replace(old_turn, "", 1)
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("patched: no boundary, infinite field")