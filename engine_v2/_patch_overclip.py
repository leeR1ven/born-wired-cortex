# -*- coding: utf-8 -*-
"""削「朝外」分量时多削一点（超过 100%），别刚好削到零。

刚好削到零时方向消失（norm=0 就被跳过了），球就一直朝外飘 —— 刚才它飘到 8.3 米
就是这么来的。多削 20% 之后，贴边时它会带着一点点「往里」的分量，于是拐着弯回到
场地里；因为它往里拐多少是随机弯度说了算，所以路线是随机曲线，不是一条固定的圆。
"""
import io

p = r"F:\born-wired-cortex\engine_v2\tools\chase_red_ball.py"
s = io.open(p, encoding="utf-8", newline="").read()
old = """                        t = min(1., (radius - FIELD_SOFT)/(FIELD_RADIUS - FIELD_SOFT))
                        ux, uy = math.cos(want), math.sin(want)
                        ox, oy = ball[0]/radius, ball[1]/radius
                        outward = ux*ox + uy*oy
                        if outward > 0.:
                            ux, uy = ux - t*outward*ox, uy - t*outward*oy
                            norm = math.hypot(ux, uy)
                            if norm > 1e-6:
                                want = math.atan2(uy/norm, ux/norm)
"""
new = """                        t = min(1., (radius - FIELD_SOFT)/(FIELD_RADIUS - FIELD_SOFT))*FIELD_MARGIN
                        ux, uy = math.cos(want), math.sin(want)
                        ox, oy = ball[0]/radius, ball[1]/radius
                        outward = ux*ox + uy*oy
                        if outward > 0.:
                            # 多削 FIELD_MARGIN 倍：贴边时带一点点朝里的分量，拐着弯回场地。
                            ux, uy = ux - t*outward*ox, uy - t*outward*oy
                            want = math.atan2(uy, ux)
"""
assert old in s, "no clip block"
s = s.replace(old, new, 1)
old_cfg = "FIELD_RADIUS = 3.4      # 到这个半径，「朝外」的分量削得一点不剩\n"
new_cfg = ("FIELD_RADIUS = 3.4      # 到这个半径，按倍数把「朝外」的分量削过头，让它往回拐\n"
           "FIELD_MARGIN = 1.2      # 削过头的倍数（1.0 = 刚好削平，会卡住往外飘）\n")
assert old_cfg in s, "no FIELD_RADIUS"
s = s.replace(old_cfg, new_cfg, 1)
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("patched: over-clip")