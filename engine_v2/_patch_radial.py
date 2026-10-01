# -*- coding: utf-8 -*-
"""修变量名撞车：贴边那一段我把球的角度起名 base，把外面「机身 body 的 id」盖掉了。"""
import io

p = r"F:\born-wired-cortex\engine_v2\tools\chase_red_ball.py"
s = io.open(p, encoding="utf-8", newline="").read()
old = """                        base = math.atan2(ball[1], ball[0])
                        if orbit == 0.:
                            ref = want if escape_yaw is None else escape_yaw
                            gap = math.atan2(math.sin(ref - (base + math.pi/2.)),
                                             math.cos(ref - (base + math.pi/2.)))
                            orbit = -1. if abs(gap) > math.pi/2. else 1.
                        want = base + orbit*math.pi/2.
"""
new = """                        radial = math.atan2(ball[1], ball[0])
                        if orbit == 0.:
                            ref = want if escape_yaw is None else escape_yaw
                            gap = math.atan2(math.sin(ref - (radial + math.pi/2.)),
                                             math.cos(ref - (radial + math.pi/2.)))
                            orbit = -1. if abs(gap) > math.pi/2. else 1.
                        want = radial + orbit*math.pi/2.
"""
assert old in s, "no orbit block"
s = s.replace(old, new, 1)
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("fixed: radial, not base")