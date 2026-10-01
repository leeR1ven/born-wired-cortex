# -*- coding: utf-8 -*-
"""measure 里顺手把狗自己的路线也记下来（好核对「球是不是始终在远离狗」）。"""
import io
p = r"F:\born-wired-cortex\engine_v2\tools\chase_red_ball.py"
s = io.open(p, encoding="utf-8", newline="").read()
old = "    ball_track = []\n"
assert old in s
s = s.replace(old, "    ball_track = []\n    dog_track = []\n", 1)
old2 = "        if flee > 0.:\n            ball_track.append([float(ball[0]), float(ball[1])])\n"
new2 = ("        if flee > 0.:\n"
        "            ball_track.append([float(ball[0]), float(ball[1])])\n"
        "            dog_track.append([float(body.data.xpos[base][0]),\n"
        "                              float(body.data.xpos[base][1])])\n")
assert old2 in s
s = s.replace(old2, new2, 1)
old3 = "                headings=headings, ranges=ranges, ball_track=ball_track)"
new3 = "                headings=headings, ranges=ranges, ball_track=ball_track,\n                dog_track=dog_track)"
assert old3 in s
s = s.replace(old3, new3, 1)
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("patched dog_track")