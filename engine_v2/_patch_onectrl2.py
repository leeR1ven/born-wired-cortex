# -*- coding: utf-8 -*-
"""chase_task.exam：「球搬走」对照只跑一次（用户 2026-09-30：所有去掉球的状态都一样，
只需要测出一边去掉球后的移动就行）。省下每一只候选 20 秒模拟，迭代更快。"""
import io

p = r"F:\born-wired-cortex\engine_v2\tools\chase_task.py"
s = io.open(p, encoding="utf-8", newline="").read()

old = (
    "    chase_rows, no_ball = [], None\n"
    "    for bearing in CHASE_BEARINGS:\n"
    "        body, brain, eyes, geom = C.build(genome, chase, seed=seed, spec=spec)\n"
    "        saved = C.snapshot(brain)\n"
    "        if aware:\n"
    "            C.restore(brain, saved)\n"
    "            free = C.measure(body, brain, eyes, geom, chase_seconds, CHASE_BEARINGS[0],\n"
    "                             CHASE_DISTANCE, seed=seed, ball=False)\n"
    "            no_ball = np.asarray(free[\"end_xy\"], dtype=float)\n"
    "        C.restore(brain, saved)\n"
    "        got = C.measure(body, brain, eyes, geom, chase_seconds, bearing, CHASE_DISTANCE,\n"
    "                        seed=seed, flee=flee, escape=CHASE_ESCAPE, curve=curve, ball=True)\n"
)
new = (
    "    chase_rows, no_ball = [], None\n"
    "    if aware:\n"
    "        # 球搬走之后，狗面对的世界每个位置都一样，所以这一趟只跑一次。\n"
    "        body, brain, eyes, geom = C.build(genome, chase, seed=seed, spec=spec)\n"
    "        free = C.measure(body, brain, eyes, geom, chase_seconds, CHASE_BEARINGS[0],\n"
    "                         CHASE_DISTANCE, seed=seed, ball=False)\n"
    "        no_ball = np.asarray(free[\"end_xy\"], dtype=float)\n"
    "        eyes.close()\n"
    "    for bearing in CHASE_BEARINGS:\n"
    "        body, brain, eyes, geom = C.build(genome, chase, seed=seed, spec=spec)\n"
    "        got = C.measure(body, brain, eyes, geom, chase_seconds, bearing, CHASE_DISTANCE,\n"
    "                        seed=seed, flee=flee, escape=CHASE_ESCAPE, curve=curve, ball=True)\n"
)
assert old in s, "no exam chase block"
s = s.replace(old, new, 1)

old2 = "三、对照：把球搬走，同样跑一趟 20 秒（球搬走之后哪个位置的世界都一样，所以只跑一趟）。\n"
new2 = ("三、对照：把球搬走，同样跑一趟 20 秒。球搬走之后哪个位置的世界都一样，所以只跑一趟就够，\n"
        "        不用每个位置各测一遍（用户 2026-09-30）。\n")
assert old2 in s, "no doc line"
s = s.replace(old2, new2, 1)

io.open(p, "w", encoding="utf-8", newline="").write(s)
print("patched exam control trip")