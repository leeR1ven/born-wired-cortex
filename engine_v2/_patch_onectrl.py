# -*- coding: utf-8 -*-
"""chase_task.exam：「球搬走」对照只跑一次（用户 2026-09-30：所有去掉球的状态都一样，
只需要测出一边去掉球后的移动就行）。省下每一只候选 20 秒模拟，迭代更快。"""
import io

p = r"F:\born-wired-cortex\engine_v2\tools\chase_task.py"
s = io.open(p, encoding="utf-8", newline="").read()

old = '''    chase_rows, no_ball = [], None
    for bearing in CHASE_BEARINGS:
        body, brain, eyes, geom = C.build(genome, chase, seed=seed, spec=spec)
        saved = C.snapshot(brain)
        if aware:
            C.restore(brain, saved)
            free = C.measure(body, brain, eyes, geom, chase_seconds, CHASE_BEARINGS[0],
                             CHASE_DISTANCE, seed=seed, ball=False)
            no_ball = np.asarray(free["end_xy"], dtype=float)
        C.restore(brain, saved)
        got = C.measure(body, brain, eyes, geom, chase_seconds, bearing, CHASE_DISTANCE,
                        seed=seed, flee=flee, escape=CHASE_ESCAPE, curve=curve, ball=True)
'''
new = '''    chase_rows, no_ball = [], None
    if aware:
        # 「球搬走」对照：球被挪到天边之后，狗面对的世界每个位置都一样，所以只跑一趟。
        body, brain, eyes, geom = C.build(genome, chase, seed=seed, spec=spec)
        free = C.measure(body, brain, eyes, geom, chase_seconds, CHASE_BEARINGS[0],
                         CHASE_DISTANCE, seed=seed, ball=False)
        no_ball = np.asarray(free["end_xy"], dtype=float)
        eyes.close()
    for bearing in CHASE_BEARINGS:
        body, brain, eyes, geom = C.build(genome, chase, seed=seed, spec=spec)
        got = C.measure(body, brain, eyes, geom, chase_seconds, bearing, CHASE_DISTANCE,
                        seed=seed, flee=flee, escape=CHASE_ESCAPE, curve=curve, ball=True)
'''
assert old in s, "没找到 exam 里追球那一段"
s = s.replace(old, new, 1)

old2 = '''    三、对照：把球搬走，同样跑一趟 20 秒（球搬走之后哪个位置的世界都一样，所以只跑一趟）。
'''
new2 = '''    三、对照：把球搬走，同样跑一趟 20 秒（球搬走之后哪个位置的世界都一样，所以只跑一趟；
        不用每个位置都各测一遍 —— 用户 2026-09-30）。
'''
assert old2 in s, "没找到文档里对照那一段"
s = s.replace(old2, new2, 1)
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("改好 chase_task.exam：对照只跑一趟")