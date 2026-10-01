# -*- coding: utf-8 -*-
"""撤掉「三趟共用一只狗」：它省下的建狗开销可以忽略，却把追球读数改掉了。"""
import io, sys
P = r"F:\born-wired-cortex\engine_v2\tools\chase_task.py"
t = io.open(P, encoding="utf-8").read()
def rep(old, new, why):
    global t
    if t.count(old) != 1:
        print("!! %s : \u51fa\u73b0 %d \u6b21" % (why, t.count(old))); sys.exit(1)
    t = t.replace(old, new); print("ok  %s" % why)

rep(
'''    # 没球对照和两趟追球共用一只狗。measure() 每趟开场都会把狗放回起始姿势，只要两趟
    # 之间把脑子 restore 回刚建出来那一刻，三趟就是从同一个起点各跑各的，跟各建一只
    # 完全一样。省掉两次建狗（用户 2026-10-01：尽量压榨性能）。
    # 球搬走之后，狗面对的世界每个位置都一样，所以对照只跑一次就够（用户 2026-09-30）。
    chase_rows, no_ball = [], None
    body, brain, eyes, geom = C.build(genome, chase, seed=seed, spec=spec)
    saved = C.snapshot(brain)
    if aware:
        C.restore(brain, saved)
        free = C.measure(body, brain, eyes, geom, chase_seconds, CHASE_BEARINGS[0],
                         CHASE_DISTANCE, seed=seed, ball=False)
        no_ball = np.asarray(free["end_xy"], dtype=float)
    for bearing in CHASE_BEARINGS:
        C.restore(brain, saved)
        got = C.measure(body, brain, eyes, geom, chase_seconds, bearing, CHASE_DISTANCE,
                        seed=seed, flee=flee, escape=CHASE_ESCAPE, curve=curve, ball=True)''',
'''    # 2026-10-01 试过让这三趟共用一只狗（省两次建狗），结果追球读数全变了 —— 建狗这点
    # 开销本来就可以忽略，不值得为它改行为。所以还是每趟各建一只。
    # 球搬走之后，狗面对的世界每个位置都一样，所以对照只跑一次就够（用户 2026-09-30）。
    chase_rows, no_ball = [], None
    if aware:
        body, brain, eyes, geom = C.build(genome, chase, seed=seed, spec=spec)
        free = C.measure(body, brain, eyes, geom, chase_seconds, CHASE_BEARINGS[0],
                         CHASE_DISTANCE, seed=seed, ball=False)
        no_ball = np.asarray(free["end_xy"], dtype=float)
        eyes.close()
    for bearing in CHASE_BEARINGS:
        body, brain, eyes, geom = C.build(genome, chase, seed=seed, spec=spec)
        got = C.measure(body, brain, eyes, geom, chase_seconds, bearing, CHASE_DISTANCE,
                        seed=seed, flee=flee, escape=CHASE_ESCAPE, curve=curve, ball=True)''',
"\u64a4\u56de\u5171\u7528\u4e00\u53ea\u72d7")

rep(
'''                     "  追住了" if row["kept_in_front"] else ""), flush=True)
    eyes.close()
    rows += chase_rows''',
'''                     "  追住了" if row["kept_in_front"] else ""), flush=True)
        eyes.close()
    rows += chase_rows''',
"\u773c\u775b\u6536\u5c3e\u653e\u56de\u5faa\u73af\u5185")

io.open(P, "w", encoding="utf-8", newline="\n").write(t)
print("\u64a4\u56de\u5b8c\u6bd5")