# -*- coding: utf-8 -*-
"""把追球那场考试改快：静止球 4 趟 -> 1 趟（随机摆），对照和追球共用一只狗。"""
import io, sys

P = r"F:\born-wired-cortex\engine_v2\tools\chase_task.py"
t = io.open(P, encoding="utf-8").read()
orig = t
def rep(old, new, why):
    global t
    if old not in t:
        print("!! 没找到：%s" % why); sys.exit(1)
    if t.count(old) != 1:
        print("!! 出现 %d 次：%s" % (t.count(old), why)); sys.exit(1)
    t = t.replace(old, new)
    print("ok  %s" % why)

rep(
"""一、静止球（老判分，跟前几批可比）：
    球不动，摆在狗的前方：左 0.5 弧度 / 右 0.5 弧度  x  3.0 米 / 4.0 米，每趟 6 秒。
    主判分是「这一趟离球最近多少米」。""",
"""一、静止球（老判分）：
    球不动，**随机**摆在狗看得见的地方（左右 ±0.55 弧度、3.0~4.0 米，位置从种子里摇），
    只跑 1 趟、6 秒。主判分是「这一趟离球最近多少米」。
    用户 2026-10-01：模型已经真的会追红球了，红球就不用换那么多种摆法各测一遍，随机摆
    一次就够 —— 省下来的时间拿去多迭代几代。""",
"文档：静止球那节")

rep(
"""    upright / moving / travelled / straightness    走路本身
    closest / reached / covered / both             静止球那趟""",
"""    upright / moving / travelled / straightness    走路本身
    closest / reached / covered                    静止球那趟（只 1 趟了）""",
"文档：判分清单 1")

rep(
"""    chase_settled    追着跑的球，跑完平均落后多少米（越小越好，主判分）
    chase_covered    2 趟里追住了几趟（跑完落后 < 1 米）""",
"""    chase_settled    追着跑的球，跑完平均落后多少米（越小越好，主判分）
    chase_covered    2 趟里追住了几趟（跑完落后 < 1 米）
    both             左、右两趟都追住了（用户 2026-09-30：同一只狗两边都要试，
                     不然会漏掉「只认某一边」的模型。改成由追球那两趟来判，
                     因为静止球那趟只剩随机一次了）""",
"文档：判分清单 2")

rep(
"""BEARINGS = (.5, -.5)          # 正 = 狗的左前方；负 = 右前方
DISTANCES = (3.0, 4.0)""",
"""BEARINGS = (-.55, .55)        # 静止球随机摆在这个范围里（正 = 狗的左前方；负 = 右前方）
DISTANCES = (3.0, 4.0)        # 距离也在这个区间里随机取
CENTER_GAP = .15              # 离正前方至少这么远：摆正中间考不出会不会拐弯""",
"常量")

rep(
'''    """一个模型考一场：静止球 4 趟 + 追着跑的球 2 趟 + 没球对照 1 趟。"""
    spec = gaze() if spec is None else spec
    rows = []
    for distance in DISTANCES:
        body, brain, eyes, geom = C.build(genome, chase, seed=seed, spec=spec)
        saved = C.snapshot(brain)
        for bearing in BEARINGS:
            C.restore(brain, saved)
            got = C.measure(body, brain, eyes, geom, seconds, bearing, distance,
                            seed=seed, ball=True)
            row = _row(got, bearing, distance, 0.)
            rows.append(row)
            if loud:
                print("   静止球在 %+.2f 弧度、%.1f 米：最近贴到 %.2f 米%s、位移离球 %4.1f 度、"
                      "走了 %.2f 米"
                      % (bearing, distance, row["closest"],
                         "  到了" if row["reached"] else "", row["toward_deg"],
                         row["travelled_m"]), flush=True)
        eyes.close()
''',
'''    """一个模型考一场：静止球 1 趟（随机摆）+ 追着跑的球 2 趟 + 没球对照 1 趟。"""
    spec = gaze() if spec is None else spec
    rows = []
    # 静止球只跑 1 趟，位置从种子里摇 —— 同一只狗每次考都摆同一个位置，不会这次考左、
    # 下次考右。用户 2026-10-01：模型已经真的会追红球了，红球不用换那么多种摆法各测
    # 一遍，随机摆一次就够。
    rng = np.random.default_rng(int(seed)*7919 + 13)
    bearing = float(np.round(rng.uniform(*BEARINGS), 3))
    if abs(bearing) < CENTER_GAP:          # 摆正中间考不出会不会拐，往外挪一点
        bearing = float(np.copysign(CENTER_GAP, bearing if bearing else 1.))
    distance = float(np.round(rng.uniform(*DISTANCES), 3))
    body, brain, eyes, geom = C.build(genome, chase, seed=seed, spec=spec)
    got = C.measure(body, brain, eyes, geom, seconds, bearing, distance,
                    seed=seed, ball=True)
    row = _row(got, bearing, distance, 0.)
    rows.append(row)
    if loud:
        print("   静止球在 %+.2f 弧度、%.1f 米：最近贴到 %.2f 米%s、位移离球 %4.1f 度、"
              "走了 %.2f 米"
              % (bearing, distance, row["closest"],
                 "  到了" if row["reached"] else "", row["toward_deg"],
                 row["travelled_m"]), flush=True)
    eyes.close()
''',
"exam：静止球改 1 趟")

rep(
'''    chase_rows, no_ball = [], None
    if aware:
        # 球搬走之后，狗面对的世界每个位置都一样，所以这一趟只跑一次。
        body, brain, eyes, geom = C.build(genome, chase, seed=seed, spec=spec)
        free = C.measure(body, brain, eyes, geom, chase_seconds, CHASE_BEARINGS[0],
                         CHASE_DISTANCE, seed=seed, ball=False)
        no_ball = np.asarray(free["end_xy"], dtype=float)
        eyes.close()
    for bearing in CHASE_BEARINGS:
        body, brain, eyes, geom = C.build(genome, chase, seed=seed, spec=spec)
        got = C.measure(body, brain, eyes, geom, chase_seconds, bearing, CHASE_DISTANCE,
                        seed=seed, flee=flee, escape=CHASE_ESCAPE, curve=curve, ball=True)''',
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
"exam：三趟共用一只狗")

rep(
'''                     "  追住了" if row["kept_in_front"] else ""), flush=True)
        eyes.close()
    rows += chase_rows

    still = [row for row in rows if row["flee"] == 0.]
    reached = [row for row in still if row["reached"]]
    left_done = any(row["reached"] for row in reached if row["bearing"] > 0)
    right_done = any(row["reached"] for row in reached if row["bearing"] < 0)''',
'''                     "  追住了" if row["kept_in_front"] else ""), flush=True)
    eyes.close()
    rows += chase_rows

    still = [row for row in rows if row["flee"] == 0.]
    reached = [row for row in still if row["reached"]]
    # 「两边都会」改由追球那两趟来判（静止球只剩随机一次，判不了左右）。
    caught = [row for row in chase_rows if row["kept_in_front"] and row["upright"]]
    left_done = any(row["bearing"] > 0 for row in caught)
    right_done = any(row["bearing"] < 0 for row in caught)''',
"exam：both 改由追球判")

io.open(P, "w", encoding="utf-8", newline="\n").write(t)
print("\n改完，%d -> %d 字节" % (len(orig), len(t)))