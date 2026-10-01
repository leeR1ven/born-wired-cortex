import io
p = r"F:\born-wired-cortex\engine_v2\tools\chase_red_ball.py"
s = io.open(p, encoding="utf-8", newline="").read()
old = '''    ball = np.asarray(body.data.geom_xpos[geom], dtype=float).copy()

    senses = ReflexSenses(body)'''
new = '''    ball = np.asarray(body.data.geom_xpos[geom], dtype=float).copy()
    ball0 = ball.copy()

    senses = ReflexSenses(body)'''
assert old in s
s = s.replace(old, new, 1)
old2 = '''    return dict(settled=settled, facing=facing, end_xy=[float(end[0]), float(end[1])],'''
new2 = '''    return dict(settled=settled, facing=facing, end_xy=[float(end[0]), float(end[1])],
                start_xy=[float(start[0]), float(start[1])],
                ball0_xy=[float(ball0[0]), float(ball0[1])],'''
assert old2 in s
s = s.replace(old2, new2, 1)
io.open(p, "w", encoding="utf-8", newline="").write(s)

p2 = r"F:\born-wired-cortex\engine_v2\tools\chase_task.py"
t = io.open(p2, encoding="utf-8", newline="").read()

t = t.replace('''BEARINGS = (.6, -.6)          # 正 = 狗的左前方；负 = 右前方
DISTANCES = (1.3, 1.8)''',
'''BEARINGS = (.6, -.6)          # 正 = 狗的左前方；负 = 右前方
DISTANCE = 1.3                # 球必须摆在狗跟前（4 米外那颗小球，视网膜三区根本不亮）''', 1)
t = t.replace('''SECONDS = 6.
FLEE_SECONDS = 6.''', '''SECONDS = 4.
FLEE_SECONDS = 8.''', 1)
t = t.replace('''CATCH_BAR = .9                # 追上了的门槛（米）''',
              '''CATCH_BAR = 1.2               # 追上了的门槛（米）
TOWARD_DEG = 25.               # 位移方向离「球所在的方向」不超过这么多度 = 朝球走
TOWARD_MOVE = 1.0             # 这一趟至少走了这么远才判「朝球走」
FLEE_CLOSE = 2.0              # 球躲着跑 8 秒之后还留在 2 米内 = 跟住了
FLEE_MOVE = 2.0               # 跟住那两趟至少要走过这么多米''', 1)

old_row = t[t.index("def _row(got"):t.index("def exam(")]
new_row = '''def _row(got, bearing, distance, flee):
    fell = got["lowest_up_z"] < FALLEN
    headings = np.abs(np.asarray(got["headings"], dtype=float))
    kept = float(np.mean(headings <= KEEP_BAR))
    # 用户 2026-09-30：判分的重点是「朝球走」本身。球摆在跟前 1.3 米，会追的狗
    # 不到一秒就把它甩到身后 —— 所以不能拿「最后有没有正对着球」当判据（那反而
    # 罚好狗）。直接量位移方向：这一趟从起点到终点走出来的那条线，和「球一开始
    # 在哪边」差多少度。直着往前走、球在左前方 0.6 弧度，差 34 度；真朝着球走的，
    # 差几度。
    start = np.asarray(got["start_xy"], dtype=float)
    ball0 = np.asarray(got["ball0_xy"], dtype=float)
    walk = np.asarray(got["end_xy"], dtype=float) - start
    to_ball = ball0 - start
    length = float(np.linalg.norm(walk))*float(np.linalg.norm(to_ball))
    toward_deg = (float("nan") if length < 1e-9
                  else float(np.degrees(np.arccos(np.clip(float(walk @ to_ball)/length, -1., 1.)))))
    moving = bool(got["travelled_m"] >= (TOWARD_MOVE if not flee else FLEE_MOVE))
    toward = bool(not fell and moving and toward_deg <= TOWARD_DEG)
    return dict(bearing=bearing, distance=distance, flee=flee, facing=got["facing"],
                settled=got["settled"], kept=kept, range_min=got["range_min"],
                toward_deg=toward_deg, toward=toward,
                travelled_m=got["travelled_m"], straightness=got["straightness"],
                heading_end=got["heading_end"], lowest_up_z=got["lowest_up_z"],
                upright=not fell, moving=moving,
                turned=toward,
                caught=bool(got["range_min"] <= CATCH_BAR and moving),
                kept_in_front=bool(flee > 0. and not fell and moving
                                   and got["settled"] <= FLEE_CLOSE),
                end_xy=got["end_xy"], shifted_m=float("nan"), blind=False)


'''
t = t.replace(old_row, new_row, 1)

old_exam = t[t.index("def exam("):t.index("    still = [row for row")]
new_exam = '''def exam(genome, seed, seconds=SECONDS, spec=None, loud=False, aware=False, chase=None,
         flee=FLEE):
    """一个模型考一场。chase 就是这一只身上那套随机的追球接法；None 就是不接。"""
    spec = gaze() if spec is None else spec
    rows = []
    for bearing in BEARINGS:
        got = _trip(genome, seed, spec, bearing, DISTANCE, seconds, True, chase)
        row = _row(got, bearing, DISTANCE, 0.)
        if aware:
            # 「球挪到天边」那趟对照只在 --aware 时跑：它是拆穿「原地打转」的，
            # 平时省下这些时间。
            free = _trip(genome, seed, spec, bearing, DISTANCE, seconds, False, chase)
            row["shifted_m"] = float(np.linalg.norm(np.asarray(got["end_xy"])
                                                    - np.asarray(free["end_xy"])))
            row["blind"] = bool(row["shifted_m"] < BLIND_BAR)
        rows.append(row)
        if loud:
            print("   球不动、在 %+.2f 弧度：位移离球 %4.1f 度、走了 %.2f 米%s"
                  % (bearing, row["toward_deg"], row["travelled_m"],
                     "  朝球走" if row["toward"] else ""), flush=True)
    flee_rows = []
    if flee > 0.:
        for bearing in BEARINGS:
            got = _trip(genome, seed, spec, bearing, DISTANCE, FLEE_SECONDS, True, chase,
                        flee=flee)
            row = _row(got, bearing, DISTANCE, flee)
            flee_rows.append(row)
            rows.append(row)
            if loud:
                print("   球躲着跑、从 %+.2f 弧度起步：最后离球 %.2f 米、最近 %.2f 米、"
                      "位移离球 %4.1f 度、走了 %.2f 米%s"
                      % (bearing, row["settled"], row["range_min"], row["toward_deg"],
                         row["travelled_m"], "  跟住了" if row["kept_in_front"] else ""),
                      flush=True)
'''
t = t.replace(old_exam, new_exam, 1)
io.open(p2, "w", encoding="utf-8", newline="").write(t)
print("patched")