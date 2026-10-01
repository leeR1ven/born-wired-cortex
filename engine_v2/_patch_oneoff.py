import io
p = r"F:\born-wired-cortex\engine_v2\tools\evolve_chase.py"
s = io.open(p, encoding="utf-8", newline="").read()
old = '''        hits, moved, looks, lowest, shifts, detail = 0, 0, 0, 1., [], []
        for distance in job["distances"]:
            for bearing in job["bearings"]:
                C.restore(brain, saved)
                on = C.measure(body, brain, eyes, geom, job["seconds"], bearing, distance,
                               seed=seed)
                C.restore(brain, saved)
                off = C.measure(body, brain, eyes, geom, job["seconds"], bearing, distance,
                                seed=seed, ball=False)
                shifted = float(np.linalg.norm(np.asarray(on["end_xy"])
                                               - np.asarray(off["end_xy"])))'''
new = '''        # 用户 2026-09-30：球搬走之后，狗面对的世界每一个位置都是一样的（球不在了），
        # 所以「没球」那一趟只跑一次就够 —— 拿这一趟的落点跟 18 个有球的位置逐个比。
        C.restore(brain, saved)
        off = C.measure(body, brain, eyes, geom, job["seconds"], job["bearings"][0],
                        job["distances"][0], seed=seed, ball=False)
        no_ball = np.asarray(off["end_xy"])
        hits, moved, looks, lowest, shifts, detail = 0, 0, 0, 1., [], []
        for distance in job["distances"]:
            for bearing in job["bearings"]:
                C.restore(brain, saved)
                on = C.measure(body, brain, eyes, geom, job["seconds"], bearing, distance,
                               seed=seed)
                shifted = float(np.linalg.norm(np.asarray(on["end_xy"]) - no_ball))'''
assert old in s
s = s.replace(old, new, 1)
io.open(p, "w", encoding="utf-8", newline="").write(s)

p2 = r"F:\born-wired-cortex\engine_v2\tools\chase_task.py"
t = io.open(p2, encoding="utf-8", newline="").read()
old2 = '''    for distance in DISTANCES:
        body, brain, eyes, geom = C.build(genome, chase, seed=seed, spec=spec)
        saved = C.snapshot(brain)
        for bearing in BEARINGS:
            C.restore(brain, saved)
            got = C.measure(body, brain, eyes, geom, seconds, bearing, distance, seed=seed)
            row = _row(got, bearing, distance, 0.)
            if aware:
                C.restore(brain, saved)
                free = C.measure(body, brain, eyes, geom, seconds, bearing, distance,
                                 seed=seed, ball=False)
                row["shifted_m"] = float(np.linalg.norm(np.asarray(got["end_xy"])
                                                        - np.asarray(free["end_xy"])))
                row["blind"] = bool(row["shifted_m"] < BLIND_BAR)
            rows.append(row)'''
new2 = '''    for distance in DISTANCES:
        body, brain, eyes, geom = C.build(genome, chase, seed=seed, spec=spec)
        saved = C.snapshot(brain)
        # 球搬走之后每个位置的世界都一样，所以「没球」那一趟只跑一次，拿它跟每个有球
        # 的位置比（用户 2026-09-30）。
        no_ball = None
        if aware:
            C.restore(brain, saved)
            free = C.measure(body, brain, eyes, geom, seconds, BEARINGS[0], distance,
                             seed=seed, ball=False)
            no_ball = np.asarray(free["end_xy"])
        for bearing in BEARINGS:
            C.restore(brain, saved)
            got = C.measure(body, brain, eyes, geom, seconds, bearing, distance, seed=seed)
            row = _row(got, bearing, distance, 0.)
            if aware:
                row["shifted_m"] = float(np.linalg.norm(np.asarray(got["end_xy"]) - no_ball))
                row["blind"] = bool(row["shifted_m"] < BLIND_BAR)
            rows.append(row)'''
assert old2 in t
t = t.replace(old2, new2, 1)
io.open(p2, "w", encoding="utf-8", newline="").write(t)

p3 = r"F:\born-wired-cortex\engine_v2\tools\verify_chase.py"
v = io.open(p3, encoding="utf-8", newline="").read()
old3 = '''        results = []
        for distance in distances:
            for bearing in bearings:
                C.restore(brain, saved)
                on = C.measure(body, brain, eyes, geom, args.seconds, bearing, distance,
                               seed=seed)
                C.restore(brain, saved)
                off = C.measure(body, brain, eyes, geom, args.seconds, bearing, distance,
                                seed=seed, ball=False)
                shifted = float(np.linalg.norm(np.asarray(on["end_xy"])
                                               - np.asarray(off["end_xy"])))'''
new3 = '''        # 球搬走之后每个位置的世界都一样，所以「没球」那一趟只跑一次（用户 2026-09-30）。
        C.restore(brain, saved)
        off = C.measure(body, brain, eyes, geom, args.seconds, bearings[0], distances[0],
                        seed=seed, ball=False)
        no_ball = np.asarray(off["end_xy"])
        results = []
        for distance in distances:
            for bearing in bearings:
                C.restore(brain, saved)
                on = C.measure(body, brain, eyes, geom, args.seconds, bearing, distance,
                               seed=seed)
                shifted = float(np.linalg.norm(np.asarray(on["end_xy"]) - no_ball))'''
assert old3 in v
v = v.replace(old3, new3, 1)
io.open(p3, "w", encoding="utf-8", newline="").write(v)
print("patched")