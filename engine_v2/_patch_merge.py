import io
p = r"F:\born-wired-cortex\engine_v2\tools\merge_cloud_eye.py"
s = io.open(p, encoding="utf-8", newline="").read()
old = """        G.place_in_front(body.model, body.data, geom, .35*np.sin(step*DT*.8), .05, distance)"""
new = """        if ball:
            G.place_in_front(body.model, body.data, geom, .35*np.sin(step*DT*.8), .05, distance)"""
assert s.count(old) == 1
s = s.replace(old, new)
old2 = """    body.reset(seed=seed, joint_noise=.01)
    senses = ReflexSenses(body)"""
new2 = """    body.reset(seed=seed, joint_noise=.01)
    if not ball:
        body.model.geom_pos[geom] = [60., 60., -8.]   # 球挪到天边：这就是云端那一趟的条件
        mujoco.mj_forward(body.model, body.data)
    senses = ReflexSenses(body)"""
assert s.count(old2) == 1
s = s.replace(old2, new2)
old3 = """def run(genome, seconds, use_eye, spec, seed=0, distance=1.15, label=""):"""
new3 = """def run(genome, seconds, use_eye, spec, seed=0, distance=1.15, label="", ball=True):"""
assert s.count(old3) == 1
s = s.replace(old3, new3)
old4 = """    for use_eye, label in ((True, "融合：走路基因 + 红球接线"), (False, "对照：只有走路基因")):
        print("\\n%s ..." % label, flush=True)
        runs.append(run(genome, args.seconds, use_eye, spec, seed=args.seed,
                        distance=args.distance, label=label))"""
new4 = """    plans = ((True, True, "融合：走路基因 + 红球接线（球在眼前）"),
             (False, True, "对照一：只有走路基因（球也在眼前）"),
             (True, False, "对照二：走路基因 + 红球接线，但不摆球（和云端同一条件）"))
    for use_eye, ball, label in plans:
        print("\\n%s ..." % label, flush=True)
        runs.append(run(genome, args.seconds, use_eye, spec, seed=args.seed,
                        distance=args.distance, label=label, ball=ball))"""
assert s.count(old4) == 1
s = s.replace(old4, new4)
old5 = """    merged, plain = runs[0], runs[1]"""
new5 = """    merged, plain, cloud_like = runs[0], runs[1], runs[2]"""
assert s.count(old5) == 1
s = s.replace(old5, new5)
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("加好了第三趟对照")