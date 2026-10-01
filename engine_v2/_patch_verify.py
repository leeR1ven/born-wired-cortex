import io
p = r"F:\born-wired-cortex\engine_v2\tools\evolve_chase.py"
s = io.open(p, encoding="utf-8", newline="").read()

helper = '''def verify_one(job):
    """复试：换它没见过的位置再考一遍，并加一条「把球挪到天边」的对照。

    用户 2026-09-30：确认不是碰巧。考过的位置是 0.5 弧度 x 3.0/4.0 米；复试换成
    0.7/0.5/0.3 弧度 x 2.5/3.5/4.5 米，18 个位置。同一个种子、只把球搬到天边再跑
    一趟，两趟落点差多少米就是「球」对它的行为有多大影响 —— 差不到 0.05 米，说明
    它只是习惯性往一边拐，正好路过球，不是朝球走。
    """
    import numpy as np
    import chase_red_ball as C
    import tools.chase_task as ct
    started = time.perf_counter()
    genome, chase, seed = job["genome"], job["chase"], job["seed"]
    row = dict(index=job["index"], seed=seed)
    try:
        spec = ct.gaze()
        body, brain, eyes, geom = C.build(genome, chase, seed=seed, spec=spec)
        saved = C.snapshot(brain)
        hits, moved, lowest = 0, 0, 1.
        for distance in job["distances"]:
            for bearing in job["bearings"]:
                C.restore(brain, saved)
                got = C.measure(body, brain, eyes, geom, job["seconds"], bearing, distance,
                                seed=seed)
                moving = got["travelled_m"] >= 2.0
                hits += int(moving and not got["lowest_up_z"] < .3
                            and float(got["range_min"]) <= 1.0)
                moved += int(moving)
                lowest = min(lowest, float(got["lowest_up_z"]))
        C.restore(brain, saved)
        on = C.measure(body, brain, eyes, geom, job["seconds"], job["bearings"][0],
                       job["distances"][0], seed=seed)
        C.restore(brain, saved)
        off = C.measure(body, brain, eyes, geom, job["seconds"], job["bearings"][0],
                        job["distances"][0], seed=seed, ball=False)
        eyes.close()
        row.update(status="ok", hits=hits, trials=len(job["distances"])*len(job["bearings"]),
                   moved=moved, lowest_up_z=lowest,
                   shifted_m=float(np.linalg.norm(np.asarray(on["end_xy"])
                                                  - np.asarray(off["end_xy"]))))
    except Exception as exc:
        row.update(status="refused", error="%s: %s" % (type(exc).__name__, exc))
    row["wall_seconds"] = time.perf_counter() - started
    return row


def looks_at_ball(row, bar=.05):
    """复试里球挪走之后行为变了吗。变不了就是碰巧。"""
    return bool(row.get("status") == "ok" and row.get("shifted_m", 0.) >= bar)


def pick_parents(rows, keep_stable, keep_fast):'''
s = s.replace("def pick_parents(rows, keep_stable, keep_fast):", helper, 1)

s = s.replace('''    ap.add_argument("--quiet", action="store_true")''',
'''    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--verify-top", type=int, default=8,
                    help="每一代挑完爹之后，拿前几名做复试（没见过的位置 + 球挪走对照）；0 = 不复试")
    ap.add_argument("--verify-bearings", default="-0.7,-0.5,-0.3,0.3,0.5,0.7")
    ap.add_argument("--verify-distances", default="2.5,3.5,4.5")''', 1)

old = '''            parents = pick_parents(rows, args.keep_stable, args.keep_fast)
            keep = "%s_g%02d_keep.jsonl" % (args.out_prefix, gen)'''
new = '''            parents = pick_parents(rows, args.keep_stable, args.keep_fast)
            if args.verify_top > 0 and parents:
                # 用户 2026-09-30：确认不是碰巧 —— 挑完爹之后，前几名再做一次复试，
                # 球挪走行为不变的（习惯性往一边拐）从爹名单里剔掉。
                ranked = sorted(rows, key=lambda row: rank(row))[:args.verify_top]
                jobs = [dict(index=row["index"], seed=row["seed"], genome=row["genome"],
                             chase=row["chase"], seconds=args.seconds,
                             bearings=[float(v) for v in args.verify_bearings.split(",")],
                             distances=[float(v) for v in args.verify_distances.split(",")])
                        for row in ranked]
                print("  复试 %d 只：18 个没见过的位置 + 球挪走对照" % len(jobs), flush=True)
                checked = list(pool.map(verify_one, jobs))
                ev.append(dict(kind="verify", gen=gen, rows=checked),
                          "%s_g%02d_verify.jsonl" % (args.out_prefix, gen))
                by_index = {row["index"]: row for row in checked}
                for row in parents:
                    got = by_index.get(row["index"])
                    if got is not None:
                        row["verify"] = got
                kept, dropped = [], []
                for row in parents:
                    got = row.get("verify")
                    (kept if got is None or looks_at_ball(got) else dropped).append(row)
                if dropped:
                    print("  复试剔掉 %d 只（球挪走行为不变 = 碰巧）：%s"
                          % (len(dropped), "、".join(
                              "第%d只(差%.2f米)" % (row["index"], row["verify"]["shifted_m"])
                              for row in dropped)), flush=True)
                print("  复试留下 %d 只：%s" % (len(kept), "、".join(
                    "第%d只(贴到%d/%d、球挪走差%.2f米)"
                    % (row["index"], row["verify"]["hits"], row["verify"]["trials"],
                       row["verify"]["shifted_m"]) for row in kept if row.get("verify"))
                    or "（没有一只进了复试）"), flush=True)
                parents = kept or parents
            keep = "%s_g%02d_keep.jsonl" % (args.out_prefix, gen)'''
assert old in s
s = s.replace(old, new, 1)
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("patched")