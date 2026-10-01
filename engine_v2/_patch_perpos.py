import io
p = r"F:\born-wired-cortex\engine_v2\tools\evolve_chase.py"
s = io.open(p, encoding="utf-8", newline="").read()
i = s.index("def verify_one(job):")
j = s.index("def pick_parents(rows, keep_stable, keep_fast):")
new = '''def verify_one(job):
    """复试：换它没见过的位置再考一遍，而且**每一个位置**都配一趟「把球搬走」的对照。

    用户 2026-09-30：只换球的位置还不够 —— 万一某只只对球摆在某个地方有反应呢？
    所以每个位置都跑两趟：一趟球摆在那儿、一趟球搬走（同一个种子、别的什么都不动）。
    两趟落点差不到 0.05 米，说明那个位置上它根本没在看球，那一趟的「贴到球」不算数。

    考过的位置是 0.5 弧度 x 3.0/4.0 米；复试换成 -0.7~0.7 弧度 x 2.5/3.5/4.5 米。
    """
    import numpy as np
    import chase_red_ball as C
    import tools.chase_task as ct
    started = time.perf_counter()
    genome, chase, seed = job["genome"], job["chase"], job["seed"]
    bar = float(job.get("bar", .05))
    row = dict(index=job["index"], seed=seed)
    try:
        spec = ct.gaze()
        body, brain, eyes, geom = C.build(genome, chase, seed=seed, spec=spec)
        saved = C.snapshot(brain)
        hits, moved, looks, lowest, shifts, detail = 0, 0, 0, 1., [], []
        for distance in job["distances"]:
            for bearing in job["bearings"]:
                C.restore(brain, saved)
                on = C.measure(body, brain, eyes, geom, job["seconds"], bearing, distance,
                               seed=seed)
                C.restore(brain, saved)
                off = C.measure(body, brain, eyes, geom, job["seconds"], bearing, distance,
                                seed=seed, ball=False)
                shifted = float(np.linalg.norm(np.asarray(on["end_xy"])
                                               - np.asarray(off["end_xy"])))
                moving = bool(on["travelled_m"] >= 2.0 and on["lowest_up_z"] >= .3)
                close = float(on["range_min"]) <= 1.0
                looks_here = bool(shifted >= bar)
                hits += int(moving and close and looks_here)
                moved += int(moving)
                looks += int(looks_here)
                shifts.append(shifted)
                lowest = min(lowest, float(on["lowest_up_z"]))
                detail.append(dict(bearing=bearing, distance=distance,
                                   closest=float(on["range_min"]), shifted_m=shifted,
                                   moving=moving, looks_at_ball=looks_here))
        eyes.close()
        row.update(status="ok", hits=hits, trials=len(detail), moved=moved, looks=looks,
                   lowest_up_z=lowest,
                   shifted_m=float(np.mean(shifts)) if shifts else float("nan"),
                   shifted_max=float(np.max(shifts)) if shifts else float("nan"), at=detail)
    except Exception as exc:
        row.update(status="refused", error="%s: %s" % (type(exc).__name__, exc))
    row["wall_seconds"] = time.perf_counter() - started
    return row


def looks_at_ball(row, least=2):
    """复试：它有没有几个位置是真的在看球（球搬走行为就变）。一个都没有就是碰巧。"""
    return bool(row.get("status") == "ok" and row.get("looks", 0) >= least)


'''
s = s[:i] + new + s[j:]
s = s.replace('''                print("  复试留%d 只''', '''                print("  复试留%d 只''')
old = '''                    print("  复试剔掉 %d 只（球挪走行为不变 = 碰巧）：%s"
                          % (len(dropped), "、".join(
                              "第%d只(差%.2f米)" % (row["index"], row["verify"]["shifted_m"])
                              for row in dropped)), flush=True)'''
new2 = '''                    print("  复试剔掉 %d 只（球搬走行为不变 = 碰巧）：%s"
                          % (len(dropped), "、".join(
                              "第%d只(看球位置%d/18)" % (row["index"], row["verify"]["looks"])
                              for row in dropped)), flush=True)'''
assert old in s
s = s.replace(old, new2, 1)
old2 = '''                print("  复试留下 %d 只：%s" % (len(kept), "、".join(
                    "第%d只(贴到%d/%d、球挪走差%.2f米)"
                    % (row["index"], row["verify"]["hits"], row["verify"]["trials"],
                       row["verify"]["shifted_m"]) for row in kept if row.get("verify"))
                    or "（没有一只进了复试）"), flush=True)'''
new3 = '''                print("  复试留下 %d 只：%s" % (len(kept), "、".join(
                    "第%d只(贴到%d/%d、看球位置%d/18)"
                    % (row["index"], row["verify"]["hits"], row["verify"]["trials"],
                       row["verify"]["looks"]) for row in kept if row.get("verify"))
                    or "（没有一只进了复试）"), flush=True)'''
assert old2 in s
s = s.replace(old2, new3, 1)
s = s.replace('''                print("  复试 %d 只：18 个没见过的位置 + 球挪走对照" % len(jobs), flush=True)''',
              '''                print("  复试 %d 只：18 个没见过的位置，每个位置各配一趟「球搬走」对照"
                      % len(jobs), flush=True)''', 1)
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("patched")