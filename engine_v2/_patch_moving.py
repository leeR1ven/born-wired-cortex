import io
p = r"F:\born-wired-cortex\engine_v2\tools\chase_task.py"
s = io.open(p, encoding="utf-8", newline="").read()

s = s.replace('KEEP_BAR = .6                 # 「球还算在正前方」的门槛（弧度），kept 用',
              'KEEP_BAR = .35                # 「球还算在正前方」的门槛（弧度），kept 用\n'
              'MOVE_BAR = 2.0                # 一趟至少走了这么远才算「在走」（米）', 1)

old = '''    return dict(bearing=bearing, distance=distance, flee=flee, facing=got["facing"],
                settled=got["settled"], kept=kept, range_min=got["range_min"],
                travelled_m=got["travelled_m"], straightness=got["straightness"],
                heading_end=got["heading_end"], lowest_up_z=got["lowest_up_z"],
                upright=not fell,
                turned=bool(got["facing"] <= FACE_BAR and got["settled"] <= NEAR_BAR
                            and not fell),
                caught=bool(got["range_min"] <= CATCH_BAR),
                kept_in_front=bool(kept >= .5 and not fell),
                end_xy=got["end_xy"], shifted_m=float("nan"), blind=False)'''
new = '''    # 用户 2026-09-30：站着不动不算本事 —— 球摆在正前方 0.6 弧度上、径直往外跑的时候，
    # 偏角一直就是 0.6，一只原地不动的狗也能「一直正对着球」。所以「在走」是硬门槛，
    # 「朝球转了」「跟住了」都要先走起来才算数。
    moving = bool(got["travelled_m"] >= MOVE_BAR)
    return dict(bearing=bearing, distance=distance, flee=flee, facing=got["facing"],
                settled=got["settled"], kept=kept, range_min=got["range_min"],
                travelled_m=got["travelled_m"], straightness=got["straightness"],
                heading_end=got["heading_end"], lowest_up_z=got["lowest_up_z"],
                upright=not fell, moving=moving,
                turned=bool(got["facing"] <= FACE_BAR and got["settled"] <= NEAR_BAR
                            and not fell and moving),
                caught=bool(got["range_min"] <= CATCH_BAR and moving),
                kept_in_front=bool(kept >= .5 and not fell and moving),
                end_xy=got["end_xy"], shifted_m=float("nan"), blind=False)'''
assert old in s
s = s.replace(old, new, 1)

s = s.replace('''                covered=len(turns), trials=len(still),''',
              '''                covered=len(turns), trials=len(still),
                moving=sum(1 for row in still if row["moving"]),
                travels=float(np.mean([row["travelled_m"] for row in flee_rows])) if flee_rows else float("nan"),''', 1)
io.open(p, "w", encoding="utf-8", newline="").write(s)

p2 = r"F:\born-wired-cortex\engine_v2\tools\evolve_chase.py"
t = io.open(p2, encoding="utf-8", newline="").read()
t = t.replace('''    return (0 if row["upright"] else 1,
            -int(row["covered"]),''',
              '''    return (0 if row["upright"] else 1,
            -int(row["moving"]),
            -int(row["covered"]),''', 1)
t = t.replace('''    row.update(status="ok", upright=got["upright"], score=got["score"], kept=got["kept"],''',
              '''    row.update(status="ok", upright=got["upright"], moving=got["moving"],
               score=got["score"], kept=got["kept"],''', 1)
t = t.replace('''    return float(10 ** rng.uniform(-3., -.3))''',
              '''    return float(10 ** rng.uniform(-3., -.52))''', 1)
io.open(p2, "w", encoding="utf-8", newline="").write(t)
print("patched")