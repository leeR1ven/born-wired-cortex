# -*- coding: utf-8 -*-
"""每趟 40 秒 -> 20 秒；再加一个「整趟平均离球几米」的判分。

用户 2026-10-01：地图无限大、球匀速逃，狗不追 10 秒后球就跑出视野了，
后面的时间白跑。20 秒够看出会不会追（用户：至少 20 秒左右），迭代也更快。
「跑完落后几米」只看最后一秒，噪声大；加上「整趟平均距离」当主判分更稳。
"""
import io

q = r"F:\born-wired-cortex\engine_v2\tools\chase_task.py"
s = io.open(q, encoding="utf-8", newline="").read()
old = "CHASE_SECONDS = 40.           # 每趟考多少秒（球 0.6 米/秒绕场地一圈要 35 秒左右）"
new = ("CHASE_SECONDS = 20.           # 每趟考多少秒（球匀速逃，20 秒够看出会不会追；\n"
       "                              # 再久球就跑出视野了，后面的时间白跑）")
assert old in s, "no CHASE_SECONDS"
s = s.replace(old, new, 1)
s = s.replace("每趟 40 秒", "每趟 20 秒", 1)

old = """    return dict(bearing=bearing, distance=CHASE_DISTANCE, flee=CHASE_FLEE,
                settled=settled, range_min=float(got["range_min"]),"""
new = """    return dict(bearing=bearing, distance=CHASE_DISTANCE, flee=CHASE_FLEE,
                settled=settled, mean=float(np.mean(np.asarray(got["ranges"], dtype=float))),
                range_min=float(got["range_min"]),"""
assert old in s, "no chase row"
s = s.replace(old, new, 1)

old = """        chase_settled=float(np.mean([row["settled"] for row in chase_rows])) if chase_rows else float("nan"),
"""
new = """        chase_settled=float(np.mean([row["settled"] for row in chase_rows])) if chase_rows else float("nan"),
        chase_mean=float(np.mean([row["mean"] for row in chase_rows])) if chase_rows else float("nan"),
"""
assert old in s, "no chase_settled aggregate"
s = s.replace(old, new, 1)
io.open(q, "w", encoding="utf-8", newline="").write(s)

r = r"F:\born-wired-cortex\engine_v2\tools\evolve_chase.py"
s = io.open(r, encoding="utf-8", newline="").read()
old = "               chase_settled=got[\"chase_settled\"], chase_best=got[\"chase_best\"],"
new = "               chase_settled=got[\"chase_settled\"], chase_mean=got[\"chase_mean\"],\n               chase_best=got[\"chase_best\"],"
assert old in s, "no row.update chase"
s = s.replace(old, new, 1)

old = """            -int(bool(row.get("both"))),
            float(row.get("chase_settled", 99.)),
            -float(row.get("chase_in_view", 0.)),
            -float(row.get("chase_approach", -99.)),
            float(row.get("chase_min", 99.)),"""
new = """            -int(bool(row.get("both"))),
            float(row.get("chase_mean", 99.)),
            -float(row.get("chase_in_view", 0.)),
            float(row.get("chase_settled", 99.)),
            -float(row.get("chase_approach", -99.)),
            float(row.get("chase_min", 99.)),"""
assert old in s, "no rank keys"
s = s.replace(old, new, 1)

s = s.replace('default=40., help="追着跑的球那几趟每趟考多少秒"',
              'default=20., help="追着跑的球那几趟每趟考多少秒"', 1)
s = s.replace('job.get("chase_seconds", 40.)', 'job.get("chase_seconds", 20.)', 1)
s = s.replace('    ap.add_argument("--generations", type=int, default=40, help="一共生几代")',
              '    ap.add_argument("--generations", type=int, default=24, help="一共生几代")', 1)
io.open(r, "w", encoding="utf-8", newline="").write(s)
print("patched: 20 s trips + chase_mean")