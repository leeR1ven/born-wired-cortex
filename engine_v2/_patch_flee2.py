from pathlib import Path
p = Path("tools/chase_task.py")
s = p.read_text(encoding="utf-8")
pairs = [
 ("def _chase_row(got, bearing, fell=None):", "def _chase_row(got, bearing, flee=CHASE_FLEE, fell=None):"),
 ("    return dict(bearing=bearing, distance=CHASE_DISTANCE, flee=CHASE_FLEE,",
  "    return dict(bearing=bearing, distance=CHASE_DISTANCE, flee=flee,"),
 ("        row = _chase_row(got, bearing)", "        row = _chase_row(got, bearing, flee)"),
 ("CHASE_FLEE = .6               # 球逃跑的速度（米/秒）：比狗慢一点",
  "CHASE_FLEE = .6               # 球逃跑的速度默认值（米/秒）；真正跑多少由 --flee 说了算"),
 ("    只是叠了一个缓慢摆动的小偏角，所以跑的是一条弯线；快到场地边就贴着场地绕圈",
  "    只是叠了一个缓慢摆动的小偏角，所以跑的是一条弯线；地图无限大，没有场地边可绕"),
 ("    （用户 2026-10-01：始终远离狗、跑的是弯线）。左、右各一趟，每趟 20 秒",
  "    （用户 2026-10-01：始终远离狗、跑的是弯线）。左、右各一趟，每趟 20 秒"),
]
for old, new in pairs:
    n = s.count(old)
    print("命中", n, "->", old[:34])
    s = s.replace(old, new)
p.write_text(s, encoding="utf-8", newline="\n")
print("写回完成")
