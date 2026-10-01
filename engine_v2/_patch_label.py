# -*- coding: utf-8 -*-
from pathlib import Path
p = Path("tools/track_red_gaze.py")
t = p.read_text(encoding="utf-8", newline="")
old = '''        row = dict(step=step, t=round(t, 3), bearing=round(float(ask[0]["yaw"]), 4),
                   distance=args.distance)
'''
new = '''        origin, rotation = head_frame(body)
        offset = rotation.T @ (np.asarray(body.data.geom_xpos[target], dtype=float) - origin)
        row = dict(step=step, t=round(t, 3),
                   bearing=round(float(np.arctan2(offset[1], offset[0])), 4),
                   elevation=round(float(np.arctan2(offset[2], np.hypot(offset[0], offset[1]))), 4),
                   distance=args.distance)
'''
assert t.count(old) == 1
t = t.replace(old, new)
old2 = '''    figure.suptitle("红球 %+.1f 度 / %+.1f 度   t=%.2f 秒   偏离：左 %.1f 度 右 %.1f 度"
                    % (np.degrees(row["bearing"]), np.degrees(0.), row["t"],'''
new2 = '''    figure.suptitle("红球 %+.1f 度 / %+.1f 度   t=%.2f 秒   偏离：左 %.1f 度 右 %.1f 度"
                    % (np.degrees(row["bearing"]), np.degrees(row["elevation"]), row["t"],'''
assert t.count(old2) == 1
p.write_text(t.replace(old2, new2), encoding="utf-8", newline="")
print("标签改好")