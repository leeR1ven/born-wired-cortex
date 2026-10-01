# -*- coding: utf-8 -*-
import io, json, sys
from pathlib import Path
ROOT = Path(r"F:\born-wired-cortex\engine_v2")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
import numpy as np
import chase_red_ball as C
import chase_task as ct
print("FIELD_RADIUS =", getattr(C, "FIELD_RADIUS", "没这个常量"))
print("WANDER_MAX =", getattr(C, "WANDER_MAX", "?"), "WANDER_SIGMA =", getattr(C, "WANDER_SIGMA", "?"),
      "WANDER_FIELD =", getattr(C, "WANDER_FIELD", "（已删）"))

row = json.loads(io.open(ROOT / "artifacts" / "追球_云_三批_起点.jsonl", encoding="utf-8").readline())
spec = ct.gaze()
body, brain, eyes, geom = C.build(row["genome"], row["chase"], seed=5, spec=spec)
got = C.measure(body, brain, eyes, geom, 30., 0., ct.CHASE_DISTANCE, seed=5,
                flee=ct.CHASE_FLEE, escape=14., curve=1., ball=True)
eyes.close()
track = np.asarray(got["ball_track"], dtype=float)
rad = np.hypot(track[:, 0], track[:, 1])
for i in range(0, len(rad), 100):
    print("  t=%4.1f 秒  球在 (%.2f, %.2f)  离场中心 %.2f 米" % (i*C.DT, track[i,0], track[i,1], rad[i]))