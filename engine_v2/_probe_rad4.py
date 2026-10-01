import io, json, sys
from pathlib import Path
ROOT = Path(r"F:\born-wired-cortex\engine_v2")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
import numpy as np
import chase_red_ball as C
import chase_task as ct
row = json.loads(io.open(ROOT / "artifacts" / "追球_云_三批_起点.jsonl", encoding="utf-8").readline())
spec = ct.gaze()
body, brain, eyes, geom = C.build(row["genome"], row["chase"], seed=5, spec=spec)
got = C.measure(body, brain, eyes, geom, 40., 0., ct.CHASE_DISTANCE, seed=5,
                flee=ct.CHASE_FLEE, escape=14., curve=1., ball=True)
eyes.close()
t = np.asarray(got["ball_track"], dtype=float)
rad = np.hypot(t[:, 0], t[:, 1]); ang = np.unwrap(np.arctan2(t[:, 1], t[:, 0]))
print("球 40 秒走了 %.1f 米，离场中心 %.2f -> %.2f 米（半径稳不稳看这个）" % (
      np.sum(np.linalg.norm(np.diff(t, axis=0), axis=1)), rad[0], rad[-1]))
print("绕了 %.0f 度（一圈=360）" % np.degrees(abs(ang[-1] - ang[0])))
print("半径最小 %.2f、最大 %.2f" % (rad.min(), rad.max()))