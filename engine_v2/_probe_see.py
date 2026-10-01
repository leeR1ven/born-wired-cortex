# -*- coding: utf-8 -*-
"""球放多远狗眼就看不见了：数视网膜上「球那一块」的像素（红色度 2R-G-B > 0.5）。"""
import io, json, sys
from pathlib import Path
ROOT = Path(r"F:\born-wired-cortex\engine_v2")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
import numpy as np, mujoco
import chase_red_ball as C
import chase_task as ct
import eye_geometry as G

row = json.loads(io.open(ROOT / "artifacts" / "追球_云_三批_起点.jsonl", encoding="utf-8").readline())
spec = ct.gaze()
body, brain, eyes, geom = C.build(row["genome"], row["chase"], seed=1, spec=spec)
print("  距离   红球像素(左眼)   球占几个像素格   角直径")
for d in (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 8.0):
    G.place_in_front(body.model, body.data, geom, 0., 0., d)
    mujoco.mj_forward(body.model, body.data)
    raw = np.asarray(eyes.observe_raw(), dtype=float)/255.
    red = 2.*raw[0, :, :, 0] - raw[0, :, :, 1] - raw[0, :, :, 2]
    n = int((red > .5).sum())
    ang = np.degrees(2*np.arctan(.06/d))
    print("%5.1f 米   %6d            %4.1f           %.2f 度" % (d, n, n/1.0, ang))
eyes.close()