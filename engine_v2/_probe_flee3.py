# -*- coding: utf-8 -*-
"""看一眼新逃法：球是不是始终背离狗、路线是不是弯的、会不会绕场地一圈。"""
import io, json, sys
from pathlib import Path
ROOT = Path(r"F:\born-wired-cortex\engine_v2")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
import chase_red_ball as C
import chase_task as ct

start = ROOT / "artifacts" / "追球_云_三批_起点.jsonl"
row = json.loads(io.open(start, encoding="utf-8").readline())
spec = ct.gaze()
body, brain, eyes, geom = C.build(row["genome"], row["chase"], seed=5, spec=spec)
got = C.measure(body, brain, eyes, geom, 40., 0., ct.CHASE_DISTANCE, seed=5,
                flee=ct.CHASE_FLEE, escape=ct.CHASE_ESCAPE, curve=1., ball=True)
eyes.close()
track = np.asarray(got["ball_track"], dtype=float)
ranges = np.asarray(got["ranges"], dtype=float)
rad = np.hypot(track[:, 0], track[:, 1])
ang = np.unwrap(np.arctan2(track[:, 1], track[:, 0]))
print("球走了 %.1f 米，离场中心 %.2f -> %.2f 米（最远 %.2f）"
      % (np.sum(np.linalg.norm(np.diff(track, axis=0), axis=1)), rad[0], rad[-1], rad.max()))
print("球绕的角度：%.0f 度（一圈=360）" % np.degrees(abs(ang[-1] - ang[0])))
print("狗离球：%.2f 米 -> %.2f 米（最近 %.2f）" % (ranges[0], ranges[-1], ranges.min()))
print("球走的路程 %.1f 米、狗走的路程 %.1f 米" % (got["path_m"], got["travelled_m"]))
fig, ax = plt.subplots(figsize=(6.4, 6.4))
th = np.linspace(0, 2*np.pi, 200)
ax.plot(3.5*np.cos(th), 3.5*np.sin(th), "--", color="0.75", lw=1, label="原来的墙（3.5 米）")
ax.plot(track[:, 0], track[:, 1], "-", color="crimson", lw=2, label="红球路线")
ax.plot(track[0, 0], track[0, 1], "o", color="crimson", ms=7)
ax.plot(0, 0, "s", color="navy", ms=5, label="狗出发点")
ax.set_aspect("equal"); ax.grid(alpha=.3); ax.legend(fontsize=8)
ax.set_title("红球新逃法：始终背离狗 + 弯线 + 贴场地绕圈（40 秒 = 绕场地一圈）")
ax.set_xlabel("x（米）"); ax.set_ylabel("y（米）")
out = ROOT / "artifacts" / "新逃法_球路线.png"
fig.tight_layout(); fig.savefig(out, dpi=110)
print("图：", out)