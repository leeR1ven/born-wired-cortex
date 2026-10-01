# -*- coding: utf-8 -*-
"""看球和狗的距离：是不是始终保持在几米之内（不会太近也不会太远）。"""
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

rows = [json.loads(l) for l in io.open(ROOT / "artifacts" / "追球_云_三批_起点.jsonl", encoding="utf-8") if l.strip()]
spec = ct.gaze()
fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.6))
for k in range(3):
    row = rows[k]
    body, brain, eyes, geom = C.build(row["genome"], row["chase"], seed=100+k, spec=spec)
    got = C.measure(body, brain, eyes, geom, 40., 0., ct.CHASE_DISTANCE, seed=100+k,
                    flee=ct.CHASE_FLEE, escape=14., curve=1., ball=True)
    eyes.close()
    t = np.asarray(got["ball_track"], dtype=float); d = np.asarray(got["dog_track"], dtype=float)
    step = np.diff(t, axis=0); away = t[:-1] - d[:-1]
    ok = float(np.mean(np.sum(step*away, axis=1) > 0))*100
    rng = np.asarray(got["ranges"], dtype=float)
    print("第 %d 只：球始终背离狗的步子 %.0f%%；狗离球 %.2f~%.2f 米（开始 %.2f、结束 %.2f）"
          % (k, ok, rng.min(), rng.max(), rng[0], rng[-1]))
    axes[0].plot(t[:, 0], t[:, 1], lw=2, label="球（跑法 %d）" % (k+1))
    axes[1].plot(np.arange(len(rng))*C.DT, rng, lw=2, label="第 %d 只" % (k+1))
axes[0].plot(0, 0, "s", color="navy", ms=6, label="狗出发点")
axes[0].set_aspect("equal"); axes[0].grid(alpha=.3); axes[0].legend(fontsize=8)
axes[0].set_title("球的路线（地图无限大、匀速 0.6 米/秒、随机曲线背离狗）")
axes[0].set_xlabel("x（米）"); axes[0].set_ylabel("y（米）")
axes[1].axhline(ct.CHASE_NEAR, color="orange", ls="--", lw=1.2,
                 label="算「追住了」的门槛 %.1f 米" % ct.CHASE_NEAR)
axes[1].grid(alpha=.3); axes[1].legend(fontsize=8)
axes[1].set_title("狗到球的距离（40 秒）")
axes[1].set_xlabel("秒"); axes[1].set_ylabel("米")
out = ROOT / "artifacts" / "新逃法_球路线.png"
fig.tight_layout(); fig.savefig(out, dpi=110)
print("图：", out)