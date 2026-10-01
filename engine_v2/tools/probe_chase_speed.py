# -*- coding: utf-8 -*-
"""这题是不是做得出来：有球在眼前的时候，狗自己跑多快？

球跑 0.6 米/秒。狗要是连 0.6 都跑不到，那追球就是从数学上不可能，考题得改。
这里把「球不动」和「球躲着跑」两种跑法并排量出来：每 0.5 秒报一次离球多远、已经走了多少米。

    python tools/probe_chase_speed.py --bearing 0.0
"""
import argparse
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "tools"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import mujoco                                                       # noqa: E402
from born_wired.reflex_senses import ReflexSenses                   # noqa: E402
import eye_geometry as G                                            # noqa: E402
import chase_red_ball as C                                          # noqa: E402

DT = .01


def trial(genome, seed, spec, bearing, distance, flees, seconds):
    body, brain, eyes, geom = C.build(genome, None, seed=seed, spec=spec)
    base = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_BODY, "base")
    left = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_BODY, "eye_left")
    right = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_BODY, "eye_right")
    body.reset(seed=seed, joint_noise=.01)
    G.place_in_front(body.model, body.data, geom, bearing, 0., distance)
    ball = np.asarray(body.data.geom_xpos[geom], dtype=float).copy()
    senses = ReflexSenses(body)
    observation, environment, pixels = body.observe(), senses.observe(), eyes.observe_raw()
    start = np.asarray(body.data.xpos[base], dtype=float).copy()
    trace = []
    for step in range(int(round(seconds/DT))):
        if getattr(brain, "eye_encoder", None) is not None:
            body.command_eyes(brain.eye_command())
        observation = body.observe()
        if step % 10 == 0:
            pixels = eyes.observe_raw()
        target, activation = brain.step(observation, environment=environment, autonomy=True,
                                        locomotion=0., dt=DT, learn=False, eye_pixels=pixels,
                                        startle=0.)
        observation = body.step(target, duration=DT, activation=activation)
        environment = senses.observe()
        if flees:
            here = np.asarray(body.data.xpos[base], dtype=float)
            away = ball - here
            length = float(np.linalg.norm(away))
            if length > 1e-6 and length < 6.:
                ball = ball + away/length*.6*DT
                body.model.geom_pos[geom] = ball
                mujoco.mj_forward(body.model, body.data)
        if step % 50 == 0:
            here = np.asarray(body.data.xpos[base], dtype=float)
            trace.append((step*DT, float(np.linalg.norm(ball - here)),
                          float(np.linalg.norm(here - start))))
    eyes.close()
    return trace


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bearing", type=float, default=0.)
    ap.add_argument("--distance", type=float, default=1.2)
    ap.add_argument("--seconds", type=float, default=6.)
    args = ap.parse_args(argv)
    row = C.load_champion()
    genome, seed, spec = row["genome"], int(row["seed"]), C.gaze_spec()
    for flees in (False, True):
        trace = trial(genome, seed, spec, args.bearing, args.distance, flees, args.seconds)
        head = "球躲着跑（0.6 米/秒）" if flees else "球不动"
        print("\n%s：球在 %+.2f 弧度、%.1f 米" % (head, args.bearing, args.distance))
        print("   %5s %10s %10s %10s" % ("秒", "离球(米)", "已走(米)", "时速(米/秒)"))
        for i, (when, away, walked) in enumerate(trace):
            speed = ((trace[i][2] - trace[i-1][2])/(trace[i][0] - trace[i-1][0])) if i else 0.
            print("   %5.1f %10.2f %10.2f %10.2f" % (when, away, walked, speed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())