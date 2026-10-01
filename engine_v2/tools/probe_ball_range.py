# -*- coding: utf-8 -*-
"""球放多远还看得见：把球摆在正前方偏一点，量眼睛有没有转过去、眼位那一排亮没亮。

    python tools/probe_ball_range.py --bearings 0.0,0.5 --distances 1.2,2.0,3.0,4.0,6.0
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
import chase_red_ball as C                                          # noqa: E402
import eye_geometry as G                                            # noqa: E402
from born_wired.reflex_senses import ReflexSenses                   # noqa: E402

DT = .01


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--champion", default=str(C.CHAMPION))
    ap.add_argument("--bearings", default="0.0,0.5")
    ap.add_argument("--distances", default="1.2,2.0,3.0,4.0,6.0")
    ap.add_argument("--seconds", type=float, default=1.2)
    args = ap.parse_args(argv)

    genome = dict(C.load_champion(Path(args.champion))["genome"])
    spec = C.gaze_spec()
    bearings = [float(v) for v in args.bearings.split(",")]
    distances = [float(v) for v in args.distances.split(",")]
    print("球偏(弧度) 距离(米) | 左眼yaw(度)  眼位那一排的峰值(0最右…15最左) 峰值高度")
    for bearing in bearings:
        for distance in distances:
            body, brain, eyes, geom = C.build(genome, None, seed=0, spec=spec)
            le = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_BODY, "eye_left")
            re = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_BODY, "eye_right")
            body.reset(seed=0, joint_noise=.01)
            G.place_in_front(body.model, body.data, geom, bearing, 0., distance)
            senses = ReflexSenses(body)
            observation = body.observe(); environment = senses.observe()
            pixels = eyes.observe_raw()
            bank = brain.groups["eye_proprioception"]; per = len(bank)//4
            row = np.zeros(per)
            for step in range(int(round(args.seconds/DT))):
                if getattr(brain, "eye_encoder", None) is not None:
                    body.command_eyes(brain.eye_command())
                observation = body.observe()
                if step % 10 == 0:
                    pixels = eyes.observe_raw()
                target, activation = brain.step(observation, environment=environment,
                                                autonomy=True, locomotion=0., dt=DT,
                                                learn=False, eye_pixels=pixels, startle=0.)
                observation = body.step(target, duration=DT, activation=activation)
                environment = senses.observe()
                row = np.asarray(brain.network.rates_at(bank[0*per:1*per]))
            left = np.asarray(brain.network.rates_at(bank[0*per:1*per]))
            right = np.asarray(brain.network.rates_at(bank[2*per:3*per]))
            peak_l, peak_r = int(np.argmax(left)), int(np.argmax(right))
            print("%9.2f %8.2f | %10.1f  左眼 %2d (%.2f)  右眼 %2d (%.2f)"
                  % (bearing, distance, np.degrees(body.data.qpos[19]),
                     peak_l, float(left[peak_l]), peak_r, float(right[peak_r])))
            eyes.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())