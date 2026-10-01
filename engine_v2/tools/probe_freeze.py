# -*- coding: utf-8 -*-
"""球摆在正前方时狗不走 —— 看看是哪个反射把它按住了。

    python tools/probe_freeze.py --bearing 0.0
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
WATCH = ("near", "brake", "avoidance", "withdrawal", "retreat", "startle", "initiation",
         "fatigue", "rest", "appetitive", "orienting", "rhythm_recruitment", "locomotion",
         "curiosity", "steering", "wall_contact", "wall_turn", "aversive", "protective_tilt",
         "eye_fusion", "eye_near")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bearing", type=float, default=0.)
    ap.add_argument("--distance", type=float, default=1.5)
    ap.add_argument("--seconds", type=float, default=4.)
    ap.add_argument("--no-ball", action="store_true")
    args = ap.parse_args(argv)
    genome = C.load_champion()["genome"]
    body, brain, eyes, geom = C.build(genome, None, seed=0)
    base = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_BODY, "base")
    body.reset(seed=0, joint_noise=.01)
    if args.no_ball:
        body.model.geom_pos[geom] = [60., 60., -8.]
        mujoco.mj_forward(body.model, body.data)
    else:
        G.place_in_front(body.model, body.data, geom, args.bearing, 0., args.distance)
    senses = ReflexSenses(body)
    observation = body.observe()
    environment = senses.observe()
    pixels = eyes.observe_raw()
    start = np.asarray(body.data.xpos[base], dtype=float).copy()
    every = int(round(.5/DT))
    print("球：方位 %.2f、%.2f 米%s" % (args.bearing, args.distance, "（不摆球）" if args.no_ball else ""))
    print("%6s %8s | %s" % ("秒", "走了", "活跃的反射细胞"))
    for step in range(int(round(args.seconds/DT))):
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
        if step % every == 0:
            rates = brain.network.activity
            now = np.asarray(body.data.xpos[base], dtype=float)
            hot = []
            for name in WATCH:
                if name not in brain.groups:
                    continue
                value = float(np.mean(rates[brain.groups[name]]))
                if value > .05:
                    hot.append("%s %.2f" % (name, value))
            print("%6.2f %8.2f | %s" % (step*DT, float(np.linalg.norm(now - start)),
                                        "  ".join(hot) or "-"))
    eyes.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())