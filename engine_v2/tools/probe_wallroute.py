# -*- coding: utf-8 -*-
"""空旷场地上，wall_contact / wall_turn 这一路到底是不是一直亮着。

追球本来借的是 wall_turn 那条「往旁边拐」的路。要是它平时就半亮着，借来用就会一直拐，
跟球在哪边没关系 —— 那就不是追球。这里把每一拍的读数打出来，看它是真的平时全灭，
还是一直有人喂。

    python tools/probe_wallroute.py --seconds 4
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
import chase_red_ball as C                                          # noqa: E402

DT = .01
WATCH = ("touch", "wall_contact", "wall_turn", "wall_turn_inhibition",
         "avoidance", "avoidance_inhibition", "steering", "steering_inhibition",
         "orienting", "brain_stem_orienting" if False else "orienting")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seconds", type=float, default=4.)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    genome = C.load_champion()["genome"]
    body, brain, eyes, geom = C.build(genome, None, seed=args.seed)
    body.model.geom_pos[geom] = [60., 60., -8.]     # 球挪到天边：空旷场地，什么都别碰
    body.reset(seed=args.seed, joint_noise=.01)
    mujoco.mj_forward(body.model, body.data)
    senses = ReflexSenses(body)
    observation = body.observe()
    environment = senses.observe()
    pixels = eyes.observe_raw()
    names = [n for n in dict.fromkeys(WATCH) if n in brain.groups]
    print("空旷场地（球挪走），跑 %.1f 秒" % args.seconds)
    print("%6s | %-22s | %s" % ("秒", "摸到东西身体接触量", " ".join("%s" % n for n in names)))
    every = int(round(.5/DT))
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
            rates = np.asarray(brain.network.activity)
            got = " ".join("%.2f" % float(np.mean(rates[brain.groups[n]])) for n in names)
            print("%6.2f | %-22s | %s" % (step*DT, np.round(environment["body_touch"], 2), got))
    eyes.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())