# -*- coding: utf-8 -*-
"""走路的时候眼球到底盯没盯住球：球偏在哪、眼睛转到哪、眼位那一排亮的是哪半边。

    python tools/probe_gaze_walking.py --seconds 2 --bearing 0.6 --distance 1.2
"""
import argparse
import sys
import time
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
    ap.add_argument("--seconds", type=float, default=2.)
    ap.add_argument("--bearing", type=float, default=.6)
    ap.add_argument("--distance", type=float, default=1.2)
    ap.add_argument("--turn-gain", type=float, default=None)
    ap.add_argument("--flip", action="store_true")
    ap.add_argument("--gain", type=float, default=1.)
    ap.add_argument("--no-eye", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--every", type=float, default=.05)
    args = ap.parse_args(argv)

    genome = dict(C.load_champion(Path(args.champion))["genome"])
    chase = None
    if args.turn_gain is not None:
        chase = {"turn_gain": args.turn_gain, "gain": args.gain, "source": "position",
                 "flip": bool(args.flip)}
    body, brain, eyes, geom = C.build(genome, chase, seed=args.seed, eye=not args.no_eye)
    base = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_BODY, "base")
    left_eye = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_BODY, "eye_left")
    right_eye = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_BODY, "eye_right")
    body.reset(seed=args.seed, joint_noise=.01)
    G.place_in_front(body.model, body.data, geom, args.bearing, 0., args.distance)
    ball = np.asarray(body.data.geom_xpos[geom], dtype=float).copy()

    senses = ReflexSenses(body)
    observation = body.observe()
    environment = senses.observe()
    pixels = eyes.observe_raw()
    bank = brain.groups["eye_proprioception"]
    per = len(bank)//4
    yaw_l = bank[0*per:1*per]
    yaw_r = bank[2*per:3*per]

    print("球：%+.2f 弧度、%.2f 米%s" % (args.bearing, args.distance,
          "" if chase is None else "；追球拧髋 %.3f%s" % (args.turn_gain, " 对调" if args.flip else "")))
    print("%5s | %8s %8s %8s | %7s %7s | %s"
          % ("秒", "球偏(度)", "左眼yaw", "右眼yaw", "眼位左半", "眼位右半", "转向细胞"))
    started = time.perf_counter()
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
        if step % max(1, int(round(args.every/DT))) == 0:
            frame = np.asarray(body.data.xmat[base], dtype=float).reshape(3, 3)
            forward, leftward = frame[:, 0], frame[:, 1]
            middle = (np.asarray(body.data.xpos[left_eye]) + np.asarray(body.data.xpos[right_eye]))/2.
            aim = ball - middle
            bearing = np.degrees(np.arctan2(float(aim @ leftward), float(aim @ forward)))
            left_rate = float(np.mean(brain.network.rates_at(yaw_l)))
            right_rate = float(np.mean(brain.network.rates_at(yaw_r)))
            turn = ""
            if chase is not None:
                turn = "%.2f/%.2f" % (
                    float(np.mean(brain.network.rates_at(brain.groups["chase_turn_left"]))),
                    float(np.mean(brain.network.rates_at(brain.groups["chase_turn_right"]))))
            print("%5.2f | %8.1f %8.1f %8.1f | %7.2f %7.2f | %s"
                  % (step*DT, bearing,
                     np.degrees(body.data.qpos[19]), np.degrees(body.data.qpos[21]),
                     left_rate, right_rate, turn))
    print("用时 %.1f 秒" % (time.perf_counter() - started))
    eyes.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())