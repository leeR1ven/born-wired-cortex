# -*- coding: utf-8 -*-
"""追球之前先看清楚：球偏在狗左边时，哪些神经元在亮、亮多久。

要回答的问题：
 1. 眼肌那 8 个神经元（4 个方向 x 2 只眼）里，哪一个亮、一直亮还是闪一下；
 2. 眼睛位置那一路（eye_proprioception，每个关节一小排，从一头到另一头）亮的是哪一头；
 3. 狗的头有没有真的转过去（只靠眼睛看不算，要看身子朝哪）。

    python tools/probe_chase_signals.py --bearing 0.5 --seconds 6
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
from born_wired.stereo_senses import RawEyes                        # noqa: E402
import eye_geometry as G                                            # noqa: E402

import chase_red_ball as C                                          # noqa: E402

DT = .01
NAMES = ("左眼往右", "左眼往左", "左眼往下", "左眼往上",
         "右眼往右", "右眼往左", "右眼往下", "右眼往上")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bearing", type=float, default=0.5)
    ap.add_argument("--distance", type=float, default=1.5)
    ap.add_argument("--seconds", type=float, default=6.)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    genome = C.load_champion()["genome"]
    body, brain, eyes, geom = C.build(genome, None, seed=args.seed)
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
    group = brain.groups['eye_proprioception']
    per_joint = len(group)//4
    print("eye_proprioception 一共 %d 个，每个关节 %d 个" % (len(group), per_joint))
    print("%6s %8s | %-34s | %s | %s" % ("秒", "头的朝向", "眼睛角度(左右/上下 左眼 右眼)",
                                          "眼肌(>0.2 的)", "位置排里最亮的"))
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
        if step % 50 == 0:
            frame = np.asarray(body.data.xmat[base], dtype=float).reshape(3, 3)
            middle = (np.asarray(body.data.xpos[left_eye], dtype=float)
                      + np.asarray(body.data.xpos[right_eye], dtype=float))/2.
            aim = ball - middle
            heading = float(np.arctan2(float(aim @ frame[:, 1]), float(aim @ frame[:, 0])))
            muscle = np.asarray(brain.network.rates_at(brain.groups['eye_motor']), dtype=float)
            hot = ["%s %.2f" % (NAMES[i], muscle[i]) for i in range(len(muscle)) if muscle[i] > .2]
            banks = []
            for joint in range(4):
                rates = np.asarray(brain.network.rates_at(group[joint*per_joint:(joint+1)*per_joint]))
                banks.append("%d:@%d(%.2f)" % (joint, int(np.argmax(rates)), float(rates.max())))
            angle = np.asarray(brain.eye_command(), dtype=float)
            print("%6.2f %+8.3f | 左眼%+.2f/%+.2f 右眼%+.2f/%+.2f | %-30s | %s"
                  % (step*DT, heading, angle[0], angle[1], angle[2], angle[3],
                     " ".join(hot) or "全灭", " ".join(banks)))
    eyes.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())