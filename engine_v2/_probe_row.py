# -*- coding: utf-8 -*-
"""眼位那一排（左眼 yaw 16 个）到底怎么亮的：球偏在几个角度时，逐个数出来。"""
import sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent
for extra in (ROOT, ROOT / "tools"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))
import mujoco
import chase_red_ball as C
import eye_geometry as G
from born_wired.reflex_senses import ReflexSenses

DT = .01
bearings = [float(v) for v in sys.argv[1].split(",")] if len(sys.argv) > 1 else [0., .2, .4, .6]
seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
spec = C.gaze_spec()
genome = dict(C.load_champion()["genome"])
print("球偏 | 左眼yaw | 眼位那一排（0号最右 … 15号最左）")
for bearing in bearings:
    body, brain, eyes, geom = C.build(genome, None, seed=0, spec=spec)
    base = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_BODY, "base")
    le = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_BODY, "eye_left")
    re = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_BODY, "eye_right")
    body.reset(seed=0, joint_noise=.01)
    G.place_in_front(body.model, body.data, geom, bearing, 0., 1.2)
    ball = np.asarray(body.data.geom_xpos[geom], dtype=float).copy()
    senses = ReflexSenses(body); observation = body.observe(); environment = senses.observe()
    pixels = eyes.observe_raw()
    bank = brain.groups["eye_proprioception"]; per = len(bank)//4
    row = None
    for step in range(int(round(seconds/DT))):
        if getattr(brain, "eye_encoder", None) is not None:
            body.command_eyes(brain.eye_command())
        observation = body.observe()
        if step % 10 == 0:
            pixels = eyes.observe_raw()
        target, activation = brain.step(observation, environment=environment, autonomy=True,
                                        locomotion=0., dt=DT, learn=False, eye_pixels=pixels, startle=0.)
        observation = body.step(target, duration=DT, activation=activation)
        environment = senses.observe()
        row = np.asarray(brain.network.rates_at(bank[0*per:1*per]))
    frame = np.asarray(body.data.xmat[base], dtype=float).reshape(3, 3)
    mid = (np.asarray(body.data.xpos[le]) + np.asarray(body.data.xpos[re]))/2.
    aim = ball - mid
    got = np.degrees(np.arctan2(float(aim @ frame[:,1]), float(aim @ frame[:,0])))
    print("%+5.2f | %+6.1f | %s" % (bearing, np.degrees(body.data.qpos[19]),
          " ".join("%4.2f" % v for v in row)))
    eyes.close()
