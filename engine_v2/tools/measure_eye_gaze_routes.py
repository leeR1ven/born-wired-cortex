"""Measure where the eyes look while a ball sweeps past, with and without the
motion-onset route and with and without the competitive cell under it.

    python tools/measure_eye_gaze_routes.py

Four scenes: a clean room with the ball sweeping slowly, a clean room with the
ball held still, a clean room with a faster sweep, and the furnished arena with
the slow sweep. The number printed is the mean absolute angle between the gaze
and the true bearing of the ball, in radians, over the last 230 of 250 steps.
Nothing here is on the control path; it only reads the eye muscles.
"""
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from born_wired.embodied import EmbodiedController      # noqa: E402
from born_wired.go2_body import Go2Body                 # noqa: E402
from born_wired.stereo_senses import RawEyes            # noqa: E402

ARENA = ROOT / "models" / "reflex_arena.xml"
SCENERY = ("red_pillar", "blue_box", "front_block", "curb", "low_step", "platform",
           "ramp", "passage_a", "passage_b", "sound_low", "sound_high")
ENVIRONMENT = ("body_touch", "foot_obstacle", "foot_load", "foot_slip")
DT, BALL = .01, .06


def sweep_gaze(steps=250, distance=.90, sweep=.30, scenery=True, **parameters):
    body = Go2Body(model_path=ARENA)
    if not scenery:
        for name in SCENERY:
            geom = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if geom >= 0:
                body.model.geom_pos[geom] = [60., 60., -8.]
    target = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_GEOM, "green_target")
    body.model.geom_size[target] = [BALL]*3
    brain = EmbodiedController(body.home_angles, body.lower_limits, body.upper_limits,
                               motor_units=20, proprio_units=8, association_units=8,
                               eye_limits=(body.eye_lower_limits, body.eye_upper_limits),
                               **parameters)
    eyes = RawEyes(body)
    observation = body.observe()
    environment = {name: np.zeros(4) for name in ENVIRONMENT}
    environment["foot_support"] = np.zeros(4, dtype=bool)
    base = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_BODY, "base")
    errors = []
    for step in range(steps):
        lateral = sweep - 2.*sweep*step/float(steps - 1)
        body.model.geom_pos[target] = [.30 + distance, lateral, .32]
        mujoco.mj_forward(body.model, body.data)
        activation = brain.step(observation, environment=environment,
                                eye_pixels=eyes.observe_raw(), dt=DT, learn=False)[1]
        body.command_eyes(brain.eye_command())
        offset = np.asarray(body.data.geom_xpos[target]) - np.asarray(body.data.xpos[base])
        rotation = np.asarray(body.data.xmat[base]).reshape(3, 3)
        bearing = np.arctan2(offset @ rotation[:, 1], offset @ rotation[:, 0])
        command = brain.eye_command()
        if step > 20:
            errors.append(.5*(command[0] + command[2]) - bearing)
        observation = body.step(np.asarray(body.home_angles), duration=DT, activation=activation)
    eyes.close()
    return float(np.abs(np.array(errors)).mean())


def main():
    configs = (("only the picture (motion route off)", dict(eye_change_gain=0.)),
               ("the paper route, no competition", dict(eye_change_common=1e-6)),
               ("plus one competitive cell (default)", dict()))
    scenes = (("clean slow sweep", dict(scenery=False, sweep=.30)),
              ("clean still ball", dict(scenery=False, sweep=0.)),
              ("clean fast sweep", dict(scenery=False, sweep=.60)),
              ("furnished arena", dict(scenery=True, sweep=.30)))
    print("mean |gaze - bearing of the ball| in rad, 250 steps, ball 0.90 m ahead")
    print("%-36s %13s %13s %13s %13s" % ("config", *[name for name, _ in scenes]))
    for label, parameters in configs:
        row = [sweep_gaze(**scene, **parameters) for _, scene in scenes]
        print("%-36s %13.4f %13.4f %13.4f %13.4f" % (label, *row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
