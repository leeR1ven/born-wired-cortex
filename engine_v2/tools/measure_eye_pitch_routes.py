"""Measure where the eyes look while a ball sweeps UP or DOWN past them.

    python tools/measure_eye_pitch_routes.py

Clean room only. The ball sits 0.90 m ahead and travels from 0.30 m above the
body to 0.30 m below it (or the reverse) over 250 steps. Printed is the mean
signed eye pitch of the two eyes over the last 230 steps, next to the elevation
of the ball, in radians. Nothing here is on the control path; it only reads the
eye muscles.
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


def sweep_pitch(steps=250, distance=.90, rise=.30, scenery=False, **parameters):
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
    pitches, true_elevations = [], []
    for step in range(steps):
        height = rise - 2.*rise*step/float(steps - 1)
        body.model.geom_pos[target] = [.30 + distance, 0., .32 + height]
        mujoco.mj_forward(body.model, body.data)
        activation = brain.step(observation, environment=environment,
                                eye_pixels=eyes.observe_raw(), dt=DT, learn=False)[1]
        body.command_eyes(brain.eye_command())
        offset = np.asarray(body.data.geom_xpos[target]) - np.asarray(body.data.xpos[base])
        rotation = np.asarray(body.data.xmat[base]).reshape(3, 3)
        elevation = np.arctan2(offset @ rotation[:, 2], offset @ rotation[:, 0])
        command = brain.eye_command()
        if step > 20:
            pitches.append(.5*(command[1] + command[3]))
            true_elevations.append(elevation)
        observation = body.step(np.asarray(body.home_angles), duration=DT, activation=activation)
    eyes.close()
    return np.array(pitches), np.array(true_elevations)


def main():
    print("mean signed eye pitch (rad) while the ball sweeps down the picture")
    print("%-38s %10s %10s %10s" % ("config", "ball high", "ball level", "ball low"))
    for label, parameters in (("motion route off", dict(eye_change_gain=0.)),
                              ("left/right only (row gain 0)", dict(eye_row_relay_gain=0.)),
                              ("both axes (default)", dict())):
        row = []
        for offset in (.30, 0., -.30):
            pitches, elevations = sweep_pitch(rise=offset, **parameters)
            row.append((float(pitches.mean()), float(elevations.mean())))
        print("%-38s %10.4f %10.4f %10.4f" % (label, *[p for p, _ in row]))
        print("%-38s %10.4f %10.4f %10.4f" % ("  (ball elevation)", *[e for _, e in row]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
