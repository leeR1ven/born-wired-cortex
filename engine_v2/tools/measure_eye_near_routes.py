"""What do the eyes tell the protective cells, and how far away?

    python tools/measure_eye_near_routes.py

A clean room with one flat surface straight ahead at a range of distances. The
only thing that changes between rows is which of the two visual voices into the
protective cells is wired: the angle between the two eyes (the distance bank)
and the disparity cells beside it. Nothing here is on the control path.
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
RANGES = (0.30, 0.50, 0.90, 1.50, 3.00)


def probe(range_m, steps=300, **parameters):
    body = Go2Body(model_path=ARENA)
    for name in SCENERY:
        geom = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom >= 0:
            body.model.geom_pos[geom] = [60., 60., -8.]
    target = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_GEOM, "green_target")
    body.model.geom_size[target] = [.60]*3
    brain = EmbodiedController(body.home_angles, body.lower_limits, body.upper_limits,
                               motor_units=20, proprio_units=8, association_units=8,
                               eye_limits=(body.eye_lower_limits, body.eye_upper_limits),
                               **parameters)
    eyes = RawEyes(body)
    observation = body.observe()
    environment = {name: np.zeros(4) for name in ENVIRONMENT}
    environment["foot_support"] = np.zeros(4, dtype=bool)
    for _ in range(steps):
        body.model.geom_pos[target] = [range_m, 0., .32]
        mujoco.mj_forward(body.model, body.data)
        activation = brain.step(observation, environment=environment,
                                eye_pixels=eyes.observe_raw(), dt=.01, learn=False)[1]
        body.command_eyes(brain.eye_command())
        observation = body.step(np.asarray(body.home_angles), duration=.01, activation=activation)
    eyes.close()
    diag = brain.diagnostics()
    reflex = diag["reflex_activity"]
    return dict(near=float(np.mean(reflex["near"][1])), brake=float(np.mean(reflex["brake"])),
                avoid=float(np.mean(reflex["avoidance"])), bank=float(np.sum(diag["eye_distance"])),
                pool=float(np.mean(diag["binocular_population_activity"][2:])))


def approach(steps=400, **parameters):
    """A small ball coming in from 2.0 m to 0.25 m, read as it arrives."""
    body = Go2Body(model_path=ARENA)
    for name in SCENERY:
        geom = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom >= 0:
            body.model.geom_pos[geom] = [60., 60., -8.]
    target = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_GEOM, "green_target")
    body.model.geom_size[target] = [.06]*3
    brain = EmbodiedController(body.home_angles, body.lower_limits, body.upper_limits,
                               motor_units=20, proprio_units=8, association_units=8,
                               eye_limits=(body.eye_lower_limits, body.eye_upper_limits),
                               **parameters)
    eyes = RawEyes(body)
    observation = body.observe()
    environment = {name: np.zeros(4) for name in ENVIRONMENT}
    environment["foot_support"] = np.zeros(4, dtype=bool)
    rows = []
    for step in range(steps):
        range_m = 2.0 - 1.75*step/float(steps - 1)
        body.model.geom_pos[target] = [range_m, 0., .32]
        mujoco.mj_forward(body.model, body.data)
        activation = brain.step(observation, environment=environment,
                                eye_pixels=eyes.observe_raw(), dt=.01, learn=False)[1]
        body.command_eyes(brain.eye_command())
        observation = body.step(np.asarray(body.home_angles), duration=.01, activation=activation)
        if step % 50 == 0 or step == steps - 1:
            diag = brain.diagnostics()
            reflex = diag["reflex_activity"]
            rows.append((range_m, float(np.sum(diag["eye_distance"])),
                         float(np.mean(reflex["near"][1])), float(np.mean(reflex["brake"]))))
    eyes.close()
    return rows


def main():
    print("a small ball coming in from 2.0 m to 0.25 m, clean room")
    print("%-28s %6s %6s %7s %7s" % ("voice", "range", "bank", "near", "brake"))
    for label, parameters in (("both (live)", dict()),
                              ("disparity cells only", dict(eye_near_gain=0.)),
                              ("angle between the eyes", dict(eye_stereo_gain=0.))):
        for range_m, bank, near, brake in approach(**parameters):
            print("%-28s %6.2f %6.2f %7.3f %7.3f" % (label, range_m, bank, near, brake))
        print()
    print("a flat surface straight ahead in a clean room, read after 3 s")
    print("bank = summed distance cells (0 parallel, 7 closest)")
    print("%-28s %6s %8s %7s %7s %7s %7s" % ("voice", "range", "near", "brake", "avoid", "bank", "pool"))
    voices = (("both (live)", dict()),
              ("angle between the eyes", dict(eye_stereo_gain=0.)),
              ("disparitycells only", dict(eye_near_gain=0.)),
              ("neither", dict(eye_near_gain=0., eye_stereo_gain=0.)))
    for label, parameters in voices:
        for range_m in RANGES:
            values = probe(range_m, **parameters)
            print("%-28s %6.2f %8.3f %7.3f %7.3f %7.3f %7.3f" % (
                label, range_m, values["near"], values["brake"], values["avoid"],
                values["bank"], values["pool"]))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
