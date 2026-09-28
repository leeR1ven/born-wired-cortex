"""Can the eyes tell how far away a striped wall is?

    python tools/probe_wall_distance.py --png artifacts/wall_distance.png

The robot is parked at a range of distances straight in front of the east wall
of the arena and the room is otherwise emptied. Nothing here is on the control
path: the only thing being read is what the visual cells say about the wall.

The wall is covered in stripes of several different widths (tools/
build_arena.py) because a flat wall has no contrast at all, so the visual cells
had nothing to look at and no answer to give about it.
"""
import argparse
import os
from pathlib import Path
import sys

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from born_wired.embodied import EmbodiedController      # noqa: E402
from born_wired.go2_body import Go2Body                 # noqa: E402
from born_wired.stereo_senses import RawEyes            # noqa: E402

ARENA = ROOT / "models" / "reflex_arena.xml"
SCENERY = ("red_pillar", "blue_box", "front_block", "left_block", "right_block",
           "green_target", "curb", "low_step", "platform", "ramp", "passage_a",
           "passage_b", "sound_low", "sound_high")
ENVIRONMENT = ("body_touch", "foot_obstacle", "foot_load", "foot_slip")
WALL_FACE = 3.5 - .08
RANGES = (0.30, 0.50, 0.80, 1.20, 1.80, 2.50, 3.20)


def font(size):
    for name in ("msyh.ttc", "consola.ttf"):
        try:
            return ImageFont.truetype(str(Path(os.environ.get("WINDIR", r"C:\Windows"))
                                          / "Fonts" / name), size)
        except OSError:
            continue
    return ImageFont.load_default()


def park(body, distance):
    start = body._base_qpos
    body.data.qpos[start] = WALL_FACE - float(distance) - .30
    body.data.qpos[start + 1] = 0.
    mujoco.mj_forward(body.model, body.data)


def probe(distance, steps=200, spinup=60, **parameters):
    body = Go2Body(model_path=ARENA)
    for name in SCENERY:
        geom = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom >= 0:
            body.model.geom_pos[geom] = [60., 60., -8.]
    brain = EmbodiedController(body.home_angles, body.lower_limits, body.upper_limits,
                               motor_units=20, proprio_units=8, association_units=8,
                               eye_limits=(body.eye_lower_limits, body.eye_upper_limits),
                               **parameters)
    eyes = RawEyes(body)
    environment = {name: np.zeros(4) for name in ENVIRONMENT}
    environment["foot_support"] = np.zeros(4, dtype=bool)
    picture = None
    bank = []
    pool = []
    near = []
    brake = []
    observation = body.observe()
    for step in range(steps):
        park(body, distance)
        pixels = eyes.observe_raw()
        if step == spinup and picture is None:
            picture = pixels.copy()
        activation = brain.step(observation, environment=environment,
                                eye_pixels=pixels, dt=.01, learn=False)[1]
        body.command_eyes(brain.eye_command())
        observation = body.step(np.asarray(body.home_angles), duration=.01, activation=activation)
        if step >= spinup:
            diag = brain.diagnostics()
            reflex = diag["reflex_activity"]
            bank.append(float(np.sum(diag["eye_distance"])))
            pool.append(float(np.mean(diag["binocular_population_activity"][2:])))
            near.append(float(np.mean(reflex["near"][1])))
            brake.append(float(np.mean(reflex["brake"])))
    eyes.close()
    return dict(bank=float(np.mean(bank)), pool=float(np.mean(pool)),
                near=float(np.mean(near)), brake=float(np.mean(brake)), picture=picture)


def contact_sheet(pictures, lines):
    zoom = 5
    panel = (32*zoom, 24*zoom)
    gap = 6
    rows = len(pictures)
    canvas = Image.new("RGB", (len(pictures[0])*panel[0] + gap*(len(pictures[0]) + 1),
                               rows*(panel[1] + 20) + gap), (18, 18, 22))
    draw = ImageDraw.Draw(canvas)
    for row, (picture, label) in enumerate(zip(pictures, lines)):
        top = row*(panel[1] + 20) + gap
        for eye in range(len(picture)):
            image = Image.fromarray(picture[eye]).resize(panel, Image.NEAREST)
            x = gap + eye*(panel[0] + gap)
            canvas.paste(image, (x, top))
            draw.line((x + panel[0]//2, top, x + panel[0]//2, top + panel[1]), fill=(255, 40, 40))
        draw.text((gap, top + panel[1] + 2), label, (235, 235, 235), font=font(12))
    return canvas


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--png", default=None)
    parser.add_argument("--steps", type=int, default=200)
    options = parser.parse_args(argv)
    print("standing in front of the striped east wall, room otherwise empty")
    print("bank = summed distance cells (0 parallel, 7 closest); pool = the")
    print("disparity cells that agree at a large offset")
    print("%8s %8s %8s %7s %7s %9s %9s" % ("range", "grey", "pool", "bank", "near", "brake", "contrast"))
    pictures = []
    labels = []
    for distance in RANGES:
        values = probe(distance, steps=options.steps)
        picture = values.pop("picture")
        grey = picture[..., :3].mean(axis=3)
        spread = float(np.mean(np.abs(np.diff(grey, axis=2))))
        print("%8.2f %8.3f %8.4f %7.3f %7.3f %9.3f %9.4f" % (
            distance, float(grey.mean()), values["pool"],
            values["bank"], values["near"], values["brake"], spread))
        if picture is not None:
            pictures.append(picture)
            labels.append(f"{distance:.2f} m   pool {values['pool']:.4f}  "
                          f"bank {values['bank']:.3f}  near {values['near']:.3f}  "
                          f"brake {values['brake']:.3f}")
    if options.png and pictures:
        output = Path(options.png)
        output.parent.mkdir(parents=True, exist_ok=True)
        contact_sheet(pictures, labels).save(output)
        print("wrote", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
