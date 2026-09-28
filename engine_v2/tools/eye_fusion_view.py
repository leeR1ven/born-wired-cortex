"""Show what the two eyes see, and the two pictures fused into one.

Display only. Nothing here feeds the brain: the fused panel is the two raw
pictures averaged so that a target the eyes have not converged on shows up as
two ghosts instead of one, which is the quickest way to see the eye loop work.

  python tools/eye_fusion_view.py --distance .6 --gif artifacts/eye_clean.gif
  python tools/eye_fusion_view.py --distance .6 --scenery arena --gif artifacts/eye_arena.gif
  python tools/eye_fusion_view.py --distance .6 --sound 1.2 --gif artifacts/eye_sound.gif

--sound puts a 262 Hz source beside the head and nothing else. That source is
made see-through for the shot, so the retinas really do see an empty room and
anything the eyes do there came down the ears. Without it the box would be a
visible landmark and the eyes could be following that instead.
"""
import argparse
import io
import json
import os
from pathlib import Path
import sys

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from born_wired.binaural_senses import BinauralSenses         # noqa: E402
from born_wired.embodied import (EmbodiedController, EYE_HEIGHT,  # noqa: E402
                                 EYE_WIDTH)
from born_wired.go2_body import Go2Body                       # noqa: E402
from born_wired.stereo_senses import RawEyes                  # noqa: E402

ARENA = ROOT / "models" / "reflex_arena.xml"
SCENERY = ("red_pillar", "blue_box", "front_block", "curb", "low_step", "platform",
           "ramp", "passage_a", "passage_b", "sound_low", "sound_high")
ENVIRONMENT = ("body_touch", "foot_obstacle", "foot_load", "foot_slip")
GAP = 6
BAR = 94


def live_eye_size():
    """The picture size the live window runs, so this view follows the animal."""
    try:
        config = json.loads((ROOT / "live_config.json").read_text(encoding="utf-8-sig"))
    except OSError:
        return EYE_WIDTH, EYE_HEIGHT
    parameters = config.get("parameters", {})
    return int(parameters.get("eye_width", EYE_WIDTH)), int(parameters.get("eye_height", EYE_HEIGHT))


def font(size):
    for name in ("msyh.ttc", "simhei.ttf", "consola.ttf"):
        try:
            return ImageFont.truetype(os.path.join(os.environ.get("WINDIR", r"C:\Windows"),
                                                   "Fonts", name), size)
        except OSError:
            continue
    return ImageFont.load_default()


def fuse(pixels):
    """The two pictures averaged, for display only."""
    return ((pixels[0].astype(np.uint16) + pixels[1].astype(np.uint16))//2).astype(np.uint8)


def sheet(pixels, lines):
    # Wide enough that the longest read-out line is not cut off at the edge,
    # and deep enough for four lines under the label row.
    width = max(3*PANEL[0] + 4*GAP, 1010)
    canvas = Image.new("RGB", (width, PANEL[1] + BAR), (18, 18, 22))
    for index, panel in enumerate((pixels[0], pixels[1], fuse(pixels))):
        image = Image.fromarray(panel).resize(PANEL, Image.NEAREST)
        # A crosshair on the middle of each picture: the ball should end up on it.
        draw = ImageDraw.Draw(image)
        middle = (PANEL[0]//2, PANEL[1]//2)
        draw.line((middle[0], 0, middle[0], PANEL[1]), fill=(255, 40, 40))
        draw.line((0, middle[1], PANEL[0], middle[1]), fill=(255, 40, 40))
        canvas.paste(image, (GAP + index*(PANEL[0] + GAP), 0))
    draw = ImageDraw.Draw(canvas)
    for index, name in enumerate(("left eye", "right eye", "the two fused")):
        draw.text((GAP + index*(PANEL[0] + GAP) + PANEL[0]//2, PANEL[1] + 2), name,
                  (210, 210, 210), font=font(13), anchor="ma")
    for index, line in enumerate(lines):
        draw.text((GAP, PANEL[1] + 18 + 15*index), line, (235, 235, 235), font=font(12))
    return canvas


def build(distance=None, lateral=0., scenery=True, ball=.06, sound=None):
    body = Go2Body(model_path=ARENA)
    if not scenery:
        for name in SCENERY:
            geom = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if geom >= 0:
                body.model.geom_pos[geom] = [60., 60., -8.]
    target = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_GEOM, "green_target")
    if distance is None:
        body.model.geom_pos[target] = [60., 60., -8.]
    else:
        body.model.geom_size[target] = [ball]*3
        body.model.geom_pos[target] = [.30 + distance, lateral, .32]
    for name, where in (("sound_low", sound), ("sound_high", None)):
        geom = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom < 0:
            continue
        body.model.geom_pos[geom] = [.60, where, .30] if where is not None else [60., 60., -8.]
        if name == "sound_low" and where is not None:
            # Display only: the source keeps its place in the sound field but is
            # not painted, so a picture of the eyes cannot be read as the eyes
            # having followed the box instead of the noise.
            body.model.geom_rgba[geom, 3] = 0.
    mujoco.mj_forward(body.model, body.data)
    return body


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distance", type=float, default=.6,
                        help="target distance in metres; a negative value leaves the view empty")
    parser.add_argument("--lateral", type=float, default=0.)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--every", type=int, default=15)
    parser.add_argument("--scenery", choices=("clean", "arena"), default="clean")
    parser.add_argument("--sound", type=float, default=None,
                        help="a 262 Hz source this far to the robot's left (+ is left)")
    parser.add_argument("--pin", action="store_true", default=True)
    parser.add_argument("--gif", default=None)
    parser.add_argument("--png", default=None)
    default_width, default_height = live_eye_size()
    parser.add_argument("--eye-width", type=int, default=default_width,
                        help="raw picture width per eye; defaults to the live window's")
    parser.add_argument("--eye-height", type=int, default=default_height)
    options = parser.parse_args(argv)
    # The panel follows whatever the retina is, and shrinks its magnification when
    # the sheet grows so three of them still fit side by side.
    global PANEL, ZOOM
    ZOOM = max(1, min(3, 1200 // (3*options.eye_width)))
    PANEL = (options.eye_width*ZOOM, options.eye_height*ZOOM)

    distance = None if options.distance is None or options.distance < 0 else options.distance
    body = build(distance,
                 options.lateral, options.scenery == "arena", sound=options.sound)
    brain = EmbodiedController(body.home_angles, body.lower_limits, body.upper_limits,
                               motor_units=20, proprio_units=8, association_units=8,
                               eye_limits=(body.eye_lower_limits, body.eye_upper_limits),
                               eye_width=options.eye_width, eye_height=options.eye_height)
    eyes = RawEyes(body, width=options.eye_width, height=options.eye_height)
    ears = None if options.sound is None else BinauralSenses(body, window_samples=160)
    observation = body.observe()
    environment = {name: np.zeros(4) for name in ENVIRONMENT}
    environment['foot_support'] = np.zeros(4, dtype=bool)
    groups = brain.groups
    frames = []
    for step in range(options.steps):
        pixels = eyes.observe_raw()
        target, activation = brain.step(observation, environment=environment,
                                        eye_pixels=pixels,
                                        ear_waveform=(None if ears is None else ears.observe()),
                                        dt=.01, learn=False)
        command = brain.eye_command()
        body.command_eyes(command)
        observation = body.step(np.asarray(body.home_angles), duration=.01,
                                activation=activation)
        if step % options.every == 0 or step == options.steps - 1:
            rates = brain.network.activity
            angle = float(rates[groups['eye_convergence_in']][0]
                          - rates[groups['eye_convergence_out']][0])
            bank = rates[groups['eye_distance']]
            need = ((0., 0.) if distance is None else
                    (-np.degrees(np.arctan2(.055, distance)),
                     +np.degrees(np.arctan2(.055, distance))))
            frames.append(sheet(pixels, [
                f"t = {body.data.time:5.2f}s    target "
                + ("nothing in view" if distance is None else f"{distance:.2f} m")
                + f"    scenery: {options.scenery}"
                + ("" if options.sound is None else
                   f"    sound 262 Hz {abs(options.sound):.2f} m to the "
                   + ("left" if options.sound > 0 else "right")),
                f"eye command   yaw L {command[0]:+.3f}  R {command[2]:+.3f}   "
                f"pitch {command[1]:+.3f}   |   it should be {need[0]:+.1f} / {need[1]:+.1f} deg",
                f"angle between the eyes {angle:+.3f}   distance cells on "
                + str(int((bank > .02).sum())) + " of 7   both eyes look "
                + ("LEFT" if (command[0] + command[2])/2. > .02 else
                   "RIGHT" if (command[0] + command[2])/2. < -.02 else "straight ahead"),
                "distance cells " + " ".join(f"{value:.2f}" for value in bank),
            ]))
    eyes.close()
    if options.gif:
        Path(options.gif).parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(options.gif, save_all=True, append_images=frames[1:], duration=120, loop=0)
        print("wrote", options.gif, len(frames), "frames")
    if options.png:
        stack = Image.new("RGB", (frames[-1].width, frames[-1].height*len(frames)), (10, 10, 12))
        for index, frame in enumerate(frames):
            stack.paste(frame, (0, index*frame.height))
        Path(options.png).parent.mkdir(parents=True, exist_ok=True)
        stack.save(options.png)
        print("wrote", options.png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())