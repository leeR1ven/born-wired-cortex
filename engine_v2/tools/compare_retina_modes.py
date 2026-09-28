"""Same scene, same eye size, different sampling density - for looking at.

  python tools/compare_retina_modes.py --gains 1 2.5 4 --png artifacts/retina_modes.png

Each row is the left retina of the same animal on the same frame, drawn from
RawEyes with a different `centre_gain`: 1 lays the cells on equal slices of the
view, higher values put that many times more cells on a degree in the middle
than at the rim. Display only; nothing here feeds the brain.
"""
import argparse
from pathlib import Path
import sys

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from born_wired.go2_body import Go2Body          # noqa: E402
from born_wired.stereo_senses import RawEyes     # noqa: E402

ARENA = ROOT / "models" / "reflex_arena.xml"


def font(size):
    try:
        return ImageFont.truetype(r"C:\Windows\Fonts\consola.ttf", size)
    except OSError:
        return ImageFont.load_default()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gains", type=float, nargs="+", default=[1., 2.5, 4.])
    parser.add_argument("--eye-width", type=int, default=120)
    parser.add_argument("--eye-height", type=int, default=90)
    parser.add_argument("--distance", type=float, default=.6)
    parser.add_argument("--zoom", type=int, default=2)
    parser.add_argument("--png", default=str(ROOT / "artifacts" / "retina_modes.png"))
    options = parser.parse_args(argv)

    body = Go2Body(model_path=ARENA)
    target = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_GEOM, "green_target")
    body.model.geom_size[target] = [.06]*3
    body.model.geom_pos[target] = [.30 + options.distance, 0., .32]
    mujoco.mj_forward(body.model, body.data)

    rows, labels = [], []
    for gain in options.gains:
        eyes = RawEyes(body, width=options.eye_width, height=options.eye_height,
                       centre_gain=gain)
        picture = eyes.observe_raw()[0]
        eyes.close()
        image = Image.fromarray(picture).resize(
            (options.eye_width*options.zoom, options.eye_height*options.zoom), Image.NEAREST)
        draw = ImageDraw.Draw(image)
        draw.text((6, 4), f"centre_gain {gain:g}", (255, 255, 60), font=font(16))
        rows.append(image)
        labels.append(gain)
        # How many drawn pixels one cell answers for, middle versus rim.
        print(f"centre_gain {gain:g}: kept {eyes.width}x{eyes.height} cells", flush=True)

    width = rows[0].width
    canvas = Image.new("RGB", (width, sum(r.height for r in rows) + 6*(len(rows)-1)), (12, 12, 14))
    top = 0
    for row in rows:
        canvas.paste(row, (0, top))
        top += row.height + 6
    Path(options.png).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(options.png)
    print("wrote", options.png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
