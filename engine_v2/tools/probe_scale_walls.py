"""Which rules, not which hardware, stop the model from growing?

    python tools/probe_scale_walls.py

tools/probe_model_scale.py measures what a bigger cortex costs.  This script
answers the other half of the question: which builds simply refuse, and which
rule refuses them.  Every attempt prints the exception verbatim, so that a wall
can be reported as a named rule in a named file rather than as "it got slow".

Nothing here changes the model: each line builds a brain, times nothing, and
throws it away.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from born_wired.go2_body import Go2Body
from born_wired.embodied import EmbodiedController
from born_wired.stereo_senses import RawEyes

MODEL = str(ROOT / "models" / "reflex_arena.xml")
BASE = dict(eye_width=192, eye_height=144, motor_units=1600,
            proprio_units=512, association_units=2048)


def attempt(body, label, **over):
    """Build one brain at the shipped configuration with `over` applied."""
    params = dict(BASE)
    params.update(over)
    try:
        brain = EmbodiedController(body.home_angles, body.lower_limits,
                                  body.upper_limits,
                                  eye_limits=(body.eye_lower_limits,
                                              body.eye_upper_limits),
                                  **params)
    except Exception as exc:
        print("%-46s REFUSED   %s: %s" % (label, type(exc).__name__, exc))
        return None
    print("%-46s ok        %8d cells %9d edges"
          % (label, brain.network.n_neurons, len(brain.synapses.src)))
    return brain


def attempt_render(body, label, width, height):
    """Ask the eyes for one frame and see whether the arena can give it."""
    eye = None
    try:
        eye = RawEyes(body, width=width, height=height)
        frame = eye.observe_raw()
        print("%-46s ok        frame %s" % (label, tuple(frame.shape)))
    except Exception as exc:
        print("%-46s REFUSED   %s: %s" % (label, type(exc).__name__, exc))
    finally:
        if eye is not None:
            eye.close()


def main():
    body = Go2Body(MODEL)

    print("=" * 100)
    print("what the arena declares for the offscreen buffer the eyes are drawn into")
    text = (ROOT / "models" / "reflex_arena.xml").read_text(encoding="utf-8")
    for tag in ("offwidth", "offheight"):
        hit = re.search(tag + r'="(\d+)"', text)
        print("  %-10s %s" % (tag, hit.group(1) if hit else "absent"))

    print("=" * 100)
    print("wall 1: a region's own size, against the incoming excitation budget")
    for proprio in (512, 640, 768, 896, 1024):
        attempt(body, "192x144 proprio_units=%d" % proprio, proprio_units=proprio)
    for association in (2048, 3072, 4096):
        attempt(body, "192x144 association_units=%d" % association,
                association_units=association)
    for motor in (1600, 3200):
        attempt(body, "192x144 motor_units=%d" % motor, motor_units=motor)

    print("=" * 100)
    print("wall 2: the width of the eye, against the offscreen buffer")
    for width, height in ((256, 192), (320, 240), (384, 288), (448, 336), (512, 384)):
        attempt_render(body, "eye %dx%d" % (width, height), width, height)

    print("=" * 100)
    print("wall 3: the whole animal at once")
    attempt(body, "320x240 2400/768/3072", eye_width=320, eye_height=240,
            motor_units=2400, proprio_units=768, association_units=3072)
    attempt(body, "320x240 2400/512/3072", eye_width=320, eye_height=240,
            motor_units=2400, proprio_units=512, association_units=3072)


if __name__ == "__main__":
    main()
