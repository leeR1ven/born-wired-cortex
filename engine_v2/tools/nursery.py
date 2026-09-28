"""Can a keeper teach this animal something it was not born with?

    python tools/nursery.py                    # both arms, the live window's own parameters
    python tools/nursery.py --arm taught --lesson 40

The animal walks. A 262 Hz tone means nothing to it: nothing in the innate
graph carries that band to the retreat cells. The keeper wants "that sound
means back off". The lesson is a keeper's, not a critic's:

  * the tone is on and the keeper has a hand on the body;
  * the hand does two things at once, and only through innate wiring: it drives
    the touch cells, which already reach the retreat cells - the animal is
    being held, and the cells that back it away are firing - and it drives the
    dopamine cell through touch_dopamine_gain;
  * dopamine scales the local rule at the cells that are firing together, so
    that moment is written harder. It carries no answer about right or wrong,
    and it is one ordinary cell among the others.

What gets written is only the backup route named at build time
(cochlea -> retreat): edges laid down at a negligible weight that do nothing
at all until something writes them. The keeper names two populations as worth
connecting; it does not name a cell, a weight or a sign of effect.

Three arms, same seed, same lesson length:

  taught   - the keeper's hand is there, dopamine released
  ignored  - the tone plays just as often; nobody ever touches the animal
  ungated  - the hand is there, but dopamine_gain is zero: the local rule runs
             at its own rate all the time, lesson or not

Read: the retreat cells and the leg angles with the tone alone, no hand; and
the weights the route holds before and after the lesson.
"""
import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from born_wired.binaural_senses import BinauralSenses   # noqa: E402
from born_wired.embodied import EmbodiedController      # noqa: E402
from born_wired.go2_body import Go2Body                 # noqa: E402
from born_wired.stereo_senses import RawEyes            # noqa: E402

ARENA = ROOT / "models" / "reflex_arena.xml"
CONFIG = ROOT / "live_config.json"
ENVIRONMENT = ("body_touch", "foot_obstacle", "foot_load", "foot_slip")
DT = .01
TONE = {"geom": "sound_low", "frequency": 262., "amplitude": 1.0}
WALK = .6                       # the keeper's other hand: keep it walking
HAND = 1.                        # a firm grip, in reflex_senses' own units
PERIOD = 2.                     # one lesson cycle: 1 s held, 1 s let go
THIGHS = (1, 4, 7, 10)


def build(seed, *, hand_teaches, gated, routes=True):
    """One animal. hand_teaches wires the hand to the touch cells and to dopamine."""
    body = Go2Body(model_path=ARENA)
    parameters = dict(json.loads(CONFIG.read_text(encoding="utf-8-sig"))["parameters"])
    parameters.update(seed=seed,
                      backup_routes=((("cochlea", "retreat"),) if routes else ()),
                      dopamine_gain=.9 if gated else 0.,
                      touch_dopamine_gain=4. if hand_teaches else 0.)
    brain = EmbodiedController(body.home_angles, body.lower_limits, body.upper_limits,
                               eye_limits=(body.eye_lower_limits, body.eye_upper_limits),
                               **parameters)
    eyes = RawEyes(body, width=brain.eye_width, height=brain.eye_height)
    ears = BinauralSenses(body, window_samples=160, emitters=(TONE,))
    return body, brain, eyes, ears


class Run:
    def __init__(self, seed, *, hand_teaches, gated, routes=True):
        self.body, self.brain, self.eyes, self.ears = build(
            seed, hand_teaches=hand_teaches, gated=gated, routes=routes)
        self.silence = np.zeros((2, self.ears.window_samples))
        self.observation = self.body.reset(seed=seed)
        self.environment = {name: np.zeros(4) for name in ENVIRONMENT}
        self.environment["foot_support"] = np.zeros(4, dtype=bool)
        self.base = mujoco.mj_name2id(self.body.model, mujoco.mjtObj.mjOBJ_BODY, "base")

    def phase(self, seconds, *, tone, walk, hand=False, learn=True):
        start = float(self.body.data.xpos[self.base][0])
        low, upright = 1., 1.
        seen = {"retreat": [], "retreat_phase": [], "brake": [], "dopamine": [], "cochlea": [],
                "walk_cell": []}
        thighs = []
        for step in range(int(round(seconds/DT))):
            held = hand and (step*DT % PERIOD) < PERIOD/2
            self.environment["body_touch"] = np.full(4, HAND) if held else np.zeros(4)
            for name in ("foot_obstacle", "foot_load", "foot_slip"):
                self.environment[name] = np.zeros(4)
            target, activation = self.brain.step(
                self.observation, environment=self.environment,
                eye_pixels=self.eyes.observe_raw(),
                ear_waveform=self.ears.observe() if tone else self.silence,
                dt=DT, learn=learn, locomotion=walk)
            self.body.command_eyes(self.brain.eye_command())
            self.observation = self.body.step(target, duration=DT, activation=activation)
            if step % 10 == 0:
                rates = self.brain.network.activity
                for name, key in (("retreat", "retreat"), ("brake", "brake"),
                                  ("dopamine", "dopamine"), ("locomotion", "walk_cell")):
                    seen[key].append(float(rates[self.brain.groups[name][0]]))
                seen["retreat_phase"].append(float(np.mean(rates[self.brain.groups["retreat_phase"]])))
                seen["cochlea"].append(float(rates[self.brain.groups["cochlea"]].max()))
                thighs.append([float(target[j]) for j in THIGHS])
            if step*DT > 1.:
                low = min(low, float(self.body.data.xpos[self.base][2]))
                upright = min(upright, float(self.body.data.xmat[self.base].reshape(3, 3)[2, 2]))
        report = {"meters": float(self.body.data.xpos[self.base][0]) - start,
                  "min_height": low, "min_up_z": upright,
                  "thighs": np.round(np.mean(thighs, axis=0), 4).tolist()}
        for name, values in seen.items():
            report[name] = float(np.mean(values)) if values else 0.
        return report

    def close(self):
        self.eyes.close()


def arm(seed, *, hand_teaches, gated, lesson, routes=True):
    run = Run(seed, hand_teaches=hand_teaches, gated=gated, routes=routes)
    brain = run.brain
    born = brain.nursery_state()["weights"]
    before = run.phase(6., tone=True, walk=WALK, learn=False)
    taught = run.phase(lesson, tone=True, walk=WALK, hand=hand_teaches, learn=True)
    after_lesson = brain.nursery_state()["weights"]
    tone = run.phase(6., tone=True, walk=WALK, learn=False)
    silence = run.phase(6., tone=False, walk=WALK, learn=False)
    left_leg = run.phase(6., tone=True, walk=WALK, learn=False)   # repeat: is it stable?
    run.close()
    return dict(before=before, lesson=taught, tone=tone, silence=silence, left_leg=left_leg,
                weights={"born": born, "after lesson": after_lesson})


ARMS = {
    "taught": dict(hand_teaches=True, gated=True, routes=True),
    "ignored": dict(hand_teaches=False, gated=True, routes=True),
    "ungated": dict(hand_teaches=True, gated=False, routes=True),
    "no route": dict(hand_teaches=True, gated=True, routes=False),
}


def show(name, seed, result):
    print(f"--- {name} (seed {seed}) ---")
    for label in ("before", "lesson", "tone", "silence", "left_leg"):
        r = result[label]
        print("  %-9s %7.3f m  retreat %5.3f  leg-drive %5.3f  brake %5.3f  walk %5.3f"
              "  dopamine %5.3f  cochlea %5.3f  min_h %5.3f  up_z %5.3f"
              % (label, r["meters"], r["retreat"], r["retreat_phase"], r["brake"],
                 r["walk_cell"], r["dopamine"], r["cochlea"], r["min_height"], r["min_up_z"]))
    print("  thighs with the tone alone: %s" % result["tone"]["thighs"])
    print("  thighs in silence:          %s" % result["silence"]["thighs"])
    for label, weights in result["weights"].items():
        print("  route %-13s %s" % (label, np.round(weights, 4).tolist()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", default="taught", choices=tuple(ARMS)+("all",))
    parser.add_argument("--lesson", type=float, default=40.)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    chosen = list(ARMS) if args.arm == "all" else [args.arm]
    print(f"262 Hz tone, {args.lesson:.0f} s lesson, the live window's own parameters")
    output = {}
    for name in chosen:
        result = arm(args.seed, lesson=args.lesson, **ARMS[name])
        show(name, args.seed, result)
        output[name] = result
    if args.json:
        Path(args.json).write_text(json.dumps(output, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
