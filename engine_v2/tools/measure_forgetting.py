"""Does a lesson survive a long life, and does it survive a second lesson?

    python tools/measure_forgetting.py --gap 60 --content quiet
    python tools/measure_forgetting.py --gap 60 --content second

tools/nursery.py teaches one thing and asks about it at once.  The exam bank's
two "later" questions ask again after ten and eight seconds.  Neither answers
the question a keeper actually has: if the animal goes on living -- learning the
whole time -- is the old lesson still there?

This tool teaches one tone meaning (262 Hz means back off), then lets the animal
live through a gap with learning on, then asks about the first tone again.  Four
ways to spend the gap:

  quiet   nothing is played; it walks (learning runs the whole time)
  tone    the tone keeps playing but nobody touches it: no teacher, no reward
  second  a second tone (880 Hz) is taught the same meaning through the same
          route, so the two lessons compete for the same cells
  life    first tone, second tone and silence take turns while it walks

Read at every checkpoint: the retreat cells and the leg drive with the tone
alone (the bank's own read-out), and the weights of the teachable route, which
is the memory itself rather than its effect.  Nothing here writes a weight: the
only thing that changes the route is the local rule, driven by dopamine through
the keeper's hand.  The hand is the keeper's, and it does what a hand does --
it touches the body -- so the lesson is the touch, not a number set by this
script.
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
from tools import nursery                               # noqa: E402

DT = nursery.DT
WALK, HAND, PERIOD = nursery.WALK, nursery.HAND, nursery.PERIOD
ENVIRONMENT = nursery.ENVIRONMENT
TONE_A = dict(nursery.TONE)                                     # 262 Hz, sound_low
TONE_B = {"geom": "sound_high", "frequency": 880., "amplitude": 1.0}
TONES = {"a": TONE_A, "b": TONE_B}


class Life:
    """One animal, kept alive across phases so that its weights carry over."""

    def __init__(self, seed, parameters=None):
        body, brain, eyes, _ = nursery.build(seed, hand_teaches=True, gated=True,
                                             routes=True, parameters=parameters)
        self.body, self.brain, self.eyes = body, brain, eyes
        self.ears = {name: BinauralSenses(body, window_samples=160, emitters=(tone,))
                     for name, tone in TONES.items()}
        self.silence = np.zeros((2, 160))
        self.observation = body.observe()
        self.environment = {name: np.zeros(4) for name in ENVIRONMENT}
        self.base = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_BODY, "base")

    def phase(self, seconds, *, tone=None, walk=WALK, hand=False, learn=True, label=""):
        ears = self.ears.get(tone) if tone else None
        seen = {"retreat": [], "brake": [], "dopamine": [], "cochlea": []}
        thighs, low, upright = [], 1., 1.
        for step in range(int(round(seconds/DT))):
            held = hand and (step*DT % PERIOD) < PERIOD/2
            self.environment["body_touch"] = np.full(4, HAND) if held else np.zeros(4)
            for name in ("foot_obstacle", "foot_load", "foot_slip"):
                self.environment[name] = np.zeros(4)
            target, activation = self.brain.step(
                self.observation, environment=self.environment,
                eye_pixels=self.eyes.observe_raw(),
                ear_waveform=ears.observe() if ears is not None else self.silence,
                dt=DT, learn=learn, locomotion=walk)
            self.body.command_eyes(self.brain.eye_command())
            self.observation = self.body.step(target, duration=DT, activation=activation)
            if step % 10 == 0:
                rates = self.brain.network.activity
                for name in seen:
                    seen[name].append(float(np.mean(rates[self.brain.groups[name]])))
                thighs.append([float(target[j]) for j in nursery.THIGHS])
            if step*DT > 1.:
                low = min(low, float(self.body.data.xpos[self.base][2]))
                upright = min(upright, float(self.body.data.xmat[self.base].reshape(3, 3)[2, 2]))
        report = {"label": label, "seconds": float(seconds), "tone": tone,
                  "learn": bool(learn), "hand": bool(hand),
                  "min_height": low, "min_up_z": upright,
                  "thighs": np.round(np.mean(thighs, axis=0), 4).tolist(),
                  "route": np.round(self.brain.nursery_state()["weights"], 4).tolist()}
        for name, values in seen.items():
            report[name] = float(np.mean(values)) if values else 0.
        return report

    def close(self):
        self.eyes.close()


def teach(gap_seconds, content, lesson_seconds=20., probe_seconds=4., block=20.,
          parameters=None, seed=0):
    life = Life(seed, parameters=parameters)
    log = []
    try:
        log.append(life.phase(probe_seconds, tone="a", label="before"))
        log.append(life.phase(lesson_seconds, tone="a", hand=True, label="lesson A"))
        log.append(life.phase(probe_seconds, tone="a", label="probe A right after"))
        if content == "second":
            log.append(life.phase(lesson_seconds, tone="b", hand=True, label="lesson B"))
            log.append(life.phase(probe_seconds, tone="b", label="probe B right after"))
        remaining, index = float(gap_seconds), 0
        while remaining > 1e-9:
            take = min(block, remaining)
            remaining -= take
            index += 1
            label = "gap %d" % index
            if content == "quiet":
                log.append(life.phase(take, tone=None, label=label))
            elif content == "tone":
                log.append(life.phase(take, tone="a", label=label))
            elif content == "second":
                log.append(life.phase(take, tone="b", label=label))
            else:
                log.append(life.phase(take/2, tone="a", label=label + " (tone a)"))
                log.append(life.phase(take/2, tone="b", label=label + " (tone b)"))
        log.append(life.phase(probe_seconds, tone="a", label="probe A after the gap"))
        if content == "second":
            log.append(life.phase(probe_seconds, tone="b", label="probe B after the gap"))
    finally:
        life.close()
    return log


def summarise(log):
    print("%-32s %8s %7s %8s %8s   %s"
          % ("checkpoint", "retreat", "brake", "cochlea", "dopamine", "route weights"))
    for row in log:
        print("%-32s %8.3f %7.3f %8.3f %8.3f   %s"
              % (row["label"], row["retreat"], row["brake"], row["cochlea"], row["dopamine"],
                 row["route"]))
    probes = [row for row in log if row["label"].startswith("probe")]
    if len(probes) >= 2:
        first, last = probes[0], probes[-1]
        print()
        print("retreat with the tone alone: %.3f right after the lesson, %.3f after the gap"
              " (kept %.0f%%)  route weights %s -> %s"
              % (first["retreat"], last["retreat"],
                 100.*last["retreat"]/max(first["retreat"], 1e-9), first["route"], last["route"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gap", type=float, default=60.)
    parser.add_argument("--content", default="quiet",
                        choices=("quiet", "tone", "second", "life"))
    parser.add_argument("--lesson", type=float, default=20.)
    parser.add_argument("--probe", type=float, default=4.)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    print("262 Hz means back off; then %.0f s of '%s' with learning on" % (args.gap, args.content))
    log = teach(args.gap, args.content, lesson_seconds=args.lesson,
                probe_seconds=args.probe, seed=args.seed)
    summarise(log)
    if args.json:
        Path(args.json).write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
        print("wrote %s" % args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())