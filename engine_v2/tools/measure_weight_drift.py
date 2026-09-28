"""While it just lives, does anything in the connection table fall?

    python tools/measure_weight_drift.py --seconds 60 --seed 0

The manuscript says no rule in the shipped configuration lowers a weight.  That
claim is about the explicit decay term, which is off.  It is not a claim that
weights never fall: the local rule lowers a weight whenever its source cell
fires while its target sits below the target activity, and the tether pulls
every non-memory edge back towards the value it was born with.  This tool
measures what actually happens to the table during ordinary life: how many
edges rise, how many fall, how far, and which populations they run between.

Read: counts of edges by direction, the largest fall and rise, and the top
pairs of (source group -> destination group) among the edges that fell.  A
memory edge (the teachable route) is reported separately, because its tether is
zero and it is the only table the animal is meant to keep.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import nursery                               # noqa: E402

DT = nursery.DT


def cell_names(brain):
    names = np.full(brain.network.n_neurons, "", dtype=object)
    for name, ids in brain.groups.items():
        for cell in np.atleast_1d(ids):
            if not names[int(cell)]:
                names[int(cell)] = name
    return names


def run(seconds, seed=0, walk=nursery.WALK, parameters=None, routes=True):
    body, brain, eyes, ears = nursery.build(seed, hand_teaches=False, gated=True,
                                            routes=routes, parameters=parameters)
    silence = np.zeros((2, ears.window_samples))
    observation = body.reset(seed=seed)
    environment = {name: np.zeros(4) for name in nursery.ENVIRONMENT}
    environment["foot_support"] = np.zeros(4, dtype=bool)
    names = cell_names(brain)
    try:
        for _ in range(200):                       # 2 s of settling first
            target, activation = brain.step(observation, environment=environment,
                                            eye_pixels=eyes.observe_raw(), ear_waveform=silence,
                                            dt=DT, learn=True, locomotion=walk)
            body.command_eyes(brain.eye_command())
            observation = body.step(target, duration=DT, activation=activation)
        start = brain.synapses.weights.copy()
        start_anchor = brain.synapses.anchor.copy()
        start_memory = brain.nursery_state()["weights"].copy()
        for _ in range(int(round(seconds/DT))):
            target, activation = brain.step(observation, environment=environment,
                                            eye_pixels=eyes.observe_raw(), ear_waveform=silence,
                                            dt=DT, learn=True, locomotion=walk)
            body.command_eyes(brain.eye_command())
            observation = body.step(target, duration=DT, activation=activation)
        end = brain.synapses.weights.copy()
        memory_end = brain.nursery_state()["weights"].copy()
    finally:
        eyes.close()
    movable = brain.synapses.plasticity > 0
    delta = (end - start)[movable]
    anchor = start_anchor[movable]
    src = brain.synapses.src[movable]
    dst = brain.synapses.dst[movable]
    source, target_name = names[src], names[dst]
    report = {
        "seconds": float(seconds), "seed": seed, "walk": walk,
        "edges": int(movable.size),
        "rose": int((delta > 1e-9).sum()),
        "fell": int((delta < -1e-9).sum()),
        "still": int((np.abs(delta) <= 1e-9).sum()),
        "moved_1e-3": int((np.abs(delta) > 1e-3).sum()),
        "moved_1e-2": int((np.abs(delta) > 1e-2).sum()),
        "largest_rise": float(delta.max()) if delta.size else 0.,
        "largest_fall": float(delta.min()) if delta.size else 0.,
        "mean_delta": float(delta.mean()) if delta.size else 0.,
        "below_anchor": int((end[movable] < anchor - 1e-9).sum()),
        "above_anchor": int((end[movable] > anchor + 1e-9).sum()),
        "memory_start": [float(v) for v in start_memory],
        "memory_end": [float(v) for v in memory_end],
    }
    fell = delta < -1e-3
    if fell.any():
        pairs = {}
        for a, b in zip(source[fell], target_name[fell]):
            key = "%s -> %s" % (a, b)
            pairs[key] = pairs.get(key, 0) + 1
        report["fell_by_pair"] = dict(sorted(pairs.items(), key=lambda kv: -kv[1])[:12])
        report["fell_total_more_than_1e-3"] = int(fell.sum())
        worst = np.argsort(delta)[:8]
        report["worst_edges"] = ["%s -> %s %.4f -> %.4f (born %.4f)"
                                 % (source[i], target_name[i], start[movable][i], end[movable][i],
                                    anchor[i]) for i in worst]
    grew = delta > 1e-3
    if grew.any():
        pairs = {}
        for a, b in zip(source[grew], target_name[grew]):
            key = "%s -> %s" % (a, b)
            pairs[key] = pairs.get(key, 0) + 1
        report["rose_by_pair"] = dict(sorted(pairs.items(), key=lambda kv: -kv[1])[:12])
        report["rose_total_more_than_1e-3"] = int(grew.sum())
        best = np.argsort(delta)[-8:][::-1]
        report["best_edges"] = ["%s -> %s %.4f -> %.4f (born %.4f)"
                                % (source[i], target_name[i], start[movable][i], end[movable][i],
                                   anchor[i]) for i in best]
    return report


def show(report):
    print("edges %d, %s s of ordinary life with learning on, seed %d"
          % (report["edges"], report["seconds"], report["seed"]))
    for key in ("rose", "fell", "still", "moved_1e-3", "moved_1e-2",
                "below_anchor", "above_anchor"):
        print("  %-14s %d" % (key, report[key]))
    print("  largest rise %+.4f   largest fall %+.4f   mean %+.6f"
          % (report["largest_rise"], report["largest_fall"], report["mean_delta"]))
    print("  memory route %s -> %s" % (report["memory_start"], report["memory_end"]))
    if "fell_by_pair" in report:
        print("  edges that fell by more than 1e-3: %d" % report["fell_total_more_than_1e-3"])
        for key, count in report["fell_by_pair"].items():
            print("    %-40s %d" % (key, count))
    for line in report.get("worst_edges", []):
        print("  worst: %s" % line)
    if "rose_by_pair" in report:
        print("  edges that rose by more than 1e-3: %d" % report["rose_total_more_than_1e-3"])
        for key, count in report["rose_by_pair"].items():
            print("    %-40s %d" % (key, count))
    for line in report.get("best_edges", []):
        print("  best: %s" % line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=60.)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    report = run(args.seconds, seed=args.seed)
    show(report)
    if args.json:
        Path(args.json).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        print("wrote %s" % args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())