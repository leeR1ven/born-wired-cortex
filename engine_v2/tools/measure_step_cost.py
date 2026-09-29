"""Where the seconds in one exam question actually go, measured, not guessed.

    python tools/measure_step_cost.py parts
    python tools/measure_step_cost.py walk-ab --seconds 10 --rounds 3
    python tools/measure_step_cost.py build

``parts`` splits one simulation step of the reference animal into the pieces the
report quotes: the neural step, the per-step diagnostic readout, the bounds
checks the run harness does on the whole weight table, and the physics.  The
numbers in docs/题库与训练方式自审_20260929.md section 7.1 come from here.

``walk-ab`` runs one walking question twice per round, alternating between the
readout as the run used to ask for it (the whole diagnostics dictionary, one
entry read out of it) and ``scaffold_drift``, which computes that one entry in
place.  It also prints the reading the question decides on, so the run proves
the two ways agree instead of asserting it.  Section 7.2 of that document.

``build`` is the per-question fixed cost: the MuJoCo body and the controller,
which a question pays once no matter how short it is (section 7.1).

The engine and the exam are the same objects the bank uses, so a number here is
a number from the bank.  ``BORN_WIRED_DEVICE`` picks the engine, as everywhere
else.
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import taskbank                                          # noqa: E402
from born_wired.innate import InnateController                      # noqa: E402


def clock(function, repeats):
    """Seconds per call, after one warm call so first-call costs are not it."""
    function()
    started = time.perf_counter()
    for _ in range(repeats):
        out = function()
    return (time.perf_counter() - started) / repeats, out


def finite(array):
    return bool(np.all(np.isfinite(array)))


def parts(args):
    ctx = taskbank.context({}, 0)
    body = taskbank.clean_body(ctx["model_path"])
    brain = taskbank.brain_for(ctx, body)
    synapses, network = brain.synapses, brain.network
    weights = synapses.weights
    activity, adaptation = network.activity, network.adaptation
    observation = body.observe()
    environment = taskbank.blank_environment()
    print("the reference animal: %d cells, %d edges"
          % (network.n_neurons, weights.size))
    rows = [
        ("brain.step (the simulation)", lambda: brain.step(
            observation, environment=environment, dt=.01, learn=True)),
        ("full diagnostics()", brain.diagnostics),
        ("  scaffold_drift()", brain.scaffold_drift),
        ("weights copy + refresh", lambda: synapses.weights),
        ("  finite(weights)", lambda: finite(weights)),
        ("  weights >= lower", lambda: bool(np.all(weights >= synapses.lower))),
        ("  weights <= w_max", lambda: bool(np.all(weights <= synapses.w_max))),
        ("  activity in [0, 1]", lambda: bool(np.all((activity >= 0) & (activity <= 1)))),
        ("  adaptation in [0, 1]", lambda: bool(np.all((adaptation >= 0) & (adaptation <= 1)))),
        ("body.step (5 physics steps)", lambda: body.step(body.home_angles, duration=.01)),
    ]
    print("%-32s %10s %12s" % ("per step", "ms", "x1000 steps"))
    for label, function in rows:
        seconds, _ = clock(function, args.repeats)
        print("%-32s %10.3f %11.2f s" % (label, seconds * 1000., seconds * 1000.))
    checks = ["weights copy + refresh", "  finite(weights)", "  weights >= lower",
              "  weights <= w_max", "  activity in [0, 1]", "  adaptation in [0, 1]"]
    total = 0.
    for label, function in rows:
        if label in checks:
            seconds, _ = clock(function, args.repeats)
            total += seconds
    print("%-32s %10.3f %11.2f s" % ("harness checks, together", total * 1000.,
                                     total * 1000.))
    per_second = int(round(1. / taskbank.DT))
    print("one second of simulation is %d steps, so a 10 s question multiplies the "
          "middle column by %d." % (per_second, 10 * per_second))


def build(args):
    ctx = taskbank.context({}, 0)
    body_seconds, body = clock(lambda: taskbank.clean_body(ctx["model_path"]), args.repeats)
    brain_seconds, _ = clock(lambda: taskbank.brain_for(ctx, body), args.repeats)
    print("clean_body()  %6.3f s" % body_seconds)
    print("brain_for()   %6.3f s" % brain_seconds)
    print("per question  %6.3f s before any simulation happens" % (body_seconds + brain_seconds))


def old_readout(controller, weights=None):
    """The readout the run loop used to ask for: the whole dictionary.

    Copied from the body of ``diagnostics`` as it stood before section 7.1, so
    the comparison is against what ran, not against a remembered idea of it.
    """
    rate = controller.network.activity
    table = controller.synapses.weights
    entry = dict(neurons=controller.network.n_neurons, edges=len(table),
                 max_weight_change=float(np.max(np.abs(table - controller.initial_weights))),
                 scaffold_max_relative_change=float(np.max(np.abs(
                     table[controller.synapses.tether > 0]
                     / controller.initial_weights[controller.synapses.tether > 0] - 1))),
                 memory_weights=table[controller.synapses.tether == 0].tolist(),
                 mean_activity=float(rate.mean()),
                 active_fraction=float(np.mean(rate > .01)))
    return entry["scaffold_max_relative_change"]


def walk_ab(args):
    ctx = taskbank.context({}, 0)
    fast = InnateController.scaffold_drift
    old, new = [], []
    print("one %g s walk, the same animal in the same process, alternating:" % args.seconds)
    for index in range(args.rounds):
        InnateController.scaffold_drift = lambda self, weights=None: old_readout(self)
        started = time.perf_counter()
        before = taskbank.walk_measure(ctx, args.seconds, "autonomous", "origin")
        old.append(time.perf_counter() - started)
        InnateController.scaffold_drift = fast
        started = time.perf_counter()
        after = taskbank.walk_measure(ctx, args.seconds, "autonomous", "origin")
        new.append(time.perf_counter() - started)
        print("  round %d: old %6.2f s  new %6.2f s   "
              "(progress %.3f / %.3f, straightness %.4f / %.4f, up_z %.4f / %.4f)"
              % (index, old[-1], new[-1], before["progress_m"], after["progress_m"],
                 before["straightness"], after["straightness"],
                 before["min_up_z"], after["min_up_z"]))
        if (before["progress_m"] != after["progress_m"]
                or before["straightness"] != after["straightness"]):
            print("  the two ways do NOT agree - that is a bug, not a speed result")
    InnateController.scaffold_drift = fast
    slow, quick = statistics.median(old), statistics.median(new)
    print()
    print("%g s walk:  old %.2f s  ->  new %.2f s   (%.0f%% off, %.2f s saved)"
          % (args.seconds, slow, quick, 100. * (slow - quick) / slow, slow - quick))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("what", choices=("parts", "build", "walk-ab"))
    parser.add_argument("--repeats", type=int, default=20,
                        help="calls per measurement for `parts` and `build`")
    parser.add_argument("--rounds", type=int, default=3, help="for `walk-ab`")
    parser.add_argument("--seconds", type=float, default=10.,
                        help="how long the walk in `walk-ab` is")
    args = parser.parse_args(argv)
    {"parts": parts, "build": build, "walk-ab": walk_ab}[args.what](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
