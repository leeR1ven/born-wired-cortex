"""Ask the animals a round kept to prove it on more than one seed.

    python tools/confirm_kept.py --kept artifacts/kept_round1.jsonl --seeds 0,1,2
    python tools/confirm_kept.py --kept artifacts/kept_round1.jsonl --stage full --seeds 0

A screening round runs one seed per question because that is what fits in a
round.  One seed is not evidence of an ability, and not because the question is
noisy: ``context(genome, seed)`` hands the same number to the brain and to the
body, so a seed is *a different random animal* - the scaffold is drawn from it
and so is the physical start.  A question like getting up from its back comes
out differently on different scaffolds, so one scaffold is one animal's luck.
This tool asks the kept animals the same questions again on several seeds and
reports only what held on *every* one of them - that, and the birth animal
measured the same way, is what goes in the paper.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import taskbank                                      # noqa: E402


def read_kept(path):
    animals = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                animals.append(json.loads(line))
    return animals


def held_tasks(genome, seeds, stage, names=None, label=None, quiet=False):
    """What this animal passed on every seed, plus the per-seed counts."""
    per_seed, passed_sets = [], []
    for seed in seeds:
        results = taskbank.run_bank(genome, seed, stage=stage, names=names)
        passed = {r["task"] for r in results if r["passed"]}
        passed_sets.append(passed)
        per_seed.append(dict(seed=int(seed), passed=len(passed),
                             total=len(results),
                             failed=sorted({r["task"] for r in results if not r["passed"]})))
        if not quiet:
            print("    %-14s seed %-3d %d/%d" % (label or "", seed, len(passed), len(results)),
                  flush=True)
    held = set(passed_sets[0])
    for passed in passed_sets[1:]:
        held &= passed
    return dict(label=label, genome=genome, seeds=[int(s) for s in seeds],
                per_seed=per_seed, held=sorted(held),
                ever=sorted(set().union(*passed_sets)))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="confirm a round's kept animals")
    parser.add_argument("--kept", type=Path, required=True)
    parser.add_argument("--stage", default="screen",
                        choices=("screen", "probe", "full"))
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--birth", default="yes", choices=("yes", "no"),
                        help="also measure the birth animal, the thing to beat")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    animals = read_kept(args.kept)
    if not animals:
        print("nothing kept in %s" % args.kept)
        return 1
    started = time.perf_counter()
    print("%d kept animals, %d seed(s), stage %s" % (len(animals), len(seeds), args.stage))
    birth = None
    if args.birth == "yes":
        print("  birth")
        birth = held_tasks({}, seeds, args.stage, label="birth")
    checked = []
    for animal in animals:
        print("  %s  (screened at %d/%d)" % (animal["label"], animal["n_passed"], animal["n_tasks"]))
        checked.append(held_tasks(animal["genome"], seeds, args.stage, label=animal["label"]))
    report = dict(stage=args.stage, seeds=seeds, birth=birth, animals=checked,
                  wall_seconds=time.perf_counter() - started)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n",
                               encoding="utf-8")
    baseline = set(birth["held"]) if birth else set()
    print()
    print("%-12s %-18s %-6s %s" % ("animal", "passed on every seed", "count", "beyond birth"))
    if birth:
        print("%-12s %-18s %-6d %s" % ("birth", " ".join(sorted(baseline))[:18], len(baseline), "-"))
    for animal in checked:
        extra = sorted(set(animal["held"]) - baseline)
        print("%-12s %-18s %-6d %s" % (animal["label"], " ".join(sorted(animal["held"]))[:18],
                                       len(animal["held"]), ", ".join(extra) or "-"))
    winners = [a for a in checked if set(a["held"]) - baseline]
    print()
    if winners:
        print("held something the birth animal did not, on every seed: %s"
              % ", ".join(a["label"] for a in winners))
    else:
        print("none of the kept animals held a new ability on every seed")
    if args.output:
        print("-> %s" % args.output)
    print("wall %.1f min" % ((time.perf_counter() - started) / 60.))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
