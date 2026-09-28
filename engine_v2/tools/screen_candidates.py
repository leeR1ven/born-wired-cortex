"""One round of: draw a lot of animals, examine every one, keep the useful.

    python tools/screen_candidates.py --candidates 200 --budget-minutes 120
    python tools/screen_candidates.py --report
    python tools/screen_candidates.py --merge --scaffold 0

A candidate is a genome: the named innate gains of tools/taskbank.py's GENES,
plus the seed that draws the random scaffold.  A round draws a genome, puts it
through the whole exam bank, and appends one line to the ledger the moment the
exam ends, so a run can be stopped and started again without losing anything.

Two candidates can only be merged if they were built on the same scaffold seed,
because merging is an arithmetic operation on one shared edge list - the same
operation R18 measured on two taught brains.  Merging two genomes built on
different scaffolds is not defined here and the code refuses to do it.

Merging is the elementwise mean of the two genomes.  Whether that is the same
thing as averaging the two weight tables is not assumed: tools/check_merge_identity.py
measures it.
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import taskbank                                      # noqa: E402

LEDGER = ROOT / "artifacts" / "screen_ledger.jsonl"
DEFAULT_SCAFFOLD = 0
MUTATION_SIGMA = .25      # one step is a quarter of the gene's own scale
MUTATION_GENES = 3        # genes touched per candidate
SCALE_FLOOR = .05         # scale of a gene that happens to be zero


def scales(genome=None):
    """How far each gene may move in one step: its own size, or 1 if it is 0."""
    base = taskbank.reference_parameters(genome)
    return {name: max(abs(float(base[name])), SCALE_FLOOR) for name in taskbank.GENES}


def mutate(genome, scaffold, rng, genes=MUTATION_GENES, sigma=MUTATION_SIGMA):
    """A copy of the genome with a few genes nudged, and why recorded."""
    base = taskbank.reference_parameters(genome)
    scale = scales(genome)
    child = dict(genome)
    touched = {}
    for name in rng.choice(taskbank.GENES, size=min(genes, len(taskbank.GENES)),
                           replace=False):
        start = float(base[name])
        step = float(rng.normal(0., sigma)) * scale[name]
        if name in taskbank.INTEGER_GENES:
            step = float(int(round(step))) or (1. if step >= 0 else -1.)
        value = start + step
        # A gene may not wander more than a few of its own sizes away, and one
        # that is positive stays positive: a negative time constant or a
        # negative number of steps is not a different animal, it is no animal.
        low, high = start - 4. * scale[name], start + 4. * scale[name]
        if start > 0.:
            low = max(low, .05 * start)
        value = float(np.clip(value, low, high))
        if name in taskbank.INTEGER_GENES:
            value = float(max(1, int(round(value))))
            if value == start:
                value = start + 1.
        child[name] = value
        touched[name] = value - start
    return child, touched


def merge(genomes):
    """Add the parents' innate gains and halve them, gene by gene."""
    names = set()
    for genome in genomes:
        names |= set(genome)
    return {name: float(np.mean([genome.get(name, float(taskbank.reference_parameters(genome)[name]))
                                 for genome in genomes])) for name in sorted(names)}


def merge_genome(first, second):
    """The average of two genomes, over the union of the genes they set."""
    names = sorted(set(first) | set(second))
    return {name: .5 * (float(first.get(name, 0.)) + float(second.get(name, 0.)))
            for name in names}


def evaluate(genome, scaffold, exam_seeds, stage="screen", names=None, label=None, extra=None):
    started = time.perf_counter()
    runs = []
    for seed in exam_seeds:
        results = taskbank.run_bank(genome, seed, stage=stage, names=names)
        runs.append(dict(seed=seed, results=results, summary=taskbank.summarise(results)))
    passed = sorted({r["task"] for run in runs for r in run["results"] if r["passed"]})
    return dict(kind="candidate", label=label, scaffold=int(scaffold), genome=genome,
                exam_seeds=list(exam_seeds), stage=stage,
                passed_tasks=passed, n_passed=len(passed),
                n_tasks=len(runs[0]["results"]) if runs else 0,
                runs=runs, wall_seconds=time.perf_counter() - started,
                extra=extra or {})


def append(record, ledger=LEDGER):
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_ledger(ledger=LEDGER):
    if not Path(ledger).exists():
        return []
    records = []
    with Path(ledger).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def keep(records, at_least=1, scaffold=None):
    """The animals worth keeping: everything that passed at least a few tasks.

    Selection here is a list, not a champion.  A round keeps every candidate
    that cleared the bar on at least ``at_least`` tasks, because which of them
    is useful only becomes clear once their abilities are laid side by side.
    """
    pool = [r for r in records if r.get("kind") == "candidate"
            and (scaffold is None or r["scaffold"] == scaffold)]
    return sorted([r for r in pool if r["n_passed"] >= at_least],
                  key=lambda r: (-r["n_passed"], r["label"]))


def wrote_keep(kept, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in kept:
            handle.write(json.dumps(dict(label=record["label"], scaffold=record["scaffold"],
                                         stage=record["stage"], exam_seeds=record["exam_seeds"],
                                         n_passed=record["n_passed"], n_tasks=record["n_tasks"],
                                         passed_tasks=record["passed_tasks"],
                                         genome=record["genome"],
                                         touched=record["extra"].get("touched", {})),
                                    ensure_ascii=False) + "\n")
    return path


def gene_association(records, task, scaffold=None, minimum=3):
    """Which named gains separate the animals that pass a task from the rest?

    Only the genes a candidate actually moved are counted, and each gene needs
    a few animals on both sides before it is reported.  This is a description of
    this round, not a claim about cause: the genes are drawn at random and a
    few dozen animals cannot tell a real effect from a coincidence.
    """
    rows = []
    for name in taskbank.GENES:
        passed, failed = [], []
        for record in records:
            if record.get("kind") != "candidate" or (scaffold is not None
                                                     and record["scaffold"] != scaffold):
                continue
            step = abs(float(record["extra"].get("touched", {}).get(name, 0.)))
            if not step:
                continue
            (passed if task in record["passed_tasks"] else failed).append(step)
        if len(passed) < minimum or len(failed) < minimum:
            continue
        rows.append(dict(gene=name, passed_mean=float(np.mean(passed)),
                         failed_mean=float(np.mean(failed)),
                         difference=float(np.mean(passed) - np.mean(failed)),
                         passed=len(passed), failed=len(failed)))
    return sorted(rows, key=lambda row: -abs(row["difference"]))


def select(records, scaffold=None):
    """Keep the animals that between them cover the most tasks.

    Greedy cover: take whichever candidate adds the most tasks nobody has
    covered yet, then the next, until nothing new appears.  The point is not
    one champion but a set whose abilities do not overlap - those are the pairs
    worth merging.
    """
    pool = [r for r in records if r.get("kind") == "candidate"
            and (scaffold is None or r["scaffold"] == scaffold)]
    uncovered = {task for record in pool for task in record["passed_tasks"]}
    chosen, covered = [], set()
    while uncovered:
        best, best_gain = None, 0
        for record in pool:
            gain = len(set(record["passed_tasks"]) - covered)
            if gain > best_gain or (gain == best_gain and gain and best is not None
                                    and record["n_passed"] > best["n_passed"]):
                best, best_gain = record, gain
        if best is None or best_gain == 0:
            break
        chosen.append(best)
        covered |= set(best["passed_tasks"])
        uncovered -= set(best["passed_tasks"])
        pool = [r for r in pool if r is not best]
    return chosen, sorted(covered)


def report(records, scaffold=None):
    pool = [r for r in records if r.get("kind") == "candidate"
            and (scaffold is None or r["scaffold"] == scaffold)]
    merged = [r for r in records if r.get("kind") == "merged"]
    print("%d candidates examined, %d merged animals" % (len(pool), len(merged)))
    if not pool:
        return
    counts = {}
    for record in pool:
        for task in record["passed_tasks"]:
            counts[task] = counts.get(task, 0) + 1
    print("%-24s %s" % ("task", "candidates passing"))
    for task in sorted(counts, key=lambda name: -counts[name]):
        print("%-24s %d / %d" % (task, counts[task], len(pool)))
    best = max(pool, key=lambda r: r["n_passed"])
    print("best candidate: %s of %d tasks %s"
          % (best["n_passed"], best["n_tasks"], best["passed_tasks"]))
    for at_least in range(best["n_passed"], 0, -1):
        kept = keep(pool, at_least, scaffold)
        if kept:
            print("kept: %d candidates pass >= %d of %d tasks"
                  % (len(kept), at_least, best["n_tasks"]))
            break
    print("%-24s %s" % ("gene", "mean |step| pass vs fail, biggest gaps"))
    for task in sorted(counts, key=lambda name: counts[name]):
        rows = gene_association(pool, task, scaffold)
        if not rows:
            continue
        shown = ", ".join("%s %.3f/%.3f" % (row["gene"], row["passed_mean"], row["failed_mean"])
                          for row in rows[:4])
        print("  %-22s %s" % (task, shown))
    if merged:
        for record in merged:
            print("merged %s -> %s of %d %s"
                  % (record["label"], record["n_passed"], record["n_tasks"],
                     record["passed_tasks"]))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="draw, examine, keep, merge")
    parser.add_argument("--candidates", type=int, default=20)
    parser.add_argument("--stage", default="screen", choices=("screen", "full"))
    parser.add_argument("--exam-seeds", default="0")
    parser.add_argument("--scaffold", type=int, default=DEFAULT_SCAFFOLD)
    parser.add_argument("--genome-seed", type=int, default=0)
    parser.add_argument("--genes", type=int, default=MUTATION_GENES)
    parser.add_argument("--sigma", type=float, default=MUTATION_SIGMA)
    parser.add_argument("--budget-minutes", type=float, default=None)
    parser.add_argument("--ledger", type=Path, default=LEDGER)
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--merge", action="store_true",
                        help="merge the greedy cover's neighbours and examine the child")
    parser.add_argument("--include-birth", action="store_true",
                        help="examine the birth animal as candidate zero")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    records = read_ledger(args.ledger)
    if args.report:
        report(records, args.scaffold)
        return 0

    exam_seeds = [int(item) for item in args.exam_seeds.split(",") if item.strip()]
    rng = np.random.default_rng(args.genome_seed)
    started = time.perf_counter()
    deadline = None if args.budget_minutes is None else started + 60. * args.budget_minutes
    drawn = sum(1 for r in records if r.get("kind") == "candidate"
                and r["scaffold"] == args.scaffold)

    if args.include_birth:
        record = evaluate({}, args.scaffold, exam_seeds, stage=args.stage,
                          label="birth", extra=dict(touched={}, origin="birth"))
        append(record, args.ledger)
        records.append(record)
        print("%-28s %d/%d %s" % (record["label"], record["n_passed"], record["n_tasks"],
                                  record["passed_tasks"]))
        drawn += 1

    for index in range(args.candidates):
        if deadline is not None and time.perf_counter() > deadline:
            print("budget reached after %d candidates" % index)
            break
        genome, touched = mutate({}, args.scaffold, rng, genes=args.genes, sigma=args.sigma)
        label = "s%d-%04d" % (args.scaffold, drawn)
        record = evaluate(genome, args.scaffold, exam_seeds, stage=args.stage, label=label,
                          extra=dict(touched=touched, origin="mutant",
                                     genome_seed=args.genome_seed))
        append(record, args.ledger)
        records.append(record)
        drawn += 1
        print("%-28s %d/%d %s" % (label, record["n_passed"], record["n_tasks"],
                                  record["passed_tasks"]))

    if args.merge:
        chosen, covered = select(records, args.scaffold)
        for first, second in zip(chosen, chosen[1:]):
            genome = merge_genome(first["genome"], second["genome"])
            label = "merge(%s,%s)" % (first["label"], second["label"])
            record = evaluate(genome, args.scaffold, exam_seeds, stage=args.stage, label=label,
                              extra=dict(origin="merge", parents=[first["label"], second["label"]]))
            record["kind"] = "merged"
            append(record, args.ledger)
            records.append(record)
            print("%-28s %d/%d %s" % (record["label"], record["n_passed"], record["n_tasks"],
                                      record["passed_tasks"]))
    report(records, args.scaffold)
    print("wall %.1f min, ledger %s" % ((time.perf_counter() - started) / 60., args.ledger))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())