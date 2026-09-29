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
        # A gene never changes sign.  A positive one stays positive, and one
        # that sits at zero at reference stays nonnegative: the engine refuses
        # a negative row_common exactly as it refuses a negative time constant,
        # so that draw bought no animal at all - it was filed as one that
        # "passed nothing".  eye_row_common is the only zero-valued gene today.
        if start > 0.:
            low = max(low, .05 * start)
        elif start == 0.:
            low = max(low, 0.)
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


def evaluate(genome, scaffold, exam_seeds, stage="screen", names=None, label=None, extra=None,
             verbose=False, stop_at_first_miss=False):
    started = time.perf_counter()
    runs = []

    def announce(seed):
        def report(result):
            print("    %-8s %-34s %-4s %6.1fs" % (label, result["task"],
                                                  "pass" if result["passed"] else "FAIL",
                                                  result["wall_seconds"]), flush=True)
        return report

    for seed in exam_seeds:
        results = taskbank.run_bank(genome, seed, stage=stage, names=names,
                                    on_result=announce(seed) if verbose else None,
                                    stop_at_first_miss=stop_at_first_miss)
        runs.append(dict(seed=seed, results=results, summary=taskbank.summarise(results)))
    passed = sorted({r["task"] for run in runs for r in run["results"] if r["passed"]})
    errors = sum(1 for run in runs for r in run["results"]
                 if r["measures"].get("status") == "error")
    # How many questions the stage holds, how many this animal was actually
    # asked, and where it stopped.  With the chain on, ``n_tasks`` is the size
    # of the exam and ``n_tasks_run`` is what it sat: a row that says 3 of 8
    # without saying it stopped at the fourth would read as a weaker animal
    # than it is, and a row that says "8 of 8" when four were never asked would
    # read as a stronger one.
    asked = [entry["name"] for entry in taskbank.stage_tasks(stage)]
    if names:
        asked = [name for name in asked if name in set(names)]
    ran = [r["task"] for r in runs[0]["results"]] if runs else []
    missed = [r["task"] for run in runs for r in run["results"] if not r["passed"]]
    return dict(kind="candidate", label=label, scaffold=int(scaffold), genome=genome,
                revision=taskbank.revision(),
                exam_seeds=list(exam_seeds), stage=stage,
                passed_tasks=passed, n_passed=len(passed), n_errors=errors,
                n_tasks=len(asked), n_tasks_run=len(ran), ran_tasks=ran,
                first_miss=(missed[0] if missed else None),
                complete=bool(runs) and len(ran) == len(asked),
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


def built(record):
    """An animal that was actually built.

    A genome the engine refuses - a negative gain, say - produces no animal at
    all: every exam comes back as an error.  That is not an animal that failed
    everything, and counting it as one would quietly drag every average down
    and hand the round a candidate that never existed.
    """
    return not record.get("n_errors")


def margin_of(result):
    """How much room a question's answer left, off the row it was written in.

    The margin was computed when the question ran and is already on the row;
    recomputing it from the readings is only for ledgers from before that was
    true, or for a slimmed ledger with the readings taken out - and a slimmed
    ledger keeps this field, so its margins still read.
    """
    value = result.get("margin")
    if value is not None:
        return float(value)
    return taskbank.margin(result.get("task"), result.get("measures") or {})


def recorded_margins(record):
    """Every margin the round's own answers leave behind, task -> slack.

    The readings are already in the ledger, so this costs no simulation: it is
    the same answer read twice, once as a yes/no and once as how much room was
    left.  ``taskbank.margin`` says which reading decides each question.
    """
    out = {}
    for run in record.get("runs", []):
        for result in run.get("results", []):
            value = margin_of(result)
            if value is None:
                continue
            task = result["task"]
            if task not in out or abs(value) < abs(out[task]) or (
                    value > 0 > out[task]):
                out[task] = value
    return out


def near_misses(record, within=.15):
    """Tasks it failed by less than ``within`` of the bar, best first.

    An animal that missed a bar by three percent is not the same animal as one
    that never moved.  A round that only keeps passes throws both away; these
    are the ones a small nudge, or one more gene, could push over the line.
    """
    misses = {}
    for run in record.get("runs", []):
        for result in run.get("results", []):
            if result.get("passed"):
                continue
            value = margin_of(result)
            if value is None or value <= -abs(within):
                continue
            task = result["task"]
            if task not in misses or value > misses[task]:
                misses[task] = value
    return dict(sorted(misses.items(), key=lambda item: -item[1]))


def keep(records, at_least=1, scaffold=None, within=None):
    """The animals worth keeping: everything that passed at least a few tasks.

    Selection here is a list, not a champion.  A round keeps every candidate
    that cleared the bar on at least ``at_least`` tasks, because which of them
    is useful only becomes clear once their abilities are laid side by side.
    """
    pool = [r for r in records if r.get("kind") == "candidate" and built(r)
            and (scaffold is None or r["scaffold"] == scaffold)]
    if within is None:
        worth = [r for r in pool if r["n_passed"] >= at_least]
    else:
        # A round that screens on questions the birth animal fails has animals
        # that pass nothing at all.  One of them may still be a hair under the
        # bar, and that is a candidate to breed from, not waste.
        worth = [r for r in pool if r["n_passed"] >= at_least or near_misses(r, within)]
    def rank(record):
        near = near_misses(record, within) if within is not None else {}
        return (-record["n_passed"], -max(near.values(), default=-9.), record["label"])

    return sorted(worth, key=rank)


def wrote_keep(kept, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in kept:
            handle.write(json.dumps(dict(label=record["label"], scaffold=record["scaffold"],
                                         stage=record["stage"], exam_seeds=record["exam_seeds"],
                                         n_passed=record["n_passed"], n_tasks=record["n_tasks"],
                                         passed_tasks=record["passed_tasks"],
                                         margins=recorded_margins(record),
                                         near_misses=near_misses(record),
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
            if (record.get("kind") != "candidate" or not built(record)
                    or (scaffold is not None and record["scaffold"] != scaffold)):
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


def direction_association(records, task, scaffold=None, minimum=3):
    """Which *way* a gene should be moved, as the round's own animals show it.

    ``gene_association`` averages the size of the step.  That cannot tell a gene
    that helps when it is raised from one that hurts when it is raised: both
    look like "the animals that moved it".  This one splits the animals that
    moved a gene into the ones that raised it and the ones that lowered it and
    compares how often each group passed the task, so the sign comes out and
    the next round can push that gene in the direction the evidence points.

    A round of a few dozen animals is a hint, not a measurement: the lift is
    read as "this is worth a line search", never as "this gene does this".
    """
    rows = []
    for name in taskbank.GENES:
        up, down = [], []
        for record in records:
            if (record.get("kind") != "candidate" or not built(record)
                    or (scaffold is not None and record["scaffold"] != scaffold)):
                continue
            step = float(record["extra"].get("touched", {}).get(name, 0.))
            if not step:
                continue
            (up if step > 0 else down).append(task in record["passed_tasks"])
        if len(up) < minimum or len(down) < minimum:
            continue
        rate_up = sum(up) / len(up)
        rate_down = sum(down) / len(down)
        rows.append(dict(gene=name, rate_up=rate_up, rate_down=rate_down,
                         n_up=len(up), n_down=len(down), lift=rate_up - rate_down,
                         points="raise" if rate_up > rate_down else "lower"))
    return sorted(rows, key=lambda row: -abs(row["lift"]))


def distance_association(records, task, scaffold=None, minimum=3):
    """The genes whose step size actually moves the reading behind the task.

    Pass/fail throws away how far from the bar an animal was.  Where a task
    declares the reading its bar looks at (taskbank.MARGIN), this correlates
    the size of each gene's step with that margin, which is the finer signal a
    line search wants: an animal that missed the bar by a hair is not the same
    as one that never moved.
    """
    if task not in taskbank.MARGIN:
        return []
    margins = {}
    for record in records:
        if (record.get("kind") != "candidate" or not built(record)
                or (scaffold is not None and record["scaffold"] != scaffold)):
            continue
        for run in record["runs"]:
            for result in run["results"]:
                if result["task"] != task:
                    continue
                value = taskbank.margin(task, result["measures"])
                if value is not None:
                    margins[record["label"]] = value
    rows = []
    for name in taskbank.GENES:
        pairs = []
        for record in records:
            if (record.get("kind") != "candidate" or not built(record)
                    or (scaffold is not None and record["scaffold"] != scaffold)):
                continue
            step = float(record["extra"].get("touched", {}).get(name, 0.))
            if not step or record["label"] not in margins:
                continue
            pairs.append((step, margins[record["label"]]))
        if len(pairs) < 4:
            continue
        steps = [p[0] for p in pairs]
        values = [p[1] for p in pairs]
        if np.std(steps) == 0. or np.std(values) == 0.:
            continue
        slope = float(np.corrcoef(steps, values)[0, 1])
        rows.append(dict(gene=name, correlation=slope, animals=len(pairs),
                         mean_margin=float(np.mean(values))))
    return sorted(rows, key=lambda row: -abs(row["correlation"]))


def parents_from_file(path, scaffold=None, limit=4):
    """The parent pool read from a kept list instead of from the ledger.

    A round that wants to breed from animals whose ability was confirmed on
    several seeds hands this tool the confirmed list; breeding from the raw
    ledger would also breed from the ones that only got lucky once.
    """
    pool = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if scaffold is None or record.get("scaffold", scaffold) == scaffold:
                pool.append(dict(label=record["label"], genome=record["genome"],
                                 n_passed=record.get("n_passed", 0),
                                 n_tasks=record.get("n_tasks", 0),
                                 passed_tasks=record.get("passed_tasks", []),
                                 scaffold=record.get("scaffold", scaffold),
                                 extra=dict(touched=record.get("touched", {}))))
    pool.sort(key=lambda r: (-r["n_passed"], r["label"]))
    return pool[:limit]


def parents_of(records, scaffold=None, limit=4, within=None):
    """The animals the next round breeds from.

    A round keeps a list, not a champion.  The pool is what the greedy cover
    picks - abilities that do not overlap, which are exactly the animals worth
    crossing - plus the best scorers, so one that is good everywhere is not
    left out just because somebody else already covered its tasks.
    """
    pool = keep(records, 1, scaffold, within=within)
    if not pool:
        return []
    chosen, _ = select(pool, scaffold)
    ordered = list(chosen)
    seen = {record["label"] for record in ordered}
    for record in pool:
        if record["label"] not in seen:
            ordered.append(record)
            seen.add(record["label"])
    return ordered[:limit]


def select(records, scaffold=None):
    """Keep the animals that between them cover the most tasks.

    Greedy cover: take whichever candidate adds the most tasks nobody has
    covered yet, then the next, until nothing new appears.  The point is not
    one champion but a set whose abilities do not overlap - those are the pairs
    worth merging.
    """
    pool = [r for r in records if r.get("kind") == "candidate" and built(r)
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


def margin_table(records, scaffold=None):
    """Per task: how many passed, and who came closest to the bar anyway.

    ``best_miss`` is the least negative margin of any animal that failed, so a
    number near zero says the question is being lost by a hair by somebody.
    ``nearest`` lists those animals, which are the ones worth breeding from.
    """
    table = {}
    for record in records:
        if record.get("kind") != "candidate" or not built(record):
            continue
        if scaffold is not None and record["scaffold"] != scaffold:
            continue
        for run in record.get("runs", []):
            for result in run.get("results", []):
                value = taskbank.margin(result.get("task"), result.get("measures") or {})
                if value is None:
                    continue
                row = table.setdefault(result["task"], dict(best_miss=None, nearest=[]))
                if not result["passed"]:
                    row["nearest"].append((record["label"], value))
                    row["best_miss"] = value if row["best_miss"] is None else max(
                        row["best_miss"], value)
    for row in table.values():
        row["nearest"] = sorted(row["nearest"], key=lambda pair: -pair[1])[:5]
    return table


def report(records, scaffold=None):
    pool = [r for r in records if r.get("kind") == "candidate" and built(r)
            and (scaffold is None or r["scaffold"] == scaffold)]
    refused = [r for r in records if r.get("kind") == "candidate" and not built(r)
               and (scaffold is None or r["scaffold"] == scaffold)]
    merged = [r for r in records if r.get("kind") == "merged"]
    print("%d candidates examined, %d merged animals" % (len(pool), len(merged)))
    if refused:
        print("   (%d genomes refused by the engine, not counted: %s)"
              % (len(refused), ", ".join(r["label"] for r in refused[:6])))
    if not pool:
        return
    counts = {}
    for record in pool:
        for task in record["passed_tasks"]:
            counts[task] = counts.get(task, 0) + 1
    table = margin_table(pool)
    print("%-24s %8s %12s  %s" % ("task", "passed", "best miss", "who is closest to the bar"))
    for task in sorted(counts, key=lambda name: -counts[name]):
        row = table.get(task)
        if row is None:
            print("%-24s %d / %d" % (task, counts[task], len(pool)))
            continue
        print("%-24s %5d/%-4d %11s   %s"
              % (task, counts[task], len(pool),
                 "nobody failed" if row["best_miss"] is None else "%.3f" % row["best_miss"],
                 ", ".join("%s %.3f" % pair for pair in row["nearest"][:3])))
    partial = [r for r in pool if not r.get("complete", True)]
    if partial:
        print("   (%d of them were cut short by the chain: `n of m` counts the "
              "questions it was asked, and `first miss` says where it stopped)"
              % len(partial))
    best = max(pool, key=lambda r: r["n_passed"])
    print("best candidate: %s of %d tasks %s%s"
          % (best["n_passed"], best["n_tasks"], best["passed_tasks"],
             "" if best.get("complete", True)
             else "  (cut short at %s)" % best.get("first_miss")))
    for at_least in range(best["n_passed"], 0, -1):
        kept = keep(pool, at_least, scaffold)
        if kept:
            print("kept: %d candidates pass >= %d of %d tasks"
                  % (len(kept), at_least, best["n_tasks"]))
            break
    # The size of a step is not comparable between genes - wall_gain moves in
    # whole numbers and steady_inhibition in hundredths - so a table of mean
    # |step| mostly ranks the genes by how big they are.  ``gene_association``
    # still computes it, and it is still the wrong question; what the next
    # round needs is the direction, which is a rate and therefore comparable.
    print("%-24s %s" % ("gene -> which way", "pass rate when raised / when lowered"))
    for task in sorted(counts, key=lambda name: counts[name]):
        rows = direction_association(pool, task, scaffold)
        if not rows and task in taskbank.MARGIN:
            rows = distance_association(pool, task, scaffold)
            if rows:
                shown = ", ".join("%s r%+.2f (%d)" % (row["gene"], row["correlation"],
                                                      row["animals"]) for row in rows[:4])
                print("  %-22s margin correlation: %s" % (task, shown))
                continue
        if not rows:
            continue
        shown = ", ".join("%s %.2f>%.2f %s (%d/%d)"
                          % (row["gene"], row["rate_up"], row["rate_down"], row["points"],
                             row["n_up"], row["n_down"])
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
    parser.add_argument("--stage", default="screen",
                        choices=("screen", "probe", "full"))
    parser.add_argument("--exam-seeds", default="0")
    parser.add_argument("--scaffold", type=int, default=DEFAULT_SCAFFOLD)
    parser.add_argument("--genome-seed", type=int, default=0)
    parser.add_argument("--rng-skip", type=int, default=0,
                        help="burn this many draws first, so a resumed round "
                             "continues instead of redrawing what it examined")
    parser.add_argument("--genes", type=int, default=MUTATION_GENES)
    parser.add_argument("--sigma", type=float, default=MUTATION_SIGMA)
    parser.add_argument("--budget-minutes", type=float, default=None)
    parser.add_argument("--ledger", type=Path, default=LEDGER)
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--merge", action="store_true",
                        help="merge the greedy cover's neighbours and examine the child")
    parser.add_argument("--include-birth", action="store_true",
                        help="examine the birth animal as candidate zero")
    parser.add_argument("--breed", action="store_true",
                        help="draw candidates from this ledger's kept animals, not from birth")
    parser.add_argument("--breed-from", type=Path, default=None,
                        help="read the parent pool from this kept list, not from the ledger")
    parser.add_argument("--parents", type=int, default=4,
                        help="how many kept animals a breeding round draws from")
    parser.add_argument("--stop-on-first-miss", action="store_true",
                        help="ask the questions as a chain and stop at the first "
                             "one the animal fails; the probe stage is ordered "
                             "cheapest first so this is where the time is")
    parser.add_argument("--verbose", action="store_true",
                        help="print every answer as it lands, so a stall is visible")
    parser.add_argument("--within", type=float, default=None,
                        help="also keep animals that failed by less than this margin")
    parser.add_argument("--write-keep", type=Path, default=None,
                        help="write the round's top-scoring animals to this JSONL")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    records = read_ledger(args.ledger)
    if args.report:
        report(records, args.scaffold)
        return 0

    exam_seeds = [int(item) for item in args.exam_seeds.split(",") if item.strip()]
    rng = np.random.default_rng(args.genome_seed)
    for _ in range(max(0, args.rng_skip)):
        # A stopped round that starts again with the same seed would walk the
        # same stream from the beginning and re-examine the genomes it already
        # has.  Burning the draws it already used puts the next candidate where
        # the round left off.  Only valid while the gene list is unchanged.
        mutate({}, args.scaffold, rng, genes=args.genes, sigma=args.sigma)
    started = time.perf_counter()
    deadline = None if args.budget_minutes is None else started + 60. * args.budget_minutes
    drawn = sum(1 for r in records if r.get("kind") == "candidate"
                and r["scaffold"] == args.scaffold)
    if args.breed and args.breed_from:
        parents = parents_from_file(args.breed_from, args.scaffold, args.parents)
    elif args.breed:
        parents = parents_of(records, args.scaffold, args.parents, args.within)
    else:
        parents = []
    if args.breed:
        if not parents:
            print("no keepable parent in %s yet; run a plain round first" % args.ledger)
            return 1
        print("breeding from %s" % ", ".join(
            "%s (%d/%d)" % (r["label"], r["n_passed"], r["n_tasks"]) for r in parents))

    if args.include_birth:
        # The birth animal is fixed by its seed, so a round that already has it
        # in the ledger adopts that reading instead of writing a second one.
        # Two records called "birth" would make the ledger ambiguous later.
        already = [r for r in records if r.get("kind") == "candidate"
                   and r.get("label") == "birth" and r["scaffold"] == args.scaffold]
        if already:
            record = already[0]
            print("%-28s %d/%d %s (already in the ledger)"
                  % (record["label"], record["n_passed"], record["n_tasks"],
                     record["passed_tasks"]))
        else:
            record = evaluate({}, args.scaffold, exam_seeds, stage=args.stage,
                              label="birth", extra=dict(touched={}, origin="birth"),
                              verbose=args.verbose)  # the birth animal is never cut short
            append(record, args.ledger)
            records.append(record)
            print("%-28s %d/%d %s" % (record["label"], record["n_passed"], record["n_tasks"],
                                      record["passed_tasks"]))
            drawn += 1

    for index in range(args.candidates):
        if deadline is not None and time.perf_counter() > deadline:
            print("budget reached after %d candidates" % index)
            break
        if parents:
            parent = parents[index % len(parents)]
            genome, touched = mutate(parent["genome"], args.scaffold, rng,
                                     genes=args.genes, sigma=args.sigma)
            origin, ancestor = "breed", parent["label"]
        else:
            genome, touched = mutate({}, args.scaffold, rng, genes=args.genes, sigma=args.sigma)
            origin, ancestor = "mutant", None
        # Labels have to stay unique when a round is resumed, restarted, or
        # mixed with an earlier one in the same ledger.
        # The label carries the draw seed as well as the scaffold: arms that
        # were run side by side each write their own ledger, and a label that
        # only counted within one ledger would repeat across them.
        label = "s%dg%d-%04d" % (args.scaffold, args.genome_seed, drawn)
        taken = {r.get("label") for r in records}
        while label in taken:
            drawn += 1
            label = "s%dg%d-%04d" % (args.scaffold, args.genome_seed, drawn)
        record = evaluate(genome, args.scaffold, exam_seeds, stage=args.stage, label=label,
                          extra=dict(touched=touched, origin=origin, parent=ancestor,
                                     genome_seed=args.genome_seed), verbose=args.verbose,
                          stop_at_first_miss=args.stop_on_first_miss)
        append(record, args.ledger)
        records.append(record)
        drawn += 1
        print("%-28s %d/%d %s" % (label, record["n_passed"], record["n_tasks"],
                                  record["passed_tasks"]))

    if args.write_keep:
        best = max((r["n_passed"] for r in records
                    if r.get("kind") == "candidate" and built(r)
                    and r["scaffold"] == args.scaffold), default=0)
        kept = keep(records, best, args.scaffold, within=args.within)
        path = wrote_keep(kept, args.write_keep)
        print("kept list -> %s (%d animals: the ones at the top score %d, plus the ones "
              "that missed a bar by less than %s)" % (path, len(kept), best, args.within))

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