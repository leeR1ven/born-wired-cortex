"""Draw a lot of candidates and examine them side by side.

    python tools/round_parallel.py --candidates 1200 --workers 6 --stage probe \
        --scaffold 0 --genome-seed-base 30 --exam-seeds 0 --out artifacts/round4

One process examines about twenty candidates a minute on the cheap stage, so
"a few hundred" is a night and "tens of thousands" is not a bigger number in
the same file - it is a different way of running the same thing.  The work is
embarrassingly parallel: one candidate is one animal and shares nothing with
the next.  This tool starts N worker processes, gives each its own draw seed
and its own ledger part, and merges the parts into one ledger at the end.

Every worker draws the way tools/screen_candidates.py draws - ``mutate`` on the
birth genome - so a candidate here is the same kind of animal as a candidate
there, and the merged ledger is read back with that tool's ``--report``.  The
label carries the worker's draw seed, so two workers never write the same
label, and a part that exists is resumed rather than redrawn.

A candidate that survives this is examined again over three seeds before
anything is claimed about it: this stage is a filter, not a result.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import screen_candidates as sc                          # noqa: E402
from tools import taskbank                                         # noqa: E402


def examine(args):
    """One worker: draw ``--count`` candidates and append each to ``--part``."""
    # The maths libraries are told to use one thread each by
    # ``worker_environment``; torch does not read those variables, it has to be
    # told.  Left alone, twelve workers ask for twelve times the threads torch
    # thinks it is allowed and the round spends its time in the scheduler.
    try:
        import torch
        torch.set_num_threads(max(1, int(args.threads)))
    except Exception:
        pass
    rng = np.random.default_rng(args.genome_seed)
    ledger = Path(args.part)
    done = sum(1 for row in sc.read_ledger(ledger) if row.get("kind") == "candidate")
    for _ in range(done):
        # resume where the part stopped instead of redrawing what it has
        sc.mutate({}, args.scaffold, rng, genes=args.genes, sigma=args.sigma)
    started = time.perf_counter()
    deadline = None if args.budget_minutes is None else started + 60. * args.budget_minutes
    for index in range(args.count):
        if deadline is not None and time.perf_counter() > deadline:
            print("budget reached after %d candidates" % index, flush=True)
            break
        genome, touched = sc.mutate({}, args.scaffold, rng, genes=args.genes,
                                    sigma=args.sigma)
        label = "s%dg%d-%04d" % (args.scaffold, args.genome_seed, done + index)
        record = sc.evaluate(genome, args.scaffold, args.exam_seeds, stage=args.stage,
                             label=label,
                             extra=dict(touched=touched, origin="mutant", parent=None,
                                        genome_seed=args.genome_seed, worker=args.worker),
                             verbose=False, stop_at_first_miss=args.stop_on_first_miss)
        sc.append(record, ledger)
        print("%-28s %d/%d %s" % (label, record["n_passed"], record["n_tasks"],
                                  record["passed_tasks"]), flush=True)
    return 0


THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def worker_environment(threads):
    """The environment every worker runs in: ``threads`` math threads, no more.

    One candidate is one animal, and the simulator plus numpy are each already
    threaded.  Left alone, six workers ask for twenty threads on ten cores and
    the round runs at about a third of the speed it should - which is what the
    first launch of round 5 measured: 2.4 min of wall clock for a candidate
    whose own exam costs 59 s.
    """
    environment = dict(os.environ)
    for name in THREAD_VARIABLES:
        environment[name] = str(int(threads))
    return environment


def parts_of(out, workers):
    return [Path("%s_part%d.jsonl" % (out, index)) for index in range(workers)]


def share(count, workers):
    """Split a candidate count evenly, giving the remainder to the first ones."""
    base, extra = divmod(int(count), int(workers))
    return [base + (1 if index < extra else 0) for index in range(workers)]


def merge(parts, merged):
    rows = 0
    with Path(merged).open("w", encoding="utf-8") as handle:
        for part in parts:
            if not Path(part).exists():
                continue
            for line in Path(part).read_text(encoding="utf-8").splitlines():
                if line.strip():
                    handle.write(line.rstrip() + "\n")
                    rows += 1
    return rows


def tally(merged, scaffold, stage):
    """What the round actually produced, read back off the merged ledger."""
    rows = sc.read_ledger(merged)
    candidates = [row for row in rows if row.get("kind") == "candidate"
                  and row.get("scaffold") == scaffold]
    built = [row for row in candidates if sc.built(row)]
    print("ledger %s: %d candidates, %d of them built (the rest are genomes the "
          "engine refused, not animals that failed)" % (merged, len(candidates), len(built)))
    if not built:
        return
    scores = sorted((row["n_passed"] for row in built), reverse=True)
    counts = {}
    for row in built:
        counts.setdefault(row["n_passed"], 0)
        counts[row["n_passed"]] += 1
    print("score spread:", ", ".join("%d passed x%d" % (score, counts[score])
                                     for score in sorted(counts, reverse=True)))
    print("best %d, median %d, worst %d"
          % (scores[0], scores[len(scores) // 2], scores[-1]))
    # With the chain on, ``passed`` counts only the questions an animal was
    # asked, so the denominator has to be how many got as far as the question,
    # not how many were drawn: otherwise a question nobody reached reads as a
    # question nobody could do.
    per_task, reached = {}, {}
    for row in built:
        for name in row.get("ran_tasks") or [r["task"] for run in row.get("runs", [])
                                             for r in run.get("results", [])]:
            reached[name] = reached.get(name, 0) + 1
        for name in row["passed_tasks"]:
            per_task[name] = per_task.get(name, 0) + 1
    print("each question in the order it is asked: how many got as far as it, "
          "and how many of those passed")
    for entry in taskbank.stage_tasks(stage):
        name = entry["name"]
        seen, hit = reached.get(name, 0), per_task.get(name, 0)
        print("   %-46s %5d of %5d reached   %5d passed  (%.1f%%)"
              % (name, seen, len(built), hit, 100. * hit / seen if seen else 0.))
    cut = [row for row in built if not row.get("complete", True)]
    if cut:
        print("the chain cut %d of %d short; the %d that answered every question are "
              "the ones this stage kept" % (len(cut), len(built), len(built) - len(cut)))
    wall = sum(row["wall_seconds"] for row in built)
    print("%.1f min of simulation in the round, %.1f s per animal"
          % (wall / 60., wall / len(built)))


def main(argv=None):
    parser = argparse.ArgumentParser(description="draw and examine candidates in parallel")
    parser.add_argument("--candidates", type=int, default=120)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--stage", default="probe",
                        choices=("screen", "probe", "full"))
    parser.add_argument("--exam-seeds", default="0")
    parser.add_argument("--scaffold", type=int, default=0)
    parser.add_argument("--genome-seed-base", type=int, default=0)
    parser.add_argument("--genes", type=int, default=sc.MUTATION_GENES)
    parser.add_argument("--sigma", type=float, default=sc.MUTATION_SIGMA)
    parser.add_argument("--budget-minutes", type=float, default=None,
                        help="per worker, not for the whole round")
    parser.add_argument("--out", type=Path, required=True,
                        help="stem for the merged ledger and the parts")
    parser.add_argument("--threads", type=int, default=1,
                        help="math threads per worker; one is right when several "
                             "workers share the machine, because the simulator and "
                             "numpy are each already threaded and threads that "
                             "outnumber cores only make the round slower")
    parser.add_argument("--stop-on-first-miss", action="store_true",
                        help="ask the probe as a chain, cheapest question first, "
                             "and stop at the first miss; most candidates then "
                             "cost the two cheap questions instead of all eight")
    parser.add_argument("--merge-only", action="store_true",
                        help="merge and tally what the parts already hold")
    parser.add_argument("--worker", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--part", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--genome-seed", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--count", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.exam_seeds = [int(item) for item in str(args.exam_seeds).split(",") if item.strip()]

    merged = Path(str(args.out) + ".jsonl")
    parts = parts_of(args.out, args.workers)
    if args.worker is not None:
        args.part = args.part or parts[args.worker]
        return examine(args)

    if not args.merge_only:
        started = time.perf_counter()
        per = share(args.candidates, args.workers)
        running = []
        for index, count in enumerate(per):
            if count <= 0:
                continue
            log = Path("%s_part%d.log" % (args.out, index))
            handle = log.open("w", encoding="utf-8")
            command = [sys.executable, "-X", "utf8", "-u", str(Path(__file__).resolve()),
                       "--out", str(args.out),
                       "--worker", str(index), "--part", str(parts[index]),
                       "--genome-seed", str(args.genome_seed_base + index),
                       "--count", str(count), "--stage", args.stage,
                       "--exam-seeds", ",".join(str(s) for s in args.exam_seeds),
                       "--scaffold", str(args.scaffold), "--genes", str(args.genes),
                       "--sigma", str(args.sigma), "--threads", str(args.threads)]
            if args.budget_minutes is not None:
                command += ["--budget-minutes", str(args.budget_minutes)]
            if args.stop_on_first_miss:
                command += ["--stop-on-first-miss"]
            running.append((index, count, subprocess.Popen(command, cwd=str(ROOT),
                                                           env=worker_environment(args.threads),
                                                           stdout=handle,
                                                           stderr=subprocess.STDOUT),
                            handle, log))
            print("worker %d -> %s (%d candidates, log %s)"
                  % (index, parts[index].name, count, log.name), flush=True)
        for index, count, process, handle, log in running:
            code = process.wait()
            handle.close()
            print("worker %d finished: %d candidates wanted, exit %s, log %s"
                  % (index, count, code, log.name), flush=True)
        print("round wall %.1f min" % ((time.perf_counter() - started) / 60.))

    rows = merge(parts, merged)
    print("merged %d rows from %d parts -> %s" % (rows, len(parts), merged))
    tally(merged, args.scaffold, args.stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())