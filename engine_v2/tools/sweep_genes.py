"""Move one gene at a time and see which way the animal changes.

    python tools/sweep_genes.py --genes righting_gain,steady_inhibition \
        --steps -2,-1,0,1,2 --names get_up_from_back,stay_up_when_pushed \
        --seeds 0 --output artifacts/扫描_扶正.jsonl
    python tools/sweep_genes.py --report --output artifacts/扫描_扶正.jsonl

A round that draws three random genes per animal cannot say which gene did
anything: a few dozen genes over a few dozen animals leaves every gene moved by
fewer than ten animals, which is not enough to tell a real effect from a
coincidence.  This tool moves *one* gene at a time, in even steps away from the
base value, and keeps the whole exam reading at every step.  That is a line
search: it answers "which way, how far, and what else did it cost", and it is
the shape of evidence a paper can print as a curve.

Steps are in units of the gene's own size - step -1 is one ``sigma`` below the
base value, exactly the same rule the round uses when it nudges a genome, so a
swept animal is an animal the round could have drawn.  Each answer is appended
the moment it lands, so a stopped sweep resumes where it stopped.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import screen_candidates                              # noqa: E402
from tools import taskbank                                       # noqa: E402


def value_at(name, reference, step, sigma):
    """The value ``step`` steps away from the base, under the round's own rules.

    A gene may not change sign, and one that sits at zero stays nonnegative:
    the engine refuses a negative time constant or a negative row_common, and a
    sweep that produced one would look like an animal that failed everything.
    """
    reference = float(reference)
    scale = max(abs(reference), screen_candidates.SCALE_FLOOR)
    value = reference + float(step) * float(sigma) * scale
    if reference > 0.:
        value = max(value, .05 * reference)
    elif reference == 0.:
        value = max(value, 0.)
    if name in taskbank.INTEGER_GENES:
        value = float(max(1, int(round(value))))
    return value


def read_rows(path):
    rows = []
    if Path(path).exists():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def margin_sum(record):
    total, counted = 0., 0
    for run in record.get("runs", []):
        for result in run.get("results", []):
            value = taskbank.margin(result.get("task"), result.get("measures") or {})
            if value is None:
                continue
            total += value
            counted += 1
    return total, counted


def sweep(args):
    base = {}
    if args.genome:
        base = json.loads(Path(args.genome).read_text(encoding="utf-8"))
        if isinstance(base.get("genome"), dict):
            base = base["genome"]
    base = {key: float(value) for key, value in base.items()}

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    done = {(row["extra"]["gene"], row["extra"]["step"], row["exam_seeds"][0])
            for row in read_rows(output) if row.get("extra", {}).get("gene") is not None}

    genes = [name.strip() for name in args.genes.split(",") if name.strip()]
    steps = [float(item) for item in args.steps.split(",") if item.strip()]
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    names = [item.strip() for item in args.names.split(",") if item.strip()] or None
    reference = taskbank.reference_parameters(base)
    print("the animal as it stands: %s" % output if output.exists() else "", flush=True)

    for gene in genes:
        if gene not in taskbank.GENES:
            print("no such gene: %s" % gene)
            continue
        for step in steps:
            value = value_at(gene, reference[gene], step, args.sigma)
            genome = dict(base)
            genome[gene] = value
            for seed in seeds:
                if (gene, step, seed) in done:
                    continue
                label = "%s@%+.2f" % (gene, step)
                record = screen_candidates.evaluate(
                    genome, args.scaffold, [seed], stage=args.stage, names=names,
                    label=label,
                    extra=dict(gene=gene, step=step, value=value, base=float(reference[gene]),
                               origin="sweep"))
                screen_candidates.append(record, output)
                total, counted = margin_sum(record)
                print("%-28s %+.1f steps -> %-12.6g  %d/%d passed  margin %+.3f over %d questions"
                      % (gene, step, value, record["n_passed"], record["n_tasks"],
                         total, counted), flush=True)
                if record["n_errors"]:
                    print("    (the engine refused this genome: %d errors)" % record["n_errors"])


def report(args):
    rows = [row for row in read_rows(args.output)
            if row.get("extra", {}).get("origin") == "sweep"]
    if not rows:
        print("nothing in %s yet" % args.output)
        return
    genes = sorted({row["extra"]["gene"] for row in rows})
    print("%-28s %6s %12s %8s %10s" % ("gene", "step", "value", "passed", "margin sum"))
    for gene in genes:
        for row in sorted([r for r in rows if r["extra"]["gene"] == gene],
                          key=lambda r: r["extra"]["step"]):
            total, _ = margin_sum(row)
            print("%-28s %+6.1f %12.6g %5d/%-3d %+10.3f"
                  % (gene, row["extra"]["step"], row["extra"]["value"],
                     row["n_passed"], row["n_tasks"], total))
        picked = max([r for r in rows if r["extra"]["gene"] == gene],
                     key=lambda r: (r["n_passed"], margin_sum(r)[0]))
        print("   best step for %s: %+.1f (value %g, %d/%d passed)"
              % (gene, picked["extra"]["step"], picked["extra"]["value"],
                 picked["n_passed"], picked["n_tasks"]))
    print()
    print("A step that raises the score and the margin and breaks nothing else is a")
    print("gene worth keeping.  One that trades a task away shows the trade in this table.")


def main(argv=None):
    parser = argparse.ArgumentParser(description="one gene at a time")
    parser.add_argument("--genes", default="", help="comma separated gene names")
    parser.add_argument("--steps", default="-2,-1,0,1,2")
    parser.add_argument("--sigma", type=float, default=screen_candidates.MUTATION_SIGMA)
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--stage", default="screen", choices=("screen", "full", "probe"))
    parser.add_argument("--names", default="", help="only these questions")
    parser.add_argument("--genome", default=None)
    parser.add_argument("--scaffold", type=int, default=screen_candidates.DEFAULT_SCAFFOLD)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args(argv)
    if args.report:
        report(args)
    else:
        sweep(args)


if __name__ == "__main__":
    main()
