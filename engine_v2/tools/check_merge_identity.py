"""Is averaging two genomes the same thing as averaging their weight tables?

    python tools/check_merge_identity.py --seeds 0 1 2

R18 measured, on two brains that had each been taught one thing, that adding the
two born-wired-plus-taught weight vectors and halving them gives a third brain
that does both.  This tool asks a narrower question about the *innate* part: if
two animals differ only in their named innate gains, is the weight table of the
animal built from the averaged gains the same as the average of the two weight
tables?

It is not assumed either way.  For each seed the three brains are built from the
same scaffold, the three weight vectors are read out at birth, and the two
numbers are compared edge by edge.  A difference here means the two kinds of
merge are not interchangeable and every claim about merging has to say which
one it used.
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

from born_wired.embodied import EmbodiedController          # noqa: E402
from born_wired.go2_body import Go2Body                     # noqa: E402
from tools import screen_candidates as screen                # noqa: E402
from tools import taskbank                                   # noqa: E402

TOLERANCE = 1e-9


def birth_weights(genome, seed):
    body = Go2Body(model_path=taskbank.ARENA)
    body.reset(seed=seed, joint_noise=.01)
    parameters = taskbank.reference_parameters(genome)
    parameters.pop("seed", None)
    brain = EmbodiedController(body.home_angles, body.lower_limits, body.upper_limits,
                               seed=seed, **parameters)
    return brain.synapses.src.copy(), brain.synapses.dst.copy(), brain.synapses.weights.copy()


def case(seed, rng, genome_seed):
    first, touched_a = screen.mutate({}, seed, np.random.default_rng(genome_seed),
                                     genes=2, sigma=screen.MUTATION_SIGMA)
    second, touched_b = screen.mutate({}, seed, np.random.default_rng(genome_seed + 1),
                                      genes=2, sigma=screen.MUTATION_SIGMA)
    averaged = screen.merge_genome(first, second)
    src_a, dst_a, w_a = birth_weights(first, seed)
    src_b, dst_b, w_b = birth_weights(second, seed)
    src_m, dst_m, w_m = birth_weights(averaged, seed)
    same_edges = bool(np.array_equal(src_a, src_b) and np.array_equal(src_a, src_m)
                      and np.array_equal(dst_a, dst_b) and np.array_equal(dst_a, dst_m))
    reference = .5 * (w_a + w_b)
    difference = np.abs(w_m - reference)
    return dict(seed=seed, genes_a=touched_a, genes_b=touched_b,
                edges=int(w_a.size), same_edge_list=same_edges,
                max_abs_difference=float(difference.max()) if difference.size else 0.,
                edges_differing=int(np.count_nonzero(difference > TOLERANCE)),
                max_abs_weight=float(np.abs(w_a).max()),
                born_equal=bool(np.array_equal(w_a, w_b)) if same_edges else None)


def single_gene(seed, name, fraction, scale):
    """Move one gene and ask whether half the move is half the weight change."""
    base = taskbank.reference_parameters({})[name]
    step = fraction * scale
    if name in taskbank.INTEGER_GENES:
        step = float(max(1, int(round(step))))
    little = {name: base + step}
    half = {name: base + .5 * step}
    if name in taskbank.INTEGER_GENES:
        half[name] = base + step
    _, _, w_base = birth_weights({}, seed)
    _, _, w_little = birth_weights(little, seed)
    _, _, w_half = birth_weights(half, seed)
    difference = np.abs(w_half - .5 * (w_base + w_little))
    return dict(gene=name, start=float(base), step=float(step),
                max_abs_difference=float(difference.max()),
                edges_differing=int(np.count_nonzero(difference > TOLERANCE)),
                weight_span=float(np.abs(w_little - w_base).max()))


def pair_step(seed, first, second, scale, fraction):
    """Move two genes together; is the result the sum of the two moves apart?"""
    base = taskbank.reference_parameters({})
    step = {name: fraction * scale[name] for name in (first, second)}
    for name in (first, second):
        if name in taskbank.INTEGER_GENES:
            step[name] = float(max(1, int(round(step[name]))))
    _, _, w_base = birth_weights({}, seed)
    _, _, w_first = birth_weights({first: base[first] + step[first]}, seed)
    _, _, w_second = birth_weights({second: base[second] + step[second]}, seed)
    _, _, w_both = birth_weights({first: base[first] + step[first],
                                  second: base[second] + step[second]}, seed)
    separate = (w_first - w_base) + (w_second - w_base)
    difference = np.abs((w_both - w_base) - separate)
    return dict(first=first, second=second,
                max_abs_difference=float(difference.max()),
                edges_differing=int(np.count_nonzero(difference > TOLERANCE)),
                span=float(np.abs(separate).max()))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--genome-seed", type=int, default=11)
    parser.add_argument("--pairs", action="store_true",
                        help="move two genes at once and ask whether the moves add up")
    parser.add_argument("--single-gene", action="store_true",
                        help="walk one gene at a time and see which ones are not linear")
    parser.add_argument("--gene-fraction", type=float, default=.5)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts" / "merge_identity.json")
    args = parser.parse_args(argv)
    started = time.perf_counter()
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    if args.single_gene:
        scale = screen.scales({})
        rows = [single_gene(seeds[0], name, args.gene_fraction, scale[name])
                for name in taskbank.GENES]
        bad = [row for row in rows if row["edges_differing"]]
        report = dict(question="which single genes make the two kinds of merge differ?",
                      tolerance=TOLERANCE, seed=seeds[0], fraction=args.gene_fraction,
                      genes=len(rows), nonlinear=[row["gene"] for row in bad], rows=rows,
                      wall_seconds=time.perf_counter() - started)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n",
                               encoding="utf-8")
        print("%-32s %8s %10s %12s" % ("gene", "start", "span", "edges differing"))
        for row in sorted(rows, key=lambda item: -item["edges_differing"])[:14]:
            print("%-32s %8.3f %10.3f %12d" % (row["gene"], row["start"],
                                               row["weight_span"], row["edges_differing"]))
        print("non-linear genes: %d of %d" % (len(bad), len(rows)))
        print("-> %s" % args.output)
        return 0
    if args.pairs:
        scale = screen.scales({})
        linear = [name for name in taskbank.GENES
                  if single_gene(seeds[0], name, .01, scale[name])["weight_span"] > 0.]
        rows = [pair_step(seeds[0], first, second, scale, args.gene_fraction)
                for index, first in enumerate(linear) for second in linear[index + 1:]]
        bad = [row for row in rows if row["edges_differing"]]
        report = dict(question="do two genes moved at once add up?",
                      tolerance=TOLERANCE, seed=seeds[0], fraction=args.gene_fraction,
                      linear_genes=linear, pairs=len(rows), interacting=len(bad), rows=rows,
                      wall_seconds=time.perf_counter() - started)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n",
                               encoding="utf-8")
        print("%d genes move weights, %d pairs tried, %d pairs do not add up"
              % (len(linear), len(rows), len(bad)))
        for row in sorted(bad, key=lambda item: -item["edges_differing"])[:12]:
            print("   %-24s + %-24s %6d edges, max |diff| %.3f"
                  % (row["first"], row["second"], row["edges_differing"],
                     row["max_abs_difference"]))
        print("-> %s" % args.output)
        return 0
    cases = [case(seed, None, args.genome_seed) for seed in seeds]
    report = dict(question=("is the weight table of the averaged genome the average of the "
                            "two weight tables?"),
                  tolerance=TOLERANCE, reference_size=taskbank.REFERENCE_SIZE,
                  cases=cases, wall_seconds=time.perf_counter() - started)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n",
                           encoding="utf-8")
    print("%5s %8s %12s %14s %12s" % ("seed", "edges", "same edges", "max |diff|", "|diff|>1e-9"))
    for item in cases:
        print("%5d %8d %12s %14.3e %12d"
              % (item["seed"], item["edges"], item["same_edge_list"],
                 item["max_abs_difference"], item["edges_differing"]))
    print("-> %s" % args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())