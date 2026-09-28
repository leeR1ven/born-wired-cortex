"""Two learned lessons in one graph: does merging two parents keep both?

Each parent is the same small graph -- same cells, same edge list, same birth
weights -- trained on ONE cue-to-motor pairing.  A merged model is a weight
vector over that same edge list, so "add the two weight tables and halve" is a
defined operation.  Nothing is retrained after the merge.  The cells, the edge
list and the structural bounds are identical in every arm; only the numbers on
the edges differ.
"""

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from born_wired.distributed import DistributedFeatureNetwork


STIMULI = (np.array([1., 0, 0, 0]), np.array([0., 0, 0, 1.]))
CONFIG = dict(dt=.005, trial_seconds=.4, rest_seconds=.2, training_pairs=30,
              teaching_strength=1.2, evaluation_seconds=.6, evaluation_tail_seconds=.15)


def build(seed):
    """Same cells, same edge list, same birth weights, for every parent."""
    net = DistributedFeatureNetwork(hidden_routes=True, seed=seed)
    return net, net.synapses.weights.copy()


def install(net, weights):
    """Put a weight vector on the graph; the graph itself is not touched."""
    return net.install_weights(weights)


def train(net, cue, motor):
    """One lesson only: this cue drives this motor cell.  No other teacher."""
    teacher = np.zeros(2)
    teacher[motor] = CONFIG["teaching_strength"]
    for _ in range(CONFIG["training_pairs"]):
        for _ in range(round(CONFIG["trial_seconds"] / CONFIG["dt"])):
            net.step(STIMULI[cue], teacher, dt=CONFIG["dt"], learn=True)
        for _ in range(round(CONFIG["rest_seconds"] / CONFIG["dt"])):
            net.step(np.zeros(4), None, dt=CONFIG["dt"], learn=True)


def measure(net):
    """Motor activity for each cue with the teacher absent and learning off."""
    table = []
    for cue in range(2):
        net.reset_state()
        rows = []
        for _ in range(round(CONFIG["evaluation_seconds"] / CONFIG["dt"])):
            rows.append(net.step(STIMULI[cue], None, dt=CONFIG["dt"], learn=False))
        tail = np.asarray(rows)[-round(CONFIG["evaluation_tail_seconds"] / CONFIG["dt"]):]
        table.append(tail.mean(axis=0).tolist())
    net.reset_state()
    return table


def winning(table):
    """win[cue][cell]: how far one motor cell beats the other on that cue."""
    return [[float(table[cue][0] - table[cue][1]), float(table[cue][1] - table[cue][0])]
            for cue in range(2)]


def margin(table):
    """Positive margin on a cue means motor cell 0 won that cue."""
    return [float(table[0][0] - table[0][1]), float(table[1][1] - table[1][0])]


def score(table, targets):
    """How far the answer this arm owes for each cue beats the other motor cell."""
    win = winning(table)
    return [win[cue][targets[cue]] for cue in range(2)]


def merge_mean(w_a, w_b):
    return .5 * (w_a + w_b)


def merge_keep(w_a, w_b, birth):
    """Per edge, keep whichever parent moved it further from its birth value."""
    return np.where(np.abs(w_a - birth) >= np.abs(w_b - birth), w_a, w_b)


def merge_sum(w_a, w_b, birth):
    return birth + (w_a - birth) + (w_b - birth)


def parent(seed, cue, motor):
    net, birth = build(seed)
    train(net, cue, motor)
    weights = net.synapses.weights.copy()
    return dict(net=net, birth=birth, weights=weights, table=measure(net),
                lesson=dict(cue=cue, motor=motor))


def moved(weights, birth, threshold=1e-9):
    return np.abs(weights - birth) > threshold


def moved_edges(weights, birth, dst, cell, threshold=1e-9):
    return int(np.count_nonzero(moved(weights, birth, threshold) & (dst == cell)))


def case(seed):
    out = {"seed": seed, "config": CONFIG, "arms": {}}
    nets = []

    def record(name, weights, targets, note=""):
        net, _ = build(seed)
        nets.append(net)
        installed = install(net, weights)
        table = measure(net)
        out["arms"][name] = dict(
            targets=dict(cue_A="motor %d" % targets[0], cue_B="motor %d" % targets[1]),
            motor_table=table, margin=margin(table), score=score(table, targets),
            max_install_deviation=float(np.max(np.abs(installed - np.asarray(weights, dtype=float)))),
            note=note)

    a = parent(seed, cue=0, motor=0)     # cue A -> motor 0
    b = parent(seed, cue=1, motor=1)     # cue B -> motor 1
    birth = a["birth"]
    dst, dst_src, signs = a["net"].synapses.dst, a["net"].synapses.src, a["net"].synapses.signs

    record("birth", birth, (0, 1), "untrained graph, no lesson")
    record("parent_A", a["weights"], (0, 1), "trained on cue A -> motor 0 only")
    record("parent_B", b["weights"], (0, 1), "trained on cue B -> motor 1 only")
    record("mean(A,B)", merge_mean(a["weights"], b["weights"]), (0, 1),
           "add the two weight tables and halve")
    record("keep(A,B)", merge_keep(a["weights"], b["weights"], birth), (0, 1),
           "per edge, keep whichever parent moved it further from birth")
    record("sum(A,B)", merge_sum(a["weights"], b["weights"], birth), (0, 1),
           "both parents' changes added to the birth value")
    record("mean(A,A)", merge_mean(a["weights"], a["weights"]), (0, 1),
           "sanity: a parent merged with itself must reproduce itself")
    record("mean(birth,birth)", merge_mean(birth, birth), (0, 1),
           "two parents that learned nothing")

    # Control: the same two lessons with the motor cells swapped.  A merge that
    # only made "both cues answer" would fail here; keeping the content of each
    # parent means cue A must now drive motor 1 and cue B must drive motor 0.
    swapped_a = parent(seed, cue=0, motor=1)
    swapped_b = parent(seed, cue=1, motor=0)
    record("mean(swapped lessons)", merge_mean(swapped_a["weights"], swapped_b["weights"]), (1, 0),
           "control: cue A -> motor 1 and cue B -> motor 0 merged")

    # Boundary: two parents taught opposite answers to the same cue.
    c = parent(seed, cue=0, motor=0)
    d = parent(seed, cue=0, motor=1)
    record("mean(C,D) conflict", merge_mean(c["weights"], d["weights"]), (0, 1),
           "two parents taught opposite answers to the same cue")

    moves_a, moves_b = a["weights"] - birth, b["weights"] - birth
    # A move of any size and a move of a size that could matter are counted
    # separately: the innate edges are plastic too, so they drift slightly.
    out["graph"] = dict(
        neurons=a["net"].n_neurons, edges=len(a["net"].synapses.src),
        edges_moved_by_A=int(np.count_nonzero(moved(a["weights"], birth))),
        edges_moved_by_B=int(np.count_nonzero(moved(b["weights"], birth))),
        edges_moved_by_A_above_0_05=int(np.count_nonzero(moved(a["weights"], birth, .05))),
        edges_moved_by_B_above_0_05=int(np.count_nonzero(moved(b["weights"], birth, .05))),
        edges_moved_in_opposite_directions=int(np.count_nonzero(moves_a * moves_b < 0)),
        edges_moved_in_opposite_directions_above_0_05=int(np.count_nonzero(
            (moves_a * moves_b < 0) & (np.abs(moves_a) > .05) & (np.abs(moves_b) > .05))),
        A_moved_edges_per_motor_cell={int(cell): moved_edges(a["weights"], birth, dst, cell, .05)
                                      for cell in (13, 14)},
        B_moved_edges_per_motor_cell={int(cell): moved_edges(b["weights"], birth, dst, cell, .05)
                                      for cell in (13, 14)},
        max_abs_change_A=float(np.max(np.abs(moves_a))),
        max_abs_change_B=float(np.max(np.abs(moves_b))))
    # Stored so that any later number can be recomputed without re-running.
    out["weights"] = dict(src=np.asarray(dst_src).tolist(), dst=dst.tolist(), signs=signs.tolist(),
                          birth=birth.tolist(), parent_A=a["weights"].tolist(),
                          parent_B=b["weights"].tolist())
    return out


def main(argv=None):
    started = time.perf_counter()
    seeds = [int(a) for a in (sys.argv[1:] if argv is None else argv)] or [0, 1, 2, 3, 4]
    cases = [case(seed) for seed in seeds]
    report = dict(python=platform.python_version(), numpy=np.__version__,
                  command="python tools/fuse_parents.py " + " ".join(str(s) for s in seeds),
                  seeds=seeds, cases=cases, wall_seconds=time.perf_counter() - started,
                  scope=("Fixed 17-cell graph, fixed edge list, same birth weights; the only "
                         "difference between parents is which cue-to-motor pairing was trained. "
                         "Synthetic four-channel input, not vision. Merging is an operation on the "
                         "weight vector; nothing is retrained after the merge."))
    out = Path(__file__).resolve().parents[1] / "artifacts" / "fusion_parents.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                   encoding="utf-8")
    for data in cases:
        print("seed", data["seed"], data["graph"])
        for name, arm in data["arms"].items():
            print("   %-22s cueA %+0.3f  cueB %+0.3f  (in favour of the intended answer)"
                  % (name, arm["score"][0], arm["score"][1]))
    print("wall %.1f s -> %s" % (report["wall_seconds"], out))


if __name__ == "__main__":
    main()
