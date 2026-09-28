"""Two taught behaviours of one body, written into two brains and merged.

One brain is taught that cue A means "walk" (the innate gait cell), another is
taught that cue B means "brace" (the innate flexion cell).  The two weight
vectors live on the same cells and the same edge list, so they can be added and
halved.  The merged brain is then put back on the same body: nothing is
retrained after the merge, and the same body pose is used for every arm.
"""

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from born_wired.feature_routed import FeatureRoutedController
from born_wired.go2_body import Go2Body


CUE = dict(A=np.array([1., 0, 0, 0]), B=np.array([0., 0, 0, 1.]), neutral=np.zeros(4))
LESSON = dict(A="locomotion", B="flexion")
CONFIG = dict(dt=.01, motor_units=100, empty_routes=("flexion", "locomotion"),
              lesson_pairs=6, lesson_seconds=2., lesson_rest_seconds=.5, teaching_drive=.6,
              settle_seconds=1., cue_seconds=8., rest_seconds=2., initial_joint_noise=.01)


class Session:
    """One brain on one body, with a fixed starting pose that can be restored."""

    def __init__(self, seed, weights=None):
        self.seed = seed
        self.body = Go2Body()
        self.body.reset(seed=seed, joint_noise=CONFIG["initial_joint_noise"])
        self.brain = FeatureRoutedController(
            self.body.home_angles, self.body.lower_limits, self.body.upper_limits,
            seed=seed, hidden_routes=True, motor_units=CONFIG["motor_units"],
            empty_routes=CONFIG["empty_routes"])
        self.installed = None
        if weights is not None:
            self.installed = self.brain.install_weights(weights)
        self.observation = self.body.observe()

    def weights(self):
        return self.brain.synapses.weights.copy()

    def rewind(self):
        """Same body pose and same cell state again, with the weights kept."""
        self.body.reset(seed=self.seed, joint_noise=CONFIG["initial_joint_noise"])
        self.brain.install_weights(self.brain.synapses.weights)
        self.observation = self.body.observe()

    def run(self, cue="neutral", drive=None, seconds=1., learn=False):
        features = CUE[cue]
        extra = {} if drive is None else {drive: CONFIG["teaching_drive"]}
        cells = self.brain.groups
        start = self.body.data.xpos[self.body._base].copy()
        rows = []
        for _ in range(round(seconds / CONFIG["dt"])):
            target, activation = self.brain.step(self.observation, features=features,
                                                 dt=CONFIG["dt"], learn=learn, **extra)
            self.observation = self.body.step(target, duration=CONFIG["dt"], activation=activation)
            activity = self.brain.network.activity
            rows.append((float(self.observation["body_height"]),
                         float(self.observation["body_up"][2]),
                         float(activity[cells["locomotion"][0]]),
                         float(activity[cells["flexion"][0]]),
                         float(self.body.data.xpos[self.body._base][0]),
                         float(self.body.data.xpos[self.body._base][1])))
        block = np.asarray(rows)
        tail = block[-min(len(block), round(1 / CONFIG["dt"])):]
        moved = float(np.hypot(block[-1, 4] - start[0], block[-1, 5] - start[1]))
        return dict(seconds=seconds, cue=cue, drive=drive, learning=learn,
                    locomotion=float(tail[:, 2].mean()), flexion=float(tail[:, 3].mean()),
                    height=float(tail[:, 0].mean()), min_height=float(block[:, 0].min()),
                    up_z=float(tail[:, 1].mean()), moved=moved)

    def teach(self, cue):
        action = LESSON[cue]
        for _ in range(CONFIG["lesson_pairs"]):
            self.run(cue=cue, drive=action, seconds=CONFIG["lesson_seconds"], learn=True)
            self.run(cue="neutral", seconds=CONFIG["lesson_rest_seconds"], learn=True)

    def exam(self):
        self.rewind()
        stages = [self.run(cue="neutral", seconds=CONFIG["settle_seconds"])]
        for cue in ("A", "B"):
            stages.append(self.run(cue=cue, seconds=CONFIG["cue_seconds"]))
            stages.append(self.run(cue="neutral", seconds=CONFIG["rest_seconds"]))
        return {stage["cue"] if stage["cue"] != "neutral" else "neutral_%d" % index: stage
                for index, stage in enumerate(stages)}


def merge_keep(w_a, w_b, birth):
    return np.where(np.abs(w_a - birth) >= np.abs(w_b - birth), w_a, w_b)


def case(seed):
    out = dict(seed=seed, arms={})
    walker = Session(seed)
    walker.teach("A")                       # cue A -> walk
    braced = Session(seed)
    braced.teach("B")                       # cue B -> brace
    birth = Session(seed).weights()

    def arm(name, weights, note=""):
        session = Session(seed, weights=weights)
        stages = session.exam()
        out["arms"][name] = dict(note=note, stages=stages,
                                 walks_on_A=stages["A"]["moved"],
                                 brace_on_B=stages["B"]["flexion"])

    w_walk, w_brace = walker.weights(), braced.weights()
    arm("birth", birth, "no lesson at all")
    arm("parent_walk", w_walk, "taught cue A -> walk, nothing else")
    arm("parent_brace", w_brace, "taught cue B -> brace, nothing else")
    arm("mean(walk,brace)", .5 * (w_walk + w_brace), "add the two weight tables and halve")
    arm("keep(walk,brace)", merge_keep(w_walk, w_brace, birth),
        "per edge, keep whichever parent moved it further from birth")
    arm("mean(walk,walk)", .5 * (w_walk + w_walk), "sanity: a parent merged with itself")

    moves_walk, moves_brace = w_walk - birth, w_brace - birth
    # A move of any size and a move big enough to matter are counted
    # separately: the innate edges are plastic too and drift a little.
    big_walk, big_brace = np.abs(moves_walk) > .05, np.abs(moves_brace) > .05
    dst = walker.brain.synapses.dst
    action_cells = {name: int(np.ravel(walker.brain.groups[name])[0])
                    for name in CONFIG["empty_routes"]}
    out["graph"] = dict(
        neurons=int(walker.brain.network.n_neurons), edges=int(len(birth)),
        empty_routes=list(CONFIG["empty_routes"]), action_cells=action_cells,
        edges_moved_by_walk=int(np.count_nonzero(big_walk)),
        edges_moved_by_brace=int(np.count_nonzero(big_brace)),
        edges_moved_by_walk_any_size=int(np.count_nonzero(np.abs(moves_walk) > 1e-9)),
        edges_moved_by_brace_any_size=int(np.count_nonzero(np.abs(moves_brace) > 1e-9)),
        edges_moved_in_opposite_directions_above_0_05=int(np.count_nonzero(
            (moves_walk * moves_brace < 0) & big_walk & big_brace)),
        edges_touched_by_both_above_0_05=int(np.count_nonzero(big_walk & big_brace)),
        walk_edges_per_action_cell={name: int(np.count_nonzero(big_walk & (dst == cell)))
                                    for name, cell in action_cells.items()},
        brace_edges_per_action_cell={name: int(np.count_nonzero(big_brace & (dst == cell)))
                                     for name, cell in action_cells.items()},
        max_abs_change_walk=float(np.max(np.abs(moves_walk))),
        max_abs_change_brace=float(np.max(np.abs(moves_brace))))
    out["moved_edges"] = dict(
        walk=[[int(s), int(d), float(v)] for s, d, v in
              zip(walker.brain.synapses.src[big_walk], dst[big_walk], moves_walk[big_walk])],
        brace=[[int(s), int(d), float(v)] for s, d, v in
               zip(walker.brain.synapses.src[big_brace], dst[big_brace], moves_brace[big_brace])])
    return out


def main(argv=None):
    started = time.perf_counter()
    seeds = [int(a) for a in (sys.argv[1:] if argv is None else argv)] or [0, 1, 2]
    cases = [case(seed) for seed in seeds]
    report = dict(python=platform.python_version(), numpy=np.__version__,
                  mujoco_version=mujoco_version(), config=CONFIG,
                  command="python tools/fuse_robot_parents.py " + " ".join(str(s) for s in seeds),
                  seeds=seeds, cases=cases, wall_seconds=time.perf_counter() - started,
                  scope=("One Go2 body, one graph, one edge list and one birth weight vector for "
                         "every arm; the only difference between parents is which cue-action "
                         "pairing was taught. Merging is an arithmetic operation on the weight "
                         "vector and nothing is retrained afterwards."))
    out = Path(__file__).resolve().parents[1] / "artifacts" / "fusion_robot_parents.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                   encoding="utf-8")
    for data in cases:
        print("seed", data["seed"], data["graph"])
        for name, a in data["arms"].items():
            print("   %-20s walked on cue A %+0.3f m   brace on cue B %+0.3f   "
                  "(cue B locomotion %+0.3f, cue A flexion %+0.3f)"
                  % (name, a["walks_on_A"], a["brace_on_B"],
                     a["stages"]["B"]["locomotion"], a["stages"]["A"]["flexion"]))
    print("wall %.1f s -> %s" % (report["wall_seconds"], out))


def mujoco_version():
    import mujoco
    return mujoco.__version__


if __name__ == "__main__":
    main()
