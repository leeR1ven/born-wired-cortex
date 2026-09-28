"""The exam bank and the screening loop, tested without a body.

These tests never step MuJoCo.  They cover the parts that decide what is asked
and what is kept: which genes may move, how two genomes are merged, whose
abilities a round keeps, and that every bar says no when it should.
"""
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import screen_candidates as screen      # noqa: E402
from tools import taskbank                          # noqa: E402


class BankShapeTest(unittest.TestCase):
    def test_every_task_is_well_formed(self):
        names = [task["name"] for task in taskbank.TASKS]
        self.assertEqual(len(names), len(set(names)))
        self.assertGreaterEqual(len(names), 12)
        for task in taskbank.TASKS:
            self.assertIn(task["stage"], ("screen", "full"))
            self.assertTrue(callable(task["measure"]))
            self.assertTrue(callable(task["bar"]))
            self.assertGreater(task["sim_seconds"], 0.)
            self.assertTrue(task["question"])
            self.assertTrue(task["note"])

    def test_screen_is_a_real_subset(self):
        screen_names = {task["name"] for task in taskbank.stage_tasks("screen")}
        full_names = {task["name"] for task in taskbank.stage_tasks("full")}
        self.assertTrue(screen_names < full_names)
        self.assertGreaterEqual(len(screen_names), 4)

    def test_genes_are_parameters_and_not_sizes(self):
        parameters = taskbank.reference_parameters()
        for name in taskbank.GENES:
            self.assertIn(name, parameters)
            self.assertNotIn(name, taskbank.SIZE_KEYS)
        self.assertGreaterEqual(len(taskbank.GENES), 20)

    def test_a_genome_may_not_name_something_else(self):
        with self.assertRaises(ValueError):
            taskbank.reference_parameters({"not_a_gene": 1.})


class MutationTest(unittest.TestCase):
    def test_only_named_genes_move(self):
        child, touched = screen.mutate({}, 0, np.random.default_rng(3), genes=4)
        self.assertEqual(set(child), set(touched))
        self.assertEqual(len(child), 4)
        self.assertTrue(set(child) <= set(taskbank.GENES))

    def test_the_same_seed_gives_the_same_child(self):
        first, _ = screen.mutate({}, 0, np.random.default_rng(7), genes=3)
        second, _ = screen.mutate({}, 0, np.random.default_rng(7), genes=3)
        self.assertEqual(first, second)

    def test_whole_number_genes_stay_whole(self):
        base = taskbank.reference_parameters({})
        rng = np.random.default_rng(5)
        for _ in range(20):
            child, _ = screen.mutate({}, 0, rng, genes=len(taskbank.GENES))
            for name in taskbank.INTEGER_GENES:
                self.assertEqual(float(child[name]), round(float(child[name])),
                                 "%s was blended into a fraction" % name)
                self.assertNotEqual(float(child[name]), float(base[name]))

    def test_a_positive_gene_stays_positive(self):
        base = taskbank.reference_parameters({})
        rng = np.random.default_rng(9)
        for _ in range(20):
            child, _ = screen.mutate({}, 0, rng, genes=len(taskbank.GENES), sigma=3.)
            for name, value in child.items():
                if float(base[name]) > 0.:
                    self.assertGreater(float(value), 0., "%s went non-positive" % name)


class MergeTest(unittest.TestCase):
    def test_merge_is_the_mean_of_the_genes(self):
        first = {"a": 1., "b": 3.}
        second = {"a": 3., "b": 5.}
        self.assertEqual(screen.merge_genome(first, second), {"a": 2., "b": 4.})

    def test_merging_with_itself_changes_nothing(self):
        genome = {"a": 1.5, "b": -2.}
        self.assertEqual(screen.merge_genome(genome, genome), genome)

    def test_merge_keeps_a_gene_only_one_parent_set(self):
        merged = screen.merge_genome({"a": 4.}, {"b": 2.})
        self.assertEqual(merged, {"a": 2., "b": 1.})


class SelectionTest(unittest.TestCase):
    @staticmethod
    def record(label, passed):
        return dict(kind="candidate", label=label, scaffold=0, passed_tasks=sorted(passed),
                    n_passed=len(passed), genome={})

    def test_cover_takes_the_animal_that_adds_the_most(self):
        # The bank is covered when every task anybody passed is covered, not
        # when the best single animal has been found.
        records = [self.record("a", ["walk", "look"]),
                   self.record("b", ["walk", "get_up"]),
                   self.record("c", ["walk", "get_up", "hear"])]
        chosen, covered = screen.select(records, 0)
        self.assertEqual(chosen[0]["label"], "c")
        self.assertEqual(covered, ["get_up", "hear", "look", "walk"])
        self.assertEqual([record["label"] for record in chosen], ["c", "a"])

    def test_cover_prefers_two_that_do_not_overlap(self):
        records = [self.record("a", ["walk", "look"]),
                   self.record("b", ["get_up", "hear"])]
        chosen, covered = screen.select(records, 0)
        self.assertEqual(len(chosen), 2)
        self.assertEqual(set(covered), {"walk", "look", "get_up", "hear"})

    def test_cover_ignores_other_scaffolds(self):
        records = [self.record("a", ["walk"]), dict(self.record("b", ["hear"]), scaffold=1)]
        chosen, _ = screen.select(records, 0)
        self.assertEqual([record["label"] for record in chosen], ["a"])


class LedgerTest(unittest.TestCase):
    def test_a_line_written_is_a_line_read(self):
        with tempfile.TemporaryDirectory() as folder:
            ledger = Path(folder) / "ledger.jsonl"
            screen.append({"kind": "candidate", "label": "x"}, ledger)
            screen.append({"kind": "candidate", "label": "y"}, ledger)
            records = screen.read_ledger(ledger)
        self.assertEqual([record["label"] for record in records], ["x", "y"])

    def test_reading_a_missing_ledger_is_empty(self):
        self.assertEqual(screen.read_ledger(Path("does-not-exist.jsonl")), [])


class BarTest(unittest.TestCase):
    def test_a_walk_that_falls_over_fails(self):
        measures = dict(status="ok", error=None, behavior_pass=False,
                        displacement_m=5., min_up_z=.2, min_height_m=.05)
        passed, why = taskbank.walk_bar(measures, "walk_flat_m")
        self.assertFalse(passed)
        self.assertIn("upright", why)

    def test_a_walk_that_stays_up_and_goes_far_passes(self):
        measures = dict(status="ok", error=None, behavior_pass=True,
                        displacement_m=5., min_up_z=.9, min_height_m=.2)
        passed, _ = taskbank.walk_bar(measures, "walk_flat_m")
        self.assertTrue(passed)

    def test_getting_up_and_falling_back_down_fails(self):
        measures = dict(status="ok", error=None, recovered=True, final_up_z=-.7,
                        final_height=.1, minimum_up_z=-1., time_to_feet_s=2.1,
                        on_feet_share=.3, pose="on_back")
        passed, why = taskbank.righting_bar(measures)
        self.assertFalse(passed)
        self.assertIn("back down", why)

    def test_a_failed_run_never_passes_a_bar(self):
        failed = dict(status="error", error="boom")
        for bar in ((lambda m: taskbank.walk_bar(m, "walk_flat_m")),
                    taskbank.righting_bar, taskbank.quiet_bar, taskbank.sound_bar,
                    taskbank.taught_reflex_bar, taskbank.visual_memory_bar,
                    taskbank.approach_bar, taskbank.slip_bar,
                    lambda m: taskbank.sudden_bar(m, "sudden_shift_rad")):
            passed, why = bar(failed)
            self.assertFalse(passed)
            self.assertTrue(why)

    def test_a_right_hand_sound_must_pull_the_eyes_right(self):
        measures = dict(status="ok", right=dict(status="ok", eye_yaw=-.20),
                        left=dict(status="ok", eye_yaw=.02))
        self.assertTrue(taskbank.sound_bar(measures)[0])
        measures["right"]["eye_yaw"] = .02
        self.assertFalse(taskbank.sound_bar(measures)[0])

    def test_the_eyes_must_swing_towards_the_new_thing(self):
        good = dict(status="ok", error=None, gaze_shift_rad=.05, bearing_after_rad=.4,
                    error_after_appearing_rad=.3)
        self.assertTrue(taskbank.sudden_bar(good, "sudden_shift_rad")[0])
        wrong_way = dict(good, gaze_shift_rad=-.05)
        self.assertFalse(taskbank.sudden_bar(wrong_way, "sudden_shift_rad")[0])
        too_small = dict(good, gaze_shift_rad=.005)
        self.assertFalse(taskbank.sudden_bar(too_small, "sudden_shift_rad")[0])

    def test_the_distance_bank_must_start_empty_and_end_full(self):
        good = dict(status="ok", error=None, bank_far=.1, bank_near=5.6, near=.25, far=2.)
        self.assertTrue(taskbank.approach_bar(good)[0])
        self.assertFalse(taskbank.approach_bar(dict(good, bank_near=1.))[0])
        self.assertFalse(taskbank.approach_bar(dict(good, bank_far=4.))[0])


class LadderTest(unittest.TestCase):
    """The questions are rungs: some only make sense once simpler ones are up."""

    def test_every_prerequisite_is_a_task_in_the_bank(self):
        names = {task["name"] for task in taskbank.TASKS}
        for task in taskbank.TASKS:
            for need in task["requires"]:
                self.assertIn(need, names, "%s needs %s" % (task["name"], need))

    def test_the_ladder_has_no_cycles(self):
        depth = taskbank.ladder()
        self.assertEqual(set(depth), {task["name"] for task in taskbank.TASKS})

    def test_nothing_requires_itself(self):
        for task in taskbank.TASKS:
            self.assertNotIn(task["name"], task["requires"])

    def test_the_first_rung_asks_nothing_of_the_animal(self):
        bottom = [task["name"] for task in taskbank.TASKS if not task["requires"]]
        self.assertGreaterEqual(len(bottom), 4)
        self.assertEqual(taskbank.ladder()[bottom[0]], 0)

    def test_a_deeper_rung_needs_something_under_it(self):
        depth = taskbank.ladder()
        for task in taskbank.TASKS:
            if task["requires"]:
                self.assertGreater(depth[task["name"]],
                                   max(depth[need] for need in task["requires"]) - 1)

    def test_an_empty_animal_can_only_be_asked_the_first_rung(self):
        open_now = taskbank.frontier([])
        self.assertEqual(sorted(open_now),
                         sorted(task["name"] for task in taskbank.TASKS
                                if not task["requires"]))

    def test_passing_a_rung_opens_the_next_one(self):
        first = [task["name"] for task in taskbank.TASKS if not task["requires"]][0]
        opened = set(taskbank.frontier([first]))
        for task in taskbank.TASKS:
            if task["requires"] == (first,):
                self.assertIn(task["name"], opened)

    def test_the_deepest_rung_counts_what_it_says(self):
        depth = taskbank.ladder()
        deep = [name for name, value in depth.items() if value == max(depth.values())]
        top, reached = taskbank.deepest_reached(deep)
        self.assertEqual(top, max(depth.values()))
        self.assertEqual(sorted(reached), sorted(deep))

    def test_nothing_passed_means_rung_zero(self):
        self.assertEqual(taskbank.deepest_reached([]), (0, []))


class EveryBarTest(unittest.TestCase):
    """A bar that cannot say no is not a bar."""

    def test_every_bar_refuses_a_failed_run(self):
        failed = {"status": "error", "error": "boom"}
        for task in taskbank.TASKS:
            passed, why = task["bar"](failed)
            self.assertFalse(passed, "%s passed a run that failed" % task["name"])
            self.assertTrue(why, "%s gave no reason" % task["name"])

    def test_every_task_says_what_it_reads_and_what_it_wants(self):
        for task in taskbank.TASKS:
            self.assertTrue(task["reads"], task["name"])
            self.assertTrue(task["bar_text"], task["name"])

    def test_a_bar_key_is_a_bar_key(self):
        for task in taskbank.TASKS:
            self.assertTrue(task["bar_text"])

    def test_separation_bar_wants_a_different_pattern(self):
        close = {"status": "ok", "separation": {"pinna": .01}}
        far = {"status": "ok", "separation": {"pinna": .50}}
        self.assertFalse(taskbank.separation_bar(close, "pinna", "pinna_gain", "x")[0])
        self.assertTrue(taskbank.separation_bar(far, "pinna", "pinna_gain", "x")[0])

    def test_contrast_bar_wants_a_real_gap(self):
        small = {"status": "ok", "gain": {"retina": .001}}
        big = {"status": "ok", "gain": {"retina": .20}}
        self.assertFalse(taskbank.contrast_bar(small, "retina", "pattern_gain", "x")[0])
        self.assertTrue(taskbank.contrast_bar(big, "retina", "pattern_gain", "x")[0])

    def test_a_negative_result_can_be_the_right_answer(self):
        quiet = {"status": "ok", "aversive_gain": 0.}
        noisy = {"status": "ok", "aversive_gain": .5}
        self.assertTrue(taskbank.visual_memory_bar(quiet, "visual_memory_control",
                                                   above=False)[0])
        self.assertFalse(taskbank.visual_memory_bar(noisy, "visual_memory_control",
                                                    above=False)[0])

    def test_the_wall_bar_wants_a_stop(self):
        stopped = dict(status="ok", behavior_pass=True, displacement_m=1., min_up_z=.9,
                       min_height_m=.2, maximum_x=3.10, peak={"wall_contact": .3})
        self.assertTrue(taskbank.wall_bar(stopped, "walk_into_wall_m")[0])
        through = dict(stopped, maximum_x=3.60)
        self.assertFalse(taskbank.wall_bar(through, "walk_into_wall_m")[0])


if __name__ == "__main__":
    unittest.main()