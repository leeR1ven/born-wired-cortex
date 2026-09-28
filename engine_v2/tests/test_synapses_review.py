import unittest

import numpy as np

from born_wired.synapses import BoundedSynapses


class SynapseReviewTests(unittest.TestCase):
    def test_mixed_frozen_edges_reserve_both_budget_types(self):
        syn = BoundedSynapses(
            [0, 1, 2, 3],
            [4, 4, 4, 4],
            [0.2, 0.1, 0.15, 0.05],
            [1, 1, -1, -1, 1],
            5,
            budgets=0.3,
            plasticity=[0, 1, 0, 1],
            learning_rate=0.1,
            decay=0,
        )
        frozen = syn.weights[[0, 2]].copy()

        syn.update(np.ones(5), np.ones(5), dt=1000)

        np.testing.assert_array_equal(syn.weights[[0, 2]], frozen)
        self.assertAlmostEqual(syn.weights[1], 0.1)
        self.assertAlmostEqual(syn.weights[3], 0.15)
        exc, inh = syn.budget_totals()
        self.assertAlmostEqual(exc[4], 0.3)
        self.assertAlmostEqual(inh[4], 0.3)

    def test_vector_modulator_is_indexed_by_postsynaptic_cell(self):
        syn = BoundedSynapses(
            [0, 1],
            [2, 3],
            [0.1, 0.1],
            [1, 1, 1, 1],
            4,
            learning_rate=0.1,
            decay=0,
        )

        syn.update(
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            modulator=[0, 0, 1, 0],
            dt=0.02,
        )

        self.assertAlmostEqual(syn.weights[0], 0.102)
        self.assertAlmostEqual(syn.weights[1], 0.1)

    def test_invalid_numeric_constructor_inputs_are_rejected(self):
        bad_values = (
            {"weights": [0.1, np.nan]},
            {"weights": [0.1, np.inf]},
            {"signs": [1, 0, 1, 1]},
            {"w_max": [1.0, np.nan]},
            {"budgets": [1.0, np.inf, 1.0, 1.0]},
            {"plasticity": [0.0, np.inf]},
        )
        for override in bad_values:
            values = {
                "src": [0, 1],
                "dst": [2, 2],
                "weights": [0.1, 0.1],
                "signs": [1, 1, 1, 1],
                "n_neurons": 4,
            }
            values.update(override)
            with self.subTest(override=override), self.assertRaises(ValueError):
                BoundedSynapses(**values)


if __name__ == "__main__":
    unittest.main()
