import unittest
import numpy as np
from born_wired.regulation import RegulatedSynapses


class RegulationTests(unittest.TestCase):
    def test_budget_reserves_scaffold_and_frozen_edges(self):
        s = RegulatedSynapses([0,1,2], [3,3,3], [.4,.3,.2], [1,1,-1,1], 4,
                              lower=[.35,.1,.15], upper=1, budgets=.8, plasticity=[0,1,1])
        for _ in range(100):
            s.update(np.ones(4), np.ones(4), dt=1)
        self.assertEqual(s.weights[0], .4)
        self.assertTrue(np.all(s.weights >= s.lower))
        self.assertLessEqual(s.budget_totals()[0][3], .8 + 1e-12)
        self.assertGreater(s.weights[1], .3)

    def test_impossible_scaffold_rejected(self):
        with self.assertRaises(ValueError):
            RegulatedSynapses([0,1], [2,2], [.6,.6], [1,1,1], 3, lower=.6, budgets=1)

    def test_memory_can_learn_without_moving_scaffold_far(self):
        s = RegulatedSynapses([0,1], [2,2], [.5,.001], [1,1,1], 3,
                              lower=[.49,0], upper=[.51,1], budgets=2, tether=[.1,0])
        for _ in range(1000):
            s.update(np.ones(3), np.ones(3), dt=.02)
        self.assertLessEqual(s.weights[0], .51)
        self.assertGreater(s.weights[0], .5)
        self.assertGreater(s.weights[1], .8)

    def test_atomic_overflow(self):
        s = RegulatedSynapses([0], [1], [.5], [1,1], 2, learning_rate=1e308, plasticity=1e308)
        before = s.weights.copy()
        with self.assertRaises(FloatingPointError):
            s.update(np.ones(2), np.ones(2), dt=100)
        np.testing.assert_array_equal(s.weights, before)


if __name__ == '__main__':
    unittest.main()
