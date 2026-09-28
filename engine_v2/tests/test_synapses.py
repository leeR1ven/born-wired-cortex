import unittest
import numpy as np

from born_wired.synapses import BoundedSynapses


class SynapseTests(unittest.TestCase):
    def make(self, **kwargs):
        return BoundedSynapses([0, 1, 2], [3, 3, 3], [.3, .4, .5], [1, 1, -1, 1], 4, **kwargs)

    def assert_bounds(self, syn):
        self.assertTrue(np.all(np.isfinite(syn.weights)))
        self.assertTrue(np.all(syn.weights >= 0))
        self.assertTrue(np.all(syn.weights <= syn.w_max))
        for total in syn.budget_totals():
            self.assertTrue(np.all(total <= syn.budgets + 1e-12))

    def test_independent_exc_inh_budgets_and_currents(self):
        syn = self.make(budgets=.5)
        self.assert_bounds(syn)
        exc, inh = syn.currents([1, 1, 1, 0])
        self.assertAlmostEqual(exc[3], .5)
        self.assertAlmostEqual(inh[3], .5)
        self.assertEqual(float(exc[:3].sum() + inh[:3].sum()), 0)

    def test_correlation_learns_and_negative_modulation_weakens(self):
        syn = BoundedSynapses([0, 1], [2, 2], [.1, .1], [1, 1, 1], 3,
                              learning_rate=.1, decay=0, budgets=2)
        for _ in range(100):
            syn.update([1, 0, 0], [0, 0, 1])
        self.assertGreater(syn.weights[0], syn.weights[1] + .15)
        previous = syn.weights.copy()
        syn.update([1, 0, 0], [0, 0, 1], modulator=-1)
        self.assertLess(syn.weights[0], previous[0])
        self.assertEqual(syn.weights[1], previous[1])

    def test_frozen_synapses_keep_their_budget(self):
        syn = self.make(budgets=.5, plasticity=[0, 1, 1])
        initial = syn.weights.copy()
        for _ in range(50):
            syn.update(np.ones(4), np.ones(4), dt=100)
            self.assertEqual(syn.weights[0], initial[0])
            self.assert_bounds(syn)
        all_frozen = self.make(plasticity=0)
        before = all_frozen.weights.copy()
        all_frozen.update(np.ones(4), np.ones(4), dt=1e6)
        np.testing.assert_array_equal(all_frozen.weights, before)

    def test_impossible_frozen_initialization_is_rejected(self):
        with self.assertRaises(ValueError):
            self.make(plasticity=0, budgets=.1)
        with self.assertRaises(ValueError):
            self.make(plasticity=0, w_max=.2)

    def test_bad_update_is_atomic(self):
        syn = self.make()
        before = syn.weights.copy()
        bad_calls = [
            dict(pre=[0, 0, float('nan'), 1]),
            dict(post=[0, 0, 0, 2]), dict(modulator=float('inf')),
            dict(modulator=2), dict(dt=0), dict(dt=float('nan')),
            dict(pre=[1, 0]), dict(modulator=[0, 0]), dict(dt=True),
        ]
        for bad in bad_calls:
            args = dict(pre=np.ones(4), post=np.ones(4))
            args.update(bad)
            with self.assertRaises((ValueError, FloatingPointError)):
                syn.update(**args)
            np.testing.assert_array_equal(syn.weights, before)
        overflow = self.make(learning_rate=1e308)
        before = overflow.weights.copy()
        with self.assertRaises(FloatingPointError):
            overflow.update(np.ones(4), np.ones(4), dt=1e308)
        np.testing.assert_array_equal(overflow.weights, before)

    def test_constructor_validation(self):
        bad_kwargs = [dict(learning_rate=-1), dict(decay=float('nan')),
                      dict(budgets=-1), dict(w_max=float('inf')), dict(plasticity=-1),
                      dict(budgets=[1, 2]), dict(learning_rate=True)]
        for kwargs in bad_kwargs:
            with self.assertRaises(ValueError):
                self.make(**kwargs)
        for src, dst in [([0, 0], [1, 1]), ([0, 1], [0, 2]),
                         ([0.5, 1], [1, 2]), ([True, False], [1, 2]),
                         ([-1, 0], [1, 2]), ([0, 1], [1, 3])]:
            with self.assertRaises(ValueError):
                BoundedSynapses(src, dst, [.1, .2], [1, -1, 1], 3)

    def test_external_arrays_cannot_change_internal_state(self):
        src, dst = np.array([0, 1]), np.array([2, 2])
        weights, signs = np.array([.2, .2]), np.array([1, -1, 1])
        caps, budget, plasticity = np.ones(2), np.ones(3), np.ones(2)
        syn = BoundedSynapses(src, dst, weights, signs, 3, w_max=caps,
                              budgets=budget, plasticity=plasticity)
        src[:] = 2; dst[:] = 0; weights[:] = 99; signs[:] = -1
        caps[:] = 0; budget[:] = 0; plasticity[:] = 0
        self.assertEqual(syn.src.tolist(), [0, 1])
        self.assertEqual(syn.signs.tolist(), [1, -1, 1])
        np.testing.assert_array_equal(syn.weights, [.2, .2])
        snapshot = syn.weights
        snapshot.flags.writeable = True
        snapshot[:] = 99
        np.testing.assert_array_equal(syn.weights, [.2, .2])
        syn.update([1, 1, 0], [0, 0, 1])
        self.assertGreater(syn.weights[0], .2)

    def test_large_finite_dt_and_zero_capacity(self):
        syn = self.make()
        for modulator in [1, -1, 1, 1, -1]:
            syn.update(np.ones(4), np.ones(4), modulator, dt=1e12)
            self.assert_bounds(syn)
        empty = BoundedSynapses([], [], [], [1, -1], 2, budgets=0)
        empty.update([1, 1], [1, 1])
        self.assertEqual(len(empty.weights), 0)
        for current in empty.currents([1, 1]):
            np.testing.assert_array_equal(current, [0, 0])


if __name__ == '__main__':
    unittest.main()
