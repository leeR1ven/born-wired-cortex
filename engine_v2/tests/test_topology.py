import unittest

import numpy as np

from born_wired.topology import make_edges


class TopologyTests(unittest.TestCase):
    def test_per_source_nearest_and_remaining_random(self):
        for n in (16, 32, 64):
            positions = np.column_stack((np.arange(n), np.arange(n) % 3))
            for seed in range(5):
                with self.subTest(n=n, seed=seed):
                    src, dst = make_edges(positions, 3, 2, seed)
                    self.assertEqual(src.dtype, np.dtype("int64"))
                    self.assertEqual(dst.dtype, np.dtype("int64"))
                    self.assertEqual(src.shape, (n * 5,))
                    np.testing.assert_array_equal(np.bincount(src, minlength=n), 5)
                    for source in range(n):
                        targets = dst[src == source]
                        expected = sorted((j for j in range(n) if j != source),
                                          key=lambda j: (sum((positions[j] - positions[source]) ** 2), j))
                        np.testing.assert_array_equal(targets[:3], expected[:3])
                        self.assertEqual(len(set(targets)), 5)
                        self.assertNotIn(source, targets)
                        self.assertTrue(set(targets[3:]).isdisjoint(expected[:3]))
                    again = make_edges(positions, 3, 2, seed)
                    np.testing.assert_array_equal(dst, again[1])

    def test_ties_use_index_including_duplicate_coordinates(self):
        positions = np.array([[0., 0.], [1., 0.], [-1., 0.], [0., 0.]])
        src, dst = make_edges(positions, 3, 0)
        np.testing.assert_array_equal(dst[src == 0], [3, 1, 2])
        np.testing.assert_array_equal(dst[src == 3], [0, 1, 2])

    def test_empty_random_only_and_complete(self):
        positions = np.arange(8).reshape(-1, 1)
        src, dst = make_edges(positions, 0, 0)
        self.assertEqual(src.shape, (0,))
        self.assertEqual(dst.shape, (0,))
        self.assertEqual(dst.dtype, np.dtype("int64"))
        for local, random in ((0, 7), (7, 0), (3, 4)):
            src, dst = make_edges(positions, local, random)
            for source in range(8):
                self.assertEqual(set(dst[src == source]), set(range(8)) - {source})

    def test_different_random_seeds(self):
        positions = np.arange(32).reshape(-1, 1)
        self.assertFalse(np.array_equal(make_edges(positions, 0, 3, 0)[1],
                                        make_edges(positions, 0, 3, 1)[1]))

    def test_invalid_positions(self):
        bad_inputs = ([], [0, 1], [[0]], np.empty((2, 0)), np.zeros((2, 2, 1)),
                      [[0], [np.nan]], [[np.inf], [0]], [[-np.inf], [0]],
                      [[True], [False]], [["1"], ["2"]], [[1j], [0j]])
        for positions in bad_inputs:
            with self.subTest(positions=positions), self.assertRaises(ValueError):
                make_edges(positions, 0, 0)

    def test_invalid_degrees(self):
        positions = np.arange(4).reshape(-1, 1)
        for value in (True, np.bool_(False), -1, 1., np.nan, np.inf, "1", None):
            for local, random in ((value, 0), (0, value)):
                with self.subTest(local=local, random=random), self.assertRaises(ValueError):
                    make_edges(positions, local, random)
        with self.assertRaises(ValueError):
            make_edges(positions, 2, 2)
        self.assertEqual(make_edges(positions, np.int64(1), np.int32(1))[0].size, 8)

    def test_finite_extreme_coordinates_and_no_input_mutation(self):
        for positions in (np.array([[-1e308], [1e308], [0.]]),
                          np.array([[0.], [1e-300], [2e-300]])):
            original = positions.copy()
            src, dst = make_edges(positions, 1, 0)
            np.testing.assert_array_equal(positions, original)
            self.assertEqual(dst[src == 0][0], 2 if positions[0, 0] < 0 else 1)


if __name__ == "__main__":
    unittest.main()
