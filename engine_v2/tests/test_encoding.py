"""Targeted tests for task 04 population encoding."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from born_wired.encoding import RecruitmentEncoder, TuningEncoder


class EncodingTests(unittest.TestCase):
    def test_constructor_defaults(self) -> None:
        recruitment = RecruitmentEncoder()
        tuning = TuningEncoder()
        self.assertEqual(
            (recruitment.n_channels, recruitment.neurons_per_channel),
            (12, 100),
        )
        self.assertEqual(
            (tuning.n_channels, tuning.neurons_per_channel), (12, 32)
        )

    def test_public_interface_and_boundaries(self) -> None:
        for encoder in (
            RecruitmentEncoder(n_channels=3, neurons_per_channel=10),
            TuningEncoder(n_channels=3, neurons_per_channel=10),
        ):
            self.assertEqual(encoder.n_channels, 3)
            self.assertEqual(encoder.neurons_per_channel, 10)
            self.assertEqual(encoder.size, 30)

            boundary = encoder.encode(np.array([0.0, 1.0, 0.5]))
            self.assertEqual(boundary.shape, (3, 10))
            self.assertTrue(np.all(boundary >= 0.0))
            self.assertTrue(np.all(boundary <= 1.0))
            if isinstance(encoder, RecruitmentEncoder):
                self.assertEqual(np.count_nonzero(boundary[0]), 0)
                self.assertEqual(np.count_nonzero(boundary[1]), 10)
            else:
                self.assertEqual(np.count_nonzero(boundary[0]), 1)
                self.assertEqual(np.count_nonzero(boundary[1]), 1)
            np.testing.assert_allclose(encoder.decode(boundary), [0.0, 1.0, 0.5])

    def test_channel_independence(self) -> None:
        values = np.array([0.2, 0.3, 0.4, 0.6, 0.7, 0.8])
        changed_values = values.copy()
        changed_values[2] = 0.9

        for encoder in (
            RecruitmentEncoder(n_channels=6, neurons_per_channel=10),
            TuningEncoder(n_channels=6, neurons_per_channel=10),
        ):
            base = encoder.encode(values)
            changed = encoder.encode(changed_values)
            other = np.arange(6) != 2
            np.testing.assert_array_equal(base[other], changed[other])
            self.assertFalse(np.array_equal(base[2], changed[2]))

    def test_invalid_configuration_is_rejected(self) -> None:
        invalid = (
            (True, 10),
            (0, 10),
            (-1, 10),
            (1.5, 10),
            (2, True),
            (2, 1),
            (2, 2.0),
        )
        for channels, neurons in invalid:
            with self.subTest(channels=channels, neurons=neurons):
                with self.assertRaises(ValueError):
                    RecruitmentEncoder(channels, neurons)
                with self.assertRaises(ValueError):
                    TuningEncoder(channels, neurons)

    def test_invalid_values_and_activity_are_rejected(self) -> None:
        for encoder in (
            RecruitmentEncoder(n_channels=3, neurons_per_channel=10),
            TuningEncoder(n_channels=3, neurons_per_channel=10),
        ):
            bad_values = (
                np.zeros((3, 1)),
                np.zeros(2),
                np.array([0.0, np.nan, 0.5]),
                np.array([0.0, np.inf, 0.5]),
                np.array([-0.1, 0.5, 0.5]),
                np.array([0.0, 1.1, 0.5]),
                np.array([False, True, False]),
            )
            for values in bad_values:
                with self.subTest(values=values):
                    with self.assertRaises(ValueError):
                        encoder.encode(values)

            bad_activity = (
                np.zeros((3,)),
                np.zeros((2, 10)),
                np.zeros((3, 9)),
                np.full((3, 10), np.nan),
                np.full((3, 10), 1.1),
            )
            for activity in bad_activity:
                with self.subTest(activity_shape=activity.shape):
                    with self.assertRaises(ValueError):
                        encoder.decode(activity)

        with self.assertRaises(ValueError):
            TuningEncoder(3, 10).decode(np.zeros((3, 10)))

    def test_recruitment_roundtrip_and_fine_difference(self) -> None:
        values = np.linspace(0.0, 1.0, 101)
        for neurons in (10, 25, 50, 100):
            encoder = RecruitmentEncoder(n_channels=1, neurons_per_channel=neurons)
            inputs = np.array([[value] for value in values])
            decoded = np.array(
                [
                    encoder.decode(encoder.encode(single))
                    for single in inputs
                ]
            )
            self.assertLessEqual(
                float(np.max(np.abs(decoded - values[:, np.newaxis]))),
                0.5 / neurons + 1e-12,
            )

        pattern_050 = np.full(12, 0.5)
        pattern_051 = pattern_050.copy()
        pattern_051[0] = 0.51
        old = RecruitmentEncoder(12, 10)
        fine = RecruitmentEncoder(12, 100)
        np.testing.assert_array_equal(
            old.encode(pattern_050), old.encode(pattern_051)
        )
        self.assertFalse(
            np.array_equal(fine.encode(pattern_050), fine.encode(pattern_051))
        )

    def test_tuning_readback_and_fine_difference(self) -> None:
        values = np.linspace(0.0, 1.0, 501)
        for neurons in (2, 10, 100):
            encoder = TuningEncoder(n_channels=1, neurons_per_channel=neurons)
            inputs = np.array([[value] for value in values])
            decoded = np.array(
                [
                    encoder.decode(encoder.encode(single))
                    for single in inputs
                ]
            )
            np.testing.assert_allclose(
                decoded, values[:, np.newaxis], atol=1e-12, rtol=0.0
            )

        pattern_050 = np.full(12, 0.5)
        pattern_051 = pattern_050.copy()
        pattern_051[0] = 0.51
        old = TuningEncoder(12, 10)
        fine = TuningEncoder(12, 100)
        np.testing.assert_allclose(
            old.decode(old.encode(pattern_051))[0], 0.51, atol=1e-12
        )
        self.assertFalse(
            np.array_equal(old.encode(pattern_050), old.encode(pattern_051))
        )
        self.assertFalse(
            np.array_equal(fine.encode(pattern_050), fine.encode(pattern_051))
        )

    def test_returns_fresh_arrays(self) -> None:
        encoder = TuningEncoder(n_channels=2, neurons_per_channel=8)
        first = encoder.encode([0.25, 0.75])
        second = encoder.encode([0.25, 0.75])
        self.assertFalse(np.shares_memory(first, second))
        first[0, 0] = 0.0
        np.testing.assert_array_equal(
            encoder.encode([0.25, 0.75]), second
        )
        centers = encoder.centers
        centers[0] = 0.5
        np.testing.assert_allclose(encoder.centers[0], 0.0)


if __name__ == "__main__":
    unittest.main()
