"""Synthetic waveform checks only; no claim of biological hearing/localization."""
import json
import time
import unittest

import numpy as np

from born_wired.auditory_neurons import AuditoryNeurons


def tone(frequency, *, delay=0, amplitude=(.8, .8), samples=8192):
    t = np.arange(samples)/16000
    return np.array([amplitude[0]*np.sin(2*np.pi*frequency*t),
                     amplitude[1]*np.sin(2*np.pi*frequency*(t-delay/16000))])


class AuditoryNeuronsTests(unittest.TestCase):
    def test_three_tones_prefer_their_band(self):
        for band, frequency in enumerate((262, 523, 880)):
            with self.subTest(frequency=frequency):
                output = AuditoryNeurons().step(tone(frequency))
                rates = output["rates"]
                self.assertTrue(np.all(rates[:, band] > .4))
                others = [index for index in range(3) if index != band]
                self.assertTrue(np.all(rates[:, band, None] > rates[:, others]+.25))
                np.testing.assert_allclose(rates[0], rates[1], atol=1e-13)
                self.assertAlmostEqual(*output["orientation"], places=12)

    def test_signed_delay_and_amplitude_preference(self):
        for frequency in (262, 523, 880):
            for delay, side in ((3, 0), (-3, 1)):
                with self.subTest(frequency=frequency, delay=delay):
                    output = AuditoryNeurons().step(tone(frequency, delay=delay))
                    activity = output["orientation"]
                    self.assertGreater(activity[side]-activity[1-side], .15)
        raw = tone(523, amplitude=(.08, .04))
        output = AuditoryNeurons().step(raw)
        working_difference = output["rates"][0, 1]-output["rates"][1, 1]
        self.assertGreater(working_difference, .15)
        self.assertGreater(output["orientation"][0]-output["orientation"][1], .1)
        swapped = AuditoryNeurons().step(raw[::-1])
        for key in ("rates", "orientation"):
            np.testing.assert_allclose(output[key][::-1], swapped[key], atol=1e-13)
        saturated = AuditoryNeurons().step(tone(523, amplitude=(.8, .4)))
        self.assertLess(saturated["rates"][0, 1]-saturated["rates"][1, 1], working_difference/2)

    def test_physical_field_amplitudes_keep_usable_band_tuning(self):
        for band, frequency in enumerate((262, 523, 880)):
            previous = 0.
            for amplitude in (.04, .06, .12, .20):
                with self.subTest(frequency=frequency, amplitude=amplitude):
                    output = AuditoryNeurons().step(tone(frequency, amplitude=(amplitude, amplitude)))
                    row = output["rates"][0]
                    others = [index for index in range(3) if index != band]
                    self.assertGreater(row[band], .05)
                    self.assertGreater(row[band], previous)
                    self.assertTrue(np.all(row[band] > row[others]+.05))
                    self.assertLess(row[band], .9)
                    previous = row[band]

    def test_chunking_preserves_neural_and_axonal_state(self):
        raw = tone(523, delay=3, amplitude=(.7, .45), samples=4096)
        original = raw.copy()
        whole, chunks = AuditoryNeurons(), AuditoryNeurons()
        expected = whole.step(raw)
        for index in range(0, raw.shape[1], 137):
            actual = chunks.step(raw[:, index:index+137])
        for key in expected:
            np.testing.assert_array_equal(expected[key], actual[key])
        one, two = whole.snapshot(), chunks.snapshot()
        for key in ("voltage", "adaptation", "axon_history"):
            np.testing.assert_array_equal(one[key], two[key])
        self.assertEqual(one["sample_count"], two["sample_count"])
        self.assertEqual(one["axon_cursor"], two["axon_cursor"])
        np.testing.assert_array_equal(original, raw)

    def test_silence_decay_adaptation_and_independent_snapshots(self):
        brain = AuditoryNeurons()
        silent = brain.step(np.zeros((2, 512)))
        for value in silent.values():
            np.testing.assert_array_equal(value, np.zeros_like(value))
        active = brain.step(tone(262))
        before = brain.snapshot()
        voltage_before_edit = before["voltage"].copy()
        self.assertGreater(before["adaptation"].max(), .4)
        active["rates"][:] = 0
        before["voltage"][:] = 100
        before["weights"][:] = 0
        before["axon_history"][:] = 0
        current = brain.snapshot()
        np.testing.assert_array_equal(current["voltage"], voltage_before_edit)
        self.assertTrue(np.all(current["weights"] > 0))
        self.assertGreater(current["axon_history"].max(), .4)
        after = brain.step(np.zeros((2, 16000)))
        for value in after.values():
            self.assertLess(float(value.max()), 1e-6)
        snapshot = brain.snapshot()
        self.assertLess(snapshot["adaptation"][snapshot['adaptation_gain'] > 0].max(), .002)
        brain.reset()
        self.assertEqual(brain.sample_count, 0)
        np.testing.assert_array_equal(brain.snapshot()["voltage"], AuditoryNeurons().snapshot()['voltage'])

    def test_invalid_input_is_transactional(self):
        brain = AuditoryNeurons()
        brain.step(tone(880, samples=512))
        before = brain.snapshot()
        invalid = [np.zeros((1, 512)), np.zeros((2, 0)), np.zeros((2, 4), dtype=bool),
                   np.full((2, 2), np.nan), np.full((2, 2), np.inf), np.full((2, 2), 1.01),
                   np.full((2, 2), -1.01), np.full((2, 2), np.iinfo(np.int64).min),
                   np.zeros((2, 4), dtype=complex), [[0., 0.], [0., 0.]]]
        for raw in invalid:
            with self.assertRaises(ValueError):
                brain.step(raw)
        after = brain.snapshot()
        for key in ("voltage", "adaptation", "axon_history"):
            np.testing.assert_array_equal(before[key], after[key])
        self.assertEqual(before["sample_count"], after["sample_count"])
        for rate in (True, 16000., 8000, np.nan):
            with self.assertRaises(ValueError):
                AuditoryNeurons(rate)

    def test_theoretical_voltage_bounds_with_saturated_receptors(self):
        brain = AuditoryNeurons()
        graph = brain.snapshot()
        n = brain.n_neurons
        excitatory = graph["signs"][graph["src"]] > 0
        incoming_e = np.bincount(graph["dst"][excitatory], weights=graph["weights"][excitatory], minlength=n)
        incoming_i = np.bincount(graph["dst"][~excitatory], weights=graph["weights"][~excitatory], minlength=n)
        receptor_max = np.zeros(n)
        receptor_max[graph["receptor_ids"]] = graph["receptor_gain"]
        # From bounded presynaptic rates/adaptation and a convex leaky update;
        # voltage is not assumed to be a rate, nor arbitrarily capped at four.
        lower = np.minimum(0, graph["bias"]-incoming_i-graph["adaptation_gain"]+graph['external_low'])
        upper = np.maximum(0, graph["bias"]+incoming_e+graph['external_high'])
        raw = np.stack([np.ones(512), -np.ones(512)])
        brain.step(raw)
        state = brain.snapshot()
        self.assertGreater(state["voltage"].max(), 4)
        self.assertTrue(np.all(state["voltage"] >= lower-1e-12))
        self.assertTrue(np.all(state["voltage"] <= upper+1e-12))
        self.assertTrue(((state["activity"] >= 0) & (state["activity"] <= 1)).all())
        self.assertTrue(((state["adaptation"] >= 0) & (state["adaptation"] <= 1)).all())
        self.assertAlmostEqual(float(lower.min()), -6.55)
        self.assertAlmostEqual(float(upper.max()), 23.85)

    def test_fixed_transmitters_bounds_and_512_sample_performance(self):
        brain = AuditoryNeurons()
        initial = brain.snapshot()
        self.assertTrue(np.isin(initial["signs"], [-1, 1]).all())
        self.assertTrue((initial["weights"] >= 0).all())
        self.assertEqual(brain.n_neurons, len(initial['names']))
        self.assertEqual(len(brain.weights), len(initial['src']))
        raw = tone(523, delay=3, samples=512*36)
        elapsed = []
        for index in range(36):
            start = time.perf_counter()
            output = brain.step(raw[:, index*512:(index+1)*512])
            duration = time.perf_counter()-start
            if index >= 4:
                elapsed.append(duration*1000)
            for value in output.values():
                self.assertTrue(np.isfinite(value).all())
                self.assertTrue(((value >= 0) & (value <= 1)).all())
        current = brain.snapshot()
        np.testing.assert_array_equal(initial["weights"], current["weights"])
        self.assertTrue(((current["adaptation"] >= 0) & (current["adaptation"] <= 1)).all())
        print("AUDITORY_BENCHMARK " + json.dumps(dict(block_samples=512, audio_ms=32,
              measured_blocks=len(elapsed), mean_ms=float(np.mean(elapsed)),
              median_ms=float(np.median(elapsed)), p95_ms=float(np.quantile(elapsed, .95)))))

    def test_the_outer_ear_shape_moves_the_two_front_back_cells(self):
        """A sound whose top band stands out reads as in front of the head and
        one whose top band is held back reads as behind it, at every level; the
        two ears agree, because the shape of the sound is the same at both."""
        t = np.arange(8192)/16000
        def shaped(amplitude, ratio):
            return np.stack([amplitude*np.sin(2*np.pi*262*t)
                             + amplitude*ratio*np.sin(2*np.pi*880*t)]*2)
        for amplitude in (.06, .08, .12, .16):
            with self.subTest(amplitude=amplitude):
                brighter = AuditoryNeurons().step(shaped(amplitude, 1.35))["pinna_rates"]
                duller = AuditoryNeurons().step(shaped(amplitude, .72))["pinna_rates"]
                self.assertGreater(brighter[0, 0], brighter[0, 1]+.02)
                self.assertGreater(duller[0, 1], duller[0, 0]+.05)
                np.testing.assert_allclose(brighter[0], brighter[1], atol=1e-13)
                np.testing.assert_allclose(duller[0], duller[1], atol=1e-13)

    def test_opposed_tones_keep_separate_spatial_populations(self):
        t = np.arange(8192)/16000
        delay = 3/16000
        for amplitude in (.04,.06,.12):
            raw = np.stack([
                amplitude*np.sin(2*np.pi*262*(t+delay))+amplitude*np.sin(2*np.pi*880*t),
                amplitude*np.sin(2*np.pi*262*t)+amplitude*np.sin(2*np.pi*880*(t+delay))])
            first = AuditoryNeurons().step(raw)['spatial_rates']
            swapped = AuditoryNeurons().step(raw[::-1])['spatial_rates']
            self.assertGreater(first[0,0], first[1,0]+.02)
            self.assertGreater(first[1,2], first[0,2]+.02)
            np.testing.assert_allclose(first[::-1], swapped, atol=1e-13)


if __name__ == "__main__":
    unittest.main()
