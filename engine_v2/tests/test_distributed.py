import unittest

import numpy as np

from born_wired.distributed import DistributedFeatureNetwork, bridge_edges
from tools.evaluate_hidden_routes import evaluate_case, measure_responses


class DistributedTests(unittest.TestCase):
    def test_global_id_bridge_has_no_layer_restriction(self):
        src, dst, weights = bridge_edges([4, 8], [13, 14], 17, weight=.02)
        np.testing.assert_array_equal(src, [4, 4, 8, 8])
        np.testing.assert_array_equal(dst, [13, 14, 13, 14])
        np.testing.assert_array_equal(weights, [.02] * 4)
        # Feedback can also target earlier/intermediate cells by global ID.
        src, dst, _ = bridge_edges([13], [0, 4, 13], 17)
        np.testing.assert_array_equal(src, [13, 13])
        np.testing.assert_array_equal(dst, [0, 4])
        self.assertEqual(bridge_edges([], [4], 17)[0].shape, (0,))

    def test_invalid_bridge_inputs(self):
        for source, target in (([1, 1], [2]), ([1], [2, 2]), ([True], [2]),
                               ([17], [2]), ([-1], [2]), ([1.5], [2])):
            with self.assertRaises(ValueError):
                bridge_edges(source, target, 17)
        with self.assertRaises(ValueError):
            bridge_edges([1], [2], 17, np.nan)

    def test_shared_graph_retains_hidden_detail_lost_in_compression(self):
        model = DistributedFeatureNetwork(seed=5)
        response = measure_responses(model)
        self.assertEqual(model.n_neurons, 17)
        self.assertEqual(len(model.synapses.weights), 62)
        np.testing.assert_allclose(response[0]["compressed"], response[1]["compressed"], atol=1e-10)
        self.assertGreater(np.linalg.norm(np.array(response[0]["hidden"]) - response[1]["hidden"]), 1)
        syn = model.synapses
        hidden_motor = np.isin(syn.src, model.groups["hidden"]) & np.isin(syn.dst, model.groups["motor"])
        self.assertEqual(np.count_nonzero(hidden_motor), 16)
        self.assertTrue(np.all(syn.signs[model.groups["inhibitory"]] == -1))
        with self.assertRaises(ValueError):
            model.step([1, 0, np.nan, 0])

    def test_local_pairing_needs_hidden_routes_and_learning(self):
        hidden = evaluate_case(7, True, True)
        compressed = evaluate_case(7, False, True)
        no_learning = evaluate_case(7, True, False)
        reversed_pairing = evaluate_case(7, True, True, motor_mapping=(1, 0))
        self.assertGreater(hidden["motor_contrast"], .5)
        self.assertLess(abs(compressed["motor_contrast"]), .01)
        self.assertEqual(no_learning["max_weight_change"], 0)
        self.assertEqual(no_learning["motor_contrast"], 0)
        self.assertLess(reversed_pairing["motor_contrast"], -.5)
        for case in (hidden, compressed, no_learning):
            self.assertEqual(case["motor_teaching_input_at_evaluation"], [0, 0])
            response = np.array([item["motor"] for item in case["after"]])
            self.assertTrue(np.all((response >= 0) & (response <= 1)))


if __name__ == "__main__":
    unittest.main()
