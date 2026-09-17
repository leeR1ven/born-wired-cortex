"""Focused invariants: old-controller equivalence, causal persistence and budget."""
import numpy as np
from hebb_text import CharacterHebb, Config
from persistent_candidates import PersistentRecall, RecallConfig


def main():
    text = '甲乙丙甲丁乙甲丙丁甲乙丙乙丁甲' * 5
    for growth in ('diminishing', 'saturating'):
        graph = CharacterHebb('甲乙丙丁', Config(trace_decay=.25, growth=growth))
        graph.train(text)
        weight_hash = graph.weight_hash()
        graph.reset_activity_for_probe()
        recall = PersistentRecall(graph, RecallConfig(candidate_count=0))
        for ch in text:
            np.testing.assert_array_equal(graph.currents(), recall.currents())
            graph.observe(ch, learn=False)
            recall.advance(ch)
            assert graph.activity == recall.activity
        assert graph.weight_hash() == weight_hash

    graph = CharacterHebb('ABCDE')
    # Numeric unit fixture; these edges are never used in the trained experiment.
    graph.weights[0] = [0, 8, 4, 2, 1]
    graph.weights[1, 4] = 3
    graph.weights[2, 3] = 5
    for count in (4, 8):
        left = PersistentRecall(graph, RecallConfig(candidate_count=count))
        right = PersistentRecall(graph, RecallConfig(candidate_count=count))
        left.advance('A')
        right.advance('A')
        raw_l = left.currents()
        raw_r = right.currents()
        np.testing.assert_array_equal(raw_l, raw_r)
        # Different actual targets cannot influence already computed candidates.
        injected_l = left.advance('B', raw_l)
        injected_r = right.advance('E', raw_r)
        assert injected_l == injected_r
        assert abs(sum(x['amount'] for x in injected_l)-1.) < 1e-12
        # Unchosen C survives and contributes to the following prediction.
        assert left.activity[2] > 0
        expected_d = np.float32(left.activity[0])*2 + np.float32(left.activity[2])*5
        assert np.isclose(left.currents()[3], expected_d)
        assert max(left.activity.values()) <= 1.
        before = graph.weight_hash()
        for _ in range(100):
            raw = left.currents()
            left.advance(None, raw)
            assert max(left.activity.values(), default=0.) <= 1.
            assert len(left.activity) <= 64
            assert all(np.isfinite(a) for a in left.activity.values())
        assert graph.weight_hash() == before
    # Silent propagation must not inject a unit-strength winner as speech input.
    silent = PersistentRecall(graph, RecallConfig(candidate_count=4))
    silent.advance('A')
    silent.advance(None, silent.currents())
    assert silent.activity[1] < 1.
    print('PASS: baseline exact equivalence; pre-target candidates; conserved budget; unchosen candidate persistence; silent propagation; fixed weights')


if __name__ == '__main__':
    main()
