"""Independent tiny arithmetic and exact-resume checks, before corpus evaluation."""
import tempfile
from pathlib import Path
import numpy as np
from hebb_text import CharacterHebb, Config


def run():
    cfg = Config(trace_decay=.8, growth="diminishing", learning_rate=.25)
    b = CharacterHebb("甲乙丙", cfg)
    b.train("甲乙丙")
    assert np.isclose(b.weights[0, 1], .25)
    assert np.isclose(b.weights[0, 2], .20)
    assert np.isclose(b.weights[1, 2], .25)
    assert b.weights[2, 0] == 0
    assert b.clock == 3 and b.learned_characters == 3
    before = b.weight_hash()
    b.observe("甲", learn=False)
    b.probabilities(recurrent_steps=2)
    assert b.weight_hash() == before and b.clock == 4
    for decay in (0., .8):
        for growth in ("linear", "diminishing", "saturating"):
            x = CharacterHebb("甲乙丙", Config(trace_decay=decay, growth=growth))
            x.train("甲乙丙甲乙")
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "model.npz"
                x.save(path)
                y = CharacterHebb.load(path)
                assert x.clock == y.clock and x.activity == y.activity
                x.train("丙甲乙丙")
                y.train("丙甲乙丙")
                assert np.array_equal(x.weights, y.weights)
                assert x.activity == y.activity and x.clock == y.clock
    # With zero trace, observed A->B does not create B->A or A->C.
    z = CharacterHebb("甲乙丙", Config(trace_decay=0.))
    z.train("甲乙丙")
    assert z.weights[0, 1] > 0 and z.weights[1, 2] > 0
    assert z.weights[1, 0] == 0 and z.weights[0, 2] == 0
    print("PASS: causal direction, trace arithmetic, frozen probing, six exact resume checks")


if __name__ == "__main__":
    run()
