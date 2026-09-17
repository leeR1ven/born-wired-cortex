"""Experimental persistent candidate activity on an unchanged character graph.

Only recall dynamics change here. Learning uses CharacterHebb's continuous real
text stream. There is one activity dictionary and one clock; no phrase memory,
answer-specific links, sequence search, sampling, or repetition ban is present.
"""
from dataclasses import dataclass
import numpy as np
from hebb_text import CharacterHebb


@dataclass(frozen=True)
class RecallConfig:
    candidate_count: int = 0
    candidate_budget: float = 1.0
    decay: float = .25
    floor: float = .02
    active_limit: int = 64


class PersistentRecall:
    def __init__(self, graph: CharacterHebb, config: RecallConfig):
        if config.candidate_count < 0 or config.candidate_budget < 0:
            raise ValueError('Invalid candidate budget/count')
        if not 0 <= config.decay < 1 or not 0 < config.floor <= 1:
            raise ValueError('Invalid activity decay/floor')
        if config.active_limit < max(1, config.candidate_count):
            raise ValueError('Activity capacity is smaller than candidate count')
        self.graph = graph
        self.config = config
        self.activity = {}
        self.clock = 0

    def reset_probe(self):
        self.activity = {}
        self.clock = 0

    def currents(self):
        if not self.activity:
            return np.zeros(len(self.graph.vocabulary), dtype=np.float32)
        ids = np.fromiter(self.activity, dtype=np.int64)
        amplitudes = np.fromiter(self.activity.values(), dtype=np.float32)
        return (self.graph.weights[ids] * amplitudes[:, None]).sum(axis=0)

    def probabilities(self, raw):
        values = raw.astype(np.float64)
        total = float(values.sum())
        if total <= 0:
            return np.full(len(values), 1. / len(values))
        return .999 * values / total + .001 / len(values)

    def advance(self, observed_character, raw=None):
        """Commit candidates from OLD state, then input the actual/new character.

        `raw` must be computed before the target is presented. Candidate current
        is L1-normalized within top K to a fixed total budget, added to decayed
        activity, and clipped at one. Actual sensory/self-output input is then
        clamped to one. None means silent association: no chosen word is fed back.
        """
        cfg = self.config
        if cfg.candidate_count and raw is None:
            raw = self.currents()
        updated = {i: a * cfg.decay for i, a in self.activity.items()
                   if a * cfg.decay >= cfg.floor}
        injected = []
        if cfg.candidate_count and np.any(raw > 0):
            candidates = np.argsort(-raw, kind='stable')[:cfg.candidate_count]
            candidates = candidates[raw[candidates] > 0]
            total = float(raw[candidates].astype(np.float64).sum())
            for index in candidates:
                i = int(index)
                amount = cfg.candidate_budget * float(raw[i]) / total
                updated[i] = min(1., updated.get(i, 0.) + amount)
                injected.append(dict(character=self.graph.vocabulary[i], amount=amount))
        if observed_character is not None:
            if len(observed_character) != 1:
                raise ValueError('One character or None expected')
            i = self.graph.ids.get(observed_character)
            if i is not None:
                updated[i] = 1.
        updated = {i: a for i, a in updated.items() if a >= cfg.floor}
        if len(updated) > cfg.active_limit:
            retained = set(sorted(updated, key=lambda i: (-updated[i], i))[:cfg.active_limit])
            updated = {i: a for i, a in updated.items() if i in retained}
        self.activity = updated
        self.clock += 1
        return injected

    def warm(self, prefix):
        for character in prefix:
            self.advance(character)

    def activity_summary(self):
        ranked = sorted(self.activity, key=lambda i: (-self.activity[i], i))
        values = np.array(list(self.activity.values()), dtype=np.float64)
        total = float(values.sum())
        squared = float(np.square(values).sum())
        return dict(count=len(values), total=total,
            maximum_share=float(values.max()/total) if total else 0.,
            effective_count=total*total/squared if squared else 0.,
            top=[dict(character=self.graph.vocabulary[i], amplitude=self.activity[i]) for i in ranked[:16]],
            all=[dict(character=self.graph.vocabulary[i], amplitude=a) for i, a in self.activity.items()])
