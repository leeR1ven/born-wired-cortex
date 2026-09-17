"""A character is a unit; the only learned object is a directed weight matrix.

No language model, parser, n-gram states, rules, answers, or task labels occur
in this module. Text identity is a reversible input/output convention. This is
an explicitly simplified experiment, not the full PetBrain architecture.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Config:
    trace_decay: float = 0.0
    trace_floor: float = 0.02
    learning_rate: float = 0.25
    growth: str = "diminishing"
    ceiling: float = 4.0


class CharacterHebb:
    def __init__(self, vocabulary: str, config: Config = Config()):
        if not vocabulary or len(set(vocabulary)) != len(vocabulary):
            raise ValueError("Vocabulary must contain unique characters")
        if not 0 <= config.trace_decay < 1 or not 0 < config.trace_floor <= 1:
            raise ValueError("Invalid activity-trace parameters")
        if config.learning_rate <= 0 or config.ceiling <= 0:
            raise ValueError("Invalid plasticity parameters")
        if config.growth not in ("linear", "diminishing", "saturating"):
            raise ValueError("Unknown weight growth")
        self.vocabulary = vocabulary
        self.ids = {c: i for i, c in enumerate(vocabulary)}
        self.config = config
        self.weights = np.zeros((len(vocabulary), len(vocabulary)), np.float32)
        self.activity: dict[int, float] = {}
        self.clock = 0
        self.learned_characters = 0
        self.update_events = 0

    def observe(self, character: str, *, learn: bool = True):
        """Learn previous activity -> newly observed character, then advance time.

        The same state/clock continues across paragraphs. Unknown evaluation
        characters advance/decay activity but have no fabricated known identity.
        """
        if len(character) != 1:
            raise ValueError("One Unicode character is one observation")
        target = self.ids.get(character)
        cfg = self.config
        if learn and target is None:
            raise ValueError("Training character not in declared vocabulary")
        if learn and target is not None and self.activity:
            sources = np.fromiter(self.activity, dtype=np.int64)
            activation = np.fromiter(self.activity.values(), dtype=np.float32)
            old = self.weights[sources, target]
            delta = cfg.learning_rate * activation
            if cfg.growth == "diminishing":
                delta /= 1.0 + old
            elif cfg.growth == "saturating":
                delta *= np.maximum(0.0, 1.0 - old / cfg.ceiling)
            self.weights[sources, target] = old + delta
            self.update_events += len(sources)
        self.activity = {i: a * cfg.trace_decay for i, a in self.activity.items()
                         if a * cfg.trace_decay >= cfg.trace_floor}
        if target is not None:
            self.activity[target] = 1.0
        self.clock += 1
        self.learned_characters += int(learn)

    def train(self, text: str):
        for character in text:
            self.observe(character)

    def currents(self, *, recurrent_steps: int = 0, active_limit: int = 32,
                 recurrent_mix: float = 0.25) -> np.ndarray:
        """Sum learned outgoing currents; optional recurrence uses the SAME graph.

        Sparsification is explicitly a computational activity-competition rule.
        It does not create hidden sentence/role neurons or learn during recall.
        """
        n = len(self.vocabulary)
        if not self.activity:
            return np.zeros(n, np.float32)
        sources = np.fromiter(self.activity, dtype=np.int64)
        activation = np.fromiter(self.activity.values(), dtype=np.float32)
        raw = (self.weights[sources] * activation[:, None]).sum(axis=0)
        initial = raw.copy()
        for _ in range(recurrent_steps):
            if not np.any(raw > 0):
                break
            count = min(active_limit, n)
            selected = np.argpartition(raw, -count)[-count:]
            selected = selected[raw[selected] > 0]
            drive = raw[selected] / raw[selected].sum()
            following = (self.weights[selected] * drive[:, None]).sum(axis=0)
            if following.sum() > 0:
                following *= max(float(initial.sum()), 1e-12) / following.sum()
            raw = (1.0 - recurrent_mix) * initial + recurrent_mix * following
        return raw

    def probabilities(self, *, recurrent_steps: int = 0):
        raw = self.currents(recurrent_steps=recurrent_steps).astype(np.float64)
        total = float(raw.sum())
        # A disclosed uniform readout floor assigns finite loss to unseen edges.
        # This is not a learned fallback policy and is not fed into training.
        if total <= 0:
            return np.full(len(raw), 1.0 / len(raw))
        return 0.999 * raw / total + 0.001 / len(raw)

    def reset_activity_for_probe(self):
        """Evaluation-only intervention; does not reset clock or learned weights."""
        self.activity.clear()

    def weight_hash(self):
        digest = hashlib.sha256(self.vocabulary.encode("utf-8"))
        digest.update(self.weights.tobytes())
        return digest.hexdigest()

    def save(self, path: str | Path):
        metadata = dict(vocabulary=self.vocabulary, config=asdict(self.config),
                        clock=self.clock, learned_characters=self.learned_characters,
                        update_events=self.update_events,
                        activity=list(self.activity.items()))
        np.savez_compressed(path, weights=self.weights,
                            metadata=np.array(json.dumps(metadata, ensure_ascii=False)))

    @classmethod
    def load(cls, path: str | Path):
        with np.load(path, allow_pickle=False) as archive:
            meta = json.loads(str(archive["metadata"]))
            obj = cls(meta["vocabulary"], Config(**meta["config"]))
            obj.weights = archive["weights"].copy()
        obj.clock = meta["clock"]
        obj.learned_characters = meta["learned_characters"]
        obj.update_events = meta["update_events"]
        obj.activity = {int(i): float(a) for i, a in meta["activity"]}
        return obj

    def stats(self):
        positive = self.weights[self.weights > 0]
        return dict(characters=len(self.vocabulary), learned_characters=self.learned_characters,
                    clock=self.clock, update_events=self.update_events,
                    connections=int(len(positive)), matrix_bytes=int(self.weights.nbytes),
                    weight_max=float(positive.max()) if len(positive) else 0.0,
                    weight_mean=float(positive.mean()) if len(positive) else 0.0,
                    active_characters=len(self.activity))
