"""Read-only audit of actual persistent state during short-context generation."""
import hashlib
import json
from pathlib import Path
import numpy as np
from hebb_text import CharacterHebb

HERE = Path(__file__).resolve().parent
OUT = HERE / 'short_context_cycle_audit_v1'


def main():
    if OUT.exists():
        raise SystemExit('Refuse to overwrite a saved audit')
    model_path = HERE / 'short_context_v2/decay_0p25/model.npz'
    saved_path = HERE / 'short_context_v2/decay_0p25/test_and_context.json'
    saved = json.loads(saved_path.read_text(encoding='utf-8'))
    model = CharacterHebb.load(model_path)
    original = (model.weight_hash(), model.activity.copy(), model.clock,
                model.learned_characters, model.update_events)
    records = []
    for index, example in enumerate(saved['continuations']):
        for recurrent_steps in (0, 2):
            model.reset_activity_for_probe()
            for character in example['prompt']:
                model.observe(character, learn=False)
            seen = {}
            rows = []
            emitted = ''
            first_repeat = None
            for step in range(64):
                # Include dictionary order: it can affect float32 accumulation.
                signature = tuple(model.activity.items())
                if signature in seen and first_repeat is None:
                    start = seen[signature]
                    first_repeat = dict(first_step=start, repeated_step=step,
                        period=step-start, emitted_cycle=emitted[start:step])
                seen.setdefault(signature, step)
                raw = model.currents(recurrent_steps=recurrent_steps)
                assert signature == tuple(model.activity.items())
                ranking = np.argsort(-raw, kind='stable')
                winner = int(ranking[0])
                chosen = model.vocabulary[winner]
                activity = [dict(character=model.vocabulary[i], amplitude=a)
                            for i, a in model.activity.items()]
                direct_contributions = [dict(source=model.vocabulary[i],
                    activity=a, edge_weight=float(model.weights[i, winner]),
                    current=float(np.float32(a)*model.weights[i, winner]))
                    for i, a in model.activity.items()]
                row = dict(step=step, previous_text_tail=(example['prompt']+emitted)[-16:],
                    persistent_activity=activity, emitted=chosen,
                    top_candidates=[dict(character=model.vocabulary[int(i)], current=float(raw[i]))
                                    for i in ranking[:5]],
                    positive_candidate_count=int(np.count_nonzero(raw > 0)),
                    currents_left_activity_unchanged=True,
                    direct_contributions_to_winner=direct_contributions)
                model.observe(chosen, learn=False)
                row['next_persistent_activity'] = [dict(character=model.vocabulary[i], amplitude=a)
                                                   for i, a in model.activity.items()]
                rows.append(row)
                emitted += chosen
            key = 'greedy' if recurrent_steps == 0 else 'greedy_two_steps'
            assert emitted[:48] == example[key]
            assert first_repeat is not None
            start = first_repeat['first_step']
            period = first_repeat['period']
            assert all(rows[i]['persistent_activity'] == rows[i+period]['persistent_activity']
                       and rows[i]['emitted'] == rows[i+period]['emitted']
                       for i in range(start, 64-period))
            records.append(dict(example=index+1, prompt=example['prompt'],
                recurrent_steps=recurrent_steps, emitted=emitted,
                matches_saved_first_48=True, first_repeat=first_repeat, rows=rows))
    model.activity, model.clock = original[1], original[2]
    assert (model.weight_hash(), model.learned_characters, model.update_events) == (original[0], original[3], original[4])
    OUT.mkdir()
    result = dict(model_weight_sha256=original[0],
        source_sha256={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in [Path(__file__), HERE/'hebb_text.py', HERE/'run_experiment.py', model_path, saved_path]},
        weights_unchanged=True, records=records)
    (OUT/'activation_trace.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    for record in records:
        print(json.dumps({k:v for k,v in record.items() if k != 'rows'},ensure_ascii=False))
    cycle = records[0]['first_repeat']
    for row in records[0]['rows'][cycle['first_step']:cycle['repeated_step']]:
        print(json.dumps(row,ensure_ascii=False))


if __name__ == '__main__':
    main()
