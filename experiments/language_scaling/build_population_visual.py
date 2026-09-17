"""Package recorded population frames; no model or neural dynamics changes."""
from pathlib import Path
import hashlib
import json
import numpy as np

HERE = Path(__file__).resolve().parent
DEST = Path(r'C:\Users\Administrator\.codex\visualizations\2026\09\06\01a074bf-f71c-7a81-97c4-7f09ba6e2a4d')
MODES = [
    ('weighted8_spoken', '原 8 候选＋单字回灌'),
    ('weighted8_silent', '原 8 候选，静默传播'),
    ('binary32', '32 个同步激活'),
    ('binary32_no_self', '32 个同步激活，去自连接'),
    ('binary128_no_self', '128 个同步激活，去自连接'),
    ('binary32_no_self_rest1', '32 个去自连接，接收休息 1 步'),
    ('binary32_no_self_rest3', '32 个去自连接，接收休息 3 步'),
]

def main():
    protocol = json.loads((HERE/'population_probe_v1/frozen_protocol.json').read_text(encoding='utf-8'))
    prompts = protocol['trace_prompts']
    all_chars = set()
    data = {'prompts': prompts, 'modes': [], 'traces': {}}
    for name, label in MODES:
        source = json.loads((HERE/'population_probe_v1'/f'{name}.json').read_text(encoding='utf-8'))
        data['modes'].append({'id':name, 'label':label, 'noSelf':source['spec']['no_self']})
        data['traces'][name] = {}
        for result in source['results']:
            if result['prompt'] not in prompts: continue
            states, frames, known, union = [], [], {}, set()
            for row in result['rows']:
                activity = row['activity']
                chars = ''.join(c for c, a in activity)
                union.update(chars)
                amplitudes = [round(a, 7) for c, a in activity]
                packed = [chars, None if all(a == 1 for a in amplitudes) else amplitudes, row.get('cooldown', [])]
                key = json.dumps(packed, ensure_ascii=False, separators=(',', ':'))
                if key not in known:
                    known[key] = len(states); states.append(packed)
                frames.append(known[key])
            data['traces'][name][result['prompt']] = dict(states=states, frames=frames, union=''.join(sorted(union)), repeat=result['first_exact_repeat'])
            all_chars.update(union)
    chars = ''.join(sorted(all_chars))
    model = HERE/protocol['model']
    metadata = json.loads((model/'metadata.json').read_text(encoding='utf-8'))
    vocabulary = metadata['vocabulary']
    indices = {c:i for i,c in enumerate(vocabulary)}
    picked = [indices[c] for c in chars]
    weights = np.load(model/'weights.npy', mmap_mode='r')
    data['chars'] = chars
    # Display-only rounding; graph weights and original float32 evidence remain intact.
    data['weights'] = [[round(float(x), 2) for x in row] for row in weights[np.ix_(picked, picked)]]
    weights._mmap.close()
    payload = json.dumps(data, ensure_ascii=False, separators=(',', ':')).replace('<', '\\u003c')
    template = (HERE/'population_visual.template.html').read_text(encoding='utf-8')
    assert template.count('__POPULATION_DATA__') == 1
    fragment = template.replace('__POPULATION_DATA__', payload)
    assert len(fragment.encode('utf-8')) < 1_000_000
    assert '<html' not in fragment and '<body' not in fragment
    DEST.mkdir(parents=True, exist_ok=True)
    out = DEST/'population-activation.html'
    out.write_text(fragment, encoding='utf-8')
    manifest = dict(fragment=str(out), bytes=out.stat().st_size, characters=len(chars),
        sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
        conditions=len(MODES), prompts=len(prompts), steps=160,
        note='Groups unordered; semantic categories are manual display-only labels. Display weights rounded to .01 and currents approximate. Full precision evidence controls cycle labels.')
    (HERE/'population_visual_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False))

if __name__ == '__main__': main()
