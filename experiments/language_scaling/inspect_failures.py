"""Post-hoc inspection only: actual inputs, learned edges, and summed currents."""
from collections import Counter
import json
from pathlib import Path
import unicodedata
import numpy as np
from hebb_text import CharacterHebb

HERE=Path(__file__).resolve().parent
RESULTS=HERE/'results_v1'


def main():
    summary=json.loads((RESULTS/'summary.json').read_text(encoding='utf-8'))
    final=summary['protocol']['train_characters']
    all_rows={}
    stats={}
    for name in summary['conditions']:
        rows=json.loads((RESULTS/name/f'stage_{final}.json').read_text(encoding='utf-8'))['test']['rows']
        all_rows[name]={r['position']:r for r in rows}
        groups={k:[] for k in ('han','punctuation','other')}
        for r in rows:
            ch=r['target']
            group='han' if '\u3400'<=ch<='\u9fff' else 'punctuation' if unicodedata.category(ch).startswith('P') else 'other'
            groups[group].append(r)
        stats[name]=dict(by_target_type={k:dict(n=len(v),correct=sum(r['rank']==1 for r in v)) for k,v in groups.items()},
                        predicted_types=len({r['predicted'] for r in rows}),
                        most_predicted=Counter(r['predicted'] for r in rows).most_common(8))
    adjacent=all_rows['adjacent_linear'];context=all_rows['context_diminishing']
    eligible=[p for p in sorted(adjacent) if adjacent[p]['rank']==1 and context[p]['rank']!=1
              and '\u3400'<=adjacent[p]['target']<='\u9fff']
    evidence=None
    if eligible:
        pos=eligible[0];text=(HERE/'data/test.txt').read_text(encoding='utf-8')
        b=CharacterHebb.load(RESULTS/'context_diminishing/model.npz')
        b.reset_activity_for_probe()
        for ch in text[:pos]:b.observe(ch,learn=False)
        raw=b.currents();winner=int(np.argmax(raw));target=b.ids[text[pos]]
        explanation={}
        for label,idx in [('actual_model_winner',winner),('heldout_target',target)]:
            entries=[dict(source=b.vocabulary[i],activity=a,weight=float(b.weights[i,idx]),
                          contribution=float(np.float32(a)*b.weights[i,idx])) for i,a in b.activity.items()]
            entries.sort(key=lambda r:-r['contribution'])
            explanation[label]=dict(character=b.vocabulary[idx],total_current=float(raw[idx]),largest_contributors=entries[:10])
        evidence=dict(selection='first CJK target correct under adjacent_linear and incorrect under context_diminishing',
                      position=pos,prefix=text[max(0,pos-80):pos],target=text[pos],explanation=explanation,
                      all_active_characters=[dict(character=b.vocabulary[i],activity=a) for i,a in b.activity.items()],
                      weights_were_not_changed=True)
    output=dict(target_type_definition='Han U+3400..U+9FFF; punctuation Unicode category P*',conditions=stats,example=evidence)
    (RESULTS/'failure_inspection.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(output,ensure_ascii=False))


if __name__=='__main__':main()
