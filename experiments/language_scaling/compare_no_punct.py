"""Compare the identical Han targets, including a readout-only masking control."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import unicodedata
import numpy as np
from hebb_text import CharacterHebb
from run_experiment import evaluate,prediction_record,summarize

HERE=Path(__file__).resolve().parent
OUT=HERE/'results_no_punct_v1'


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def dump(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')


def masked_evaluation(brain,text,positions):
    activity,clock=brain.activity.copy(),brain.clock
    before=brain.weight_hash();brain.reset_activity_for_probe()
    allowed=np.array([not unicodedata.category(c).startswith('P') for c in brain.vocabulary])
    rows=[]
    try:
        for i,ch in enumerate(text):
            if i in positions:
                probability=brain.probabilities();probability[~allowed]=0.
                probability/=probability.sum()
                rows.append(prediction_record(brain,text,i,probability))
            brain.observe(ch,learn=False)
    finally:brain.activity,brain.clock=activity,clock
    assert brain.weight_hash()==before
    return dict(metrics=summarize(rows),rows=rows,weights_unchanged=True,
                description='Original training and original input; only punctuation output scores set to zero')


def main():
    original_summary=read(HERE/'results_v1/summary.json')
    new_summary=read(OUT/'summary.json');assert new_summary['completed']
    mapping=read(HERE/'data_no_punct_v1/matched_positions.json')
    old_text=(HERE/'data/test.txt').read_text(encoding='utf-8')
    new_text=(HERE/'data_no_punct_v1/test.txt').read_text(encoding='utf-8')
    output=dict(**mapping,conditions={},source_hashes={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (Path(__file__),HERE/'prepare_no_punct.py',HERE/'data_no_punct_v1/manifest.json')})
    old_stage=original_summary['protocol']['train_characters']
    new_stage=new_summary['protocol']['train_characters']
    for name in new_summary['conditions']:
        all_old=read(HERE/'results_v1'/name/f'stage_{old_stage}.json')['test']['rows']
        old_set=set(mapping['original_positions'])
        original=[row for row in all_old if row['position'] in old_set]
        b=CharacterHebb.load(OUT/name/'model.npz')
        no_punct=evaluate(b,new_text,set(mapping['new_positions']))
        for a,c in zip(original,no_punct['rows']):assert a['target']==c['target']
        del b
        b=CharacterHebb.load(HERE/'results_v1'/name/'model.npz')
        masked=masked_evaluation(b,old_text,old_set)
        a_correct=[r['rank']==1 for r in original]
        c_correct=[r['rank']==1 for r in no_punct['rows']]
        summary=dict(n=len(original),original_correct=sum(a_correct),no_punct_correct=sum(c_correct),
                     helped=sum(not a and c for a,c in zip(a_correct,c_correct)),
                     harmed=sum(a and not c for a,c in zip(a_correct,c_correct)),
                     no_punct_predicted_types=len({r['predicted'] for r in no_punct['rows']}),
                     no_punct_most_predicted=Counter(r['predicted'] for r in no_punct['rows']).most_common(8),
                     masked_most_predicted=Counter(r['predicted'] for r in masked['rows']).most_common(8))
        output['conditions'][name]=dict(original=dict(metrics=summarize(original),rows=original),
            no_punct=no_punct,output_mask_only=masked,paired_summary=summary)
        print(json.dumps({'condition':name,'paired_summary':summary,'masked_top1':masked['metrics']['top1']},ensure_ascii=False),flush=True)
        del b
    output['training_characters']=dict(original=old_stage,no_punct=new_stage)
    dump(OUT/'matched_comparison.json',output)


if __name__=='__main__':main()
