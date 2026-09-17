"""Read-only activity/connection audit on familiar and unfamiliar prompts."""
from pathlib import Path
import json
import numpy as np
from run_modern_scale import load,read,dump,OUT,DATA,weight_hash
from persistent_candidates import PersistentRecall,RecallConfig

def main():
    summary=read(OUT/'summary.json');graph=load(OUT/summary['final_model'])
    meta=read(OUT/summary['final_model']/'metadata.json');assert weight_hash(graph)==meta['weight_sha256']
    cfg=RecallConfig(candidate_count=8,candidate_budget=.25);results=[]
    for prompt in ['你好','为什么','地球','请解释一下咕噜帕索是什么意思']:
        recall=PersistentRecall(graph,cfg);recall.warm(prompt);seen={};first_repeat=None;rows=[];emitted=''
        for step in range(160):
            state=tuple((int(i),a.hex()) for i,a in recall.activity.items())
            if state in seen and first_repeat is None:first_repeat=dict(first_step=seen[state],repeated_at=step,period=step-seen[state])
            seen.setdefault(state,step)
            raw=recall.currents();order=np.argsort(-raw,kind='stable');winner=int(order[0]);emitted+=graph.vocabulary[winner]
            if step<8 or step>=152:
                sources=np.array(list(recall.activity),np.int64);activity=np.array(list(recall.activity.values()),np.float32)
                edges=graph.weights[sources,winner];drive=edges*activity
                contributions=[dict(character=graph.vocabulary[int(sources[i])],activity=float(activity[i]),weight=float(edges[i]),current=float(drive[i])) for i in np.argsort(-drive,kind='stable')]
                zero_self=graph.weights[sources].copy();zero_self[np.arange(len(sources)),sources]=0
                counter=(zero_self*activity[:,None]).sum(axis=0)
                counter_order=np.argsort(-counter,kind='stable')[:8]
                rows.append(dict(step=step,winner=graph.vocabulary[winner],active=recall.activity_summary(),
                    positive_candidates=int(np.count_nonzero(raw>0)),winner_current_share=float(raw[winner]/raw.sum()) if raw.sum()>0 else None,
                    strongest=[dict(character=graph.vocabulary[i],current=float(raw[i])) for i in order[:8]],
                    winner_contributors=contributions,
                    one_step_self_edges_removed=[dict(character=graph.vocabulary[i],current=float(counter[i])) for i in counter_order]))
            recall.advance(graph.vocabulary[winner],raw)
        results.append(dict(prompt=prompt,emitted=emitted,first_exact_output_state_repeat=first_repeat,rows=rows))
    assert weight_hash(graph)==meta['weight_sha256']
    result=dict(completed=True,model_weight_sha256=meta['weight_sha256'],learning_disabled=True,
        state_definition='Ordered activity IDs and exact binary64 amplitudes, excluding monotonic clock because clock is not read by currents/advance; weights fixed',
        counterfactual_scope='Self edges are removed only for a separate one-step current calculation; full actual trajectory and saved weights remain unchanged',results=results)
    dump(OUT/'cycle_audit.json',result)
    graph.weights._mmap.close()
    print(json.dumps([dict(prompt=r['prompt'],repeat=r['first_exact_output_state_repeat'],tail=r['emitted'][-40:],positive_candidates_last=r['rows'][-1]['positive_candidates']) for r in results],ensure_ascii=False),flush=True)

if __name__=='__main__':main()
