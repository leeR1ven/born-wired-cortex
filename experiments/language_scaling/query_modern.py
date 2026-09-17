"""Query the locally trained modern character graph; frozen weights only."""
import argparse,json
from pathlib import Path
import numpy as np
from run_modern_scale import load,read,OUT
from persistent_candidates import PersistentRecall,RecallConfig
from prepare_modern_corpus import clean

def main():
    p=argparse.ArgumentParser();p.add_argument('prompt');p.add_argument('--model',type=Path)
    p.add_argument('--steps',type=int,default=64);p.add_argument('--silent',action='store_true')
    p.add_argument('--k',type=int,default=8);p.add_argument('--budget',type=float,default=.25)
    a=p.parse_args()
    if not 1<=a.steps<=4096:raise SystemExit('steps must be 1..4096')
    path=a.model or OUT/read(OUT/'summary.json')['final_model'];graph=load(path)
    recall=PersistentRecall(graph,RecallConfig(candidate_count=a.k,candidate_budget=a.budget));recall.warm(clean(a.prompt))
    output=''
    print('模型：',path);print('学习字符：',graph.learned_characters);print('输入：',clean(a.prompt))
    for step in range(a.steps):
        raw=recall.currents();winner=graph.vocabulary[int(np.argmax(raw))] if np.any(raw>0) else None
        if a.silent:
            active=recall.activity_summary()['top'][:12]
            print(json.dumps(dict(step=step,active=active,strongest_candidate=winner),ensure_ascii=False))
        elif winner:output+=winner
        recall.advance(None if a.silent else winner,raw)
    if not a.silent:print('原始续写：',output)
    graph.weights._mmap.close()

if __name__=='__main__':main()
