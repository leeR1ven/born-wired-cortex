"""Frozen data-volume comparison for the simplified character-Hebb model."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from hebb_text import CharacterHebb, Config

HERE = Path(__file__).resolve().parent
CONDITIONS = {
    "adjacent_linear": Config(trace_decay=0., growth="linear", learning_rate=1.),
    "adjacent_diminishing": Config(trace_decay=0., growth="diminishing"),
    "context_diminishing": Config(trace_decay=.8, growth="diminishing"),
    "context_saturating": Config(trace_decay=.8, growth="saturating", ceiling=4.),
    "long_context_diminishing": Config(trace_decay=.98, trace_floor=.01, growth="diminishing"),
}


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def selected_positions(text, count, seed):
    candidates = [i for i in range(64, len(text)) if not text[i].isspace()]
    if not candidates:
        raise ValueError("Evaluation text too short")
    rng = np.random.default_rng(seed)
    return set(map(int, rng.choice(candidates, min(count, len(candidates)), replace=False)))


def summarize(rows):
    known = [r for r in rows if r["known"]]
    return dict(n=len(rows), known_n=len(known), oov_n=len(rows)-len(known),
                top1=sum(r["rank"] == 1 for r in rows)/len(rows),
                top5=sum(0 < r["rank"] <= 5 for r in rows)/len(rows),
                mean_nll_known=float(np.mean([-math.log(r["probability"]) for r in known])) if known else None,
                bpc_known=float(np.mean([-math.log2(r["probability"]) for r in known])) if known else None)


def prediction_record(b, text, position, probabilities):
    target = text[position]
    identity = b.ids.get(target)
    ranking = np.argsort(-probabilities, kind="stable")
    rank = int(np.flatnonzero(ranking == identity)[0])+1 if identity is not None else 0
    return dict(position=position, prefix=text[max(0,position-64):position],
                target=target, known=identity is not None, rank=rank,
                probability=float(probabilities[identity]) if identity is not None else None,
                predicted=b.vocabulary[int(ranking[0])],
                top5="".join(b.vocabulary[int(i)] for i in ranking[:5]))


def evaluate(b, text, positions, *, recurrent_steps=0):
    before_hash = b.weight_hash()
    activity, clock, learned, updates = b.activity.copy(), b.clock, b.learned_characters, b.update_events
    rows=[]
    try:
        b.reset_activity_for_probe()
        for i, character in enumerate(text):
            if i in positions:
                rows.append(prediction_record(b,text,i,b.probabilities(recurrent_steps=recurrent_steps)))
            b.observe(character, learn=False)
    finally:
        b.activity, b.clock = activity, clock
    assert b.weight_hash() == before_hash
    assert (b.learned_characters,b.update_events)==(learned,updates)
    return dict(metrics=summarize(rows), rows=rows, weights_unchanged=True,
                training_state_restored=True, weight_sha256=before_hash)


def context_interventions(b, text, positions):
    """Matched prefixes; never alter the learned graph or target character."""
    original = (b.activity.copy(), b.clock)
    result={}
    try:
        for condition in ("full", "last_only", "shuffled_earlier"):
            rows=[]
            for pos in sorted(positions):
                prefix=text[max(0,pos-64):pos]
                if condition=="last_only":
                    prefix=prefix[-1:]
                elif condition=="shuffled_earlier":
                    rng=np.random.default_rng(pos+91507)
                    chars=list(prefix[:-1]);rng.shuffle(chars)
                    prefix="".join(chars)+prefix[-1:]
                b.reset_activity_for_probe()
                for ch in prefix:b.observe(ch,learn=False)
                record=prediction_record(b,text,pos,b.probabilities())
                record["presented_prefix"]=prefix
                rows.append(record)
            result[condition]=dict(metrics=summarize(rows),rows=rows)
    finally:
        b.activity,b.clock=original
    return result


def generate(b, prompt, length=48, recurrent_steps=0):
    state=(b.activity.copy(),b.clock)
    try:
        b.reset_activity_for_probe()
        for ch in prompt:b.observe(ch,learn=False)
        result=[]
        for _ in range(length):
            # Fixed identity readout; chosen characters are ordinary next inputs.
            p=b.probabilities(recurrent_steps=recurrent_steps)
            ch=b.vocabulary[int(np.argmax(p))]
            result.append(ch);b.observe(ch,learn=False)
        return "".join(result)
    finally:
        b.activity,b.clock=state


def unigram_rows(b, train, text, positions):
    counts=Counter(train)
    p=np.array([counts[ch]+1 for ch in b.vocabulary],float)
    p/=p.sum()
    rows=[prediction_record(b,text,pos,p) for pos in sorted(positions)]
    return dict(metrics=summarize(rows),rows=rows)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--data",type=Path,default=HERE/"data")
    parser.add_argument("--out",type=Path,default=HERE/"results_v1")
    parser.add_argument("--max-chars",type=int,default=400000)
    parser.add_argument("--eval-count",type=int,default=1024)
    parser.add_argument("--conditions",nargs="*",default=list(CONDITIONS))
    args=parser.parse_args()
    args.data=args.data.resolve();args.out=args.out.resolve()
    if args.max_chars < 1 or args.eval_count < 1:
        raise SystemExit("--max-chars and --eval-count must be positive")
    if args.out.exists():raise SystemExit("Use a fresh output directory; evidence is not overwritten")
    args.out.mkdir(parents=True)
    train=(args.data/"train.txt").read_text(encoding="utf-8")[:args.max_chars]
    validation=(args.data/"validation.txt").read_text(encoding="utf-8")
    test=(args.data/"test.txt").read_text(encoding="utf-8")
    vocabulary="".join(sorted(set(train)))
    stages=sorted(set([min(k,len(train)) for k in (1000,10000,100000,len(train))]))
    val_positions=selected_positions(validation,args.eval_count,19301)
    test_positions=selected_positions(test,args.eval_count,19302)
    config=dict(version=1,architecture="character-directed-Hebb-only", conditions={k:vars(CONDITIONS[k]) for k in args.conditions},
                stages=stages,characters=len(vocabulary),train_characters=len(train),
                validation_characters=len(validation),test_characters=len(test),
                eval_count=args.eval_count,validation_positions=sorted(val_positions),test_positions=sorted(test_positions),
                vocabulary_preallocated_from_training_only=True,
                sources={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in [HERE/"hebb_text.py",Path(__file__),HERE/"check_core.py",
                                   args.data/"train.txt",args.data/"validation.txt",args.data/"test.txt"]})
    dump(args.out/"frozen_protocol.json",config)
    all_results={}; started=time.perf_counter()
    for name in args.conditions:
        model_start=time.perf_counter()
        b=CharacterHebb(vocabulary,CONDITIONS[name])
        directory=args.out/name;directory.mkdir()
        stages_result=[];previous=0
        birth=evaluate(b,test,test_positions)
        dump(directory/"birth_test.json",birth)
        for end in stages:
            start=time.perf_counter();b.train(train[previous:end]);elapsed=time.perf_counter()-start
            assert b.clock==end and b.learned_characters==end
            val=evaluate(b,validation,val_positions)
            tested=evaluate(b,test,test_positions)
            base=unigram_rows(b,train[:end],test,test_positions)
            result=dict(stage=end,increment=end-previous,training_seconds=elapsed,
                        stats=b.stats(),validation=val,test=tested,unigram=base)
            dump(directory/f"stage_{end}.json",result)
            stages_result.append({k:result[k] for k in ("stage","increment","training_seconds","stats")}|{
                "validation":val["metrics"],"test":tested["metrics"],"unigram":base["metrics"]})
            previous=end
            print(json.dumps(dict(condition=name,stage=end,train_s=round(elapsed,2),test=tested["metrics"]),ensure_ascii=False),flush=True)
        prefix_positions=set(sorted(test_positions)[:min(128,len(test_positions))])
        interventions=context_interventions(b,test,prefix_positions)
        dump(directory/"context_interventions.json",interventions)
        recurrent=evaluate(b,test,prefix_positions,recurrent_steps=2)
        matched=evaluate(b,test,prefix_positions,recurrent_steps=0)
        dump(directory/"recurrence.json",dict(no_recurrence=matched,two_steps=recurrent,
                                               active_limit=32,recurrent_mix=.25))
        prompts=[test[max(0,pos-32):pos] for pos in sorted(test_positions)[::max(1,len(test_positions)//4)][:4]]
        examples=[dict(prompt=p,greedy=generate(b,p),greedy_two_steps=generate(b,p,recurrent_steps=2)) for p in prompts]
        dump(directory/"continuations.json",examples)
        saved_hash=b.weight_hash();saved_state=(b.clock,b.activity.copy())
        b.save(directory/"model.npz")
        restored=CharacterHebb.load(directory/"model.npz")
        assert restored.weight_hash()==saved_hash and (restored.clock,restored.activity)==saved_state
        # Probe exact continued learning after loading, independent of the main model.
        b.train(train[:128]);restored.train(train[:128])
        assert np.array_equal(b.weights,restored.weights) and b.activity==restored.activity
        all_results[name]=dict(stages=stages_result,total_seconds=time.perf_counter()-model_start,
                               final_checkpoint_weight_sha256=saved_hash,resume_128_exact=True,
                               recurrence={"no_recurrence":matched["metrics"],"two_steps":recurrent["metrics"]})
        dump(args.out/"summary.json",dict(protocol=config,conditions=all_results,elapsed_seconds=time.perf_counter()-started,completed=False))
        del b,restored
    dump(args.out/"summary.json",dict(protocol=config,conditions=all_results,elapsed_seconds=time.perf_counter()-started,completed=True))
    print("COMPLETE",str(args.out),flush=True)


if __name__=="__main__":main()
