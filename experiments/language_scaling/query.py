"""Inspect continuations and the strongest actual outgoing character synapses."""
import argparse
import json
from hebb_text import CharacterHebb
from run_experiment import generate

p=argparse.ArgumentParser()
p.add_argument("model")
p.add_argument("prompt")
p.add_argument("--length",type=int,default=64)
p.add_argument("--steps",type=int,default=0)
a=p.parse_args()
b=CharacterHebb.load(a.model)
print(a.prompt+generate(b,a.prompt,a.length,a.steps))
for ch in dict.fromkeys(a.prompt[-8:]):
    if ch in b.ids:
        row=b.weights[b.ids[ch]]
        indices=row.argsort()[-8:][::-1]
        print(json.dumps({"source":ch,"connections":[{"target":b.vocabulary[int(i)],"weight":float(row[i])} for i in indices if row[i]>0]},ensure_ascii=False))
