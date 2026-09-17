"""Isolated population propagation on frozen learned weights.

Binary mode is a NEW recall rule, not a claim about the old core: each positive
top-K candidate fires at amplitude one on the next tick. There is no spoken
winner feedback, residual activity, or candidate-budget/floor starvation.
Optional local refractory counters prevent a unit from receiving a new spike
for D ticks after firing; its already-present spike still transmits normally.
No unit is selected using words, correct answers, randomness, or novelty scores.
"""
import numpy as np
from persistent_candidates import PersistentRecall

def population_current(graph,activity,no_self=False):
    if not activity:return np.zeros(len(graph.vocabulary),np.float32)
    ids=np.fromiter(activity,np.int64)
    a=np.fromiter(activity.values(),np.float32)
    rows=graph.weights[ids]
    if no_self:
        rows=rows.copy();rows[np.arange(len(ids)),ids]=0
    return (rows*a[:,None]).sum(axis=0)

class MaskedRecall(PersistentRecall):
    def __init__(self,graph,config,no_self=False):
        super().__init__(graph,config);self.no_self=no_self
    def currents(self):return population_current(self.graph,self.activity,self.no_self)

class BinaryPopulation:
    def __init__(self,graph,k,no_self=False,refractory=0):
        if not 1<=k<=len(graph.vocabulary) or refractory<0:raise ValueError('Invalid population parameters')
        self.graph=graph;self.k=k;self.no_self=no_self;self.refractory=refractory
        self.activity={};self.cooldown=np.zeros(len(graph.vocabulary),np.int32);self.clock=0
    def warm(self,prompt):
        # Use only the original sensory trace for the initial state. No hidden
        # recurrent warm-up during the prompt; same seed for all binary modes.
        activity={}
        for c in prompt:
            activity={i:a*.25 for i,a in activity.items() if a*.25>=.02}
            i=self.graph.ids.get(c)
            if i is not None:activity[i]=1.
            self.clock+=1
        self.activity=dict(sorted(activity.items()))
        self.cooldown.fill(0)
        if self.refractory:
            self.cooldown[list(self.activity)]=self.refractory
    def currents(self):return population_current(self.graph,self.activity,self.no_self)
    def advance(self,raw):
        drive=raw.copy();drive[self.cooldown>0]=0
        chosen=np.argsort(-drive,kind='stable')[:self.k]
        chosen=chosen[drive[chosen]>0]
        # Canonical source order: a population has no sentence/word ordering.
        self.activity={int(i):1. for i in sorted(chosen)}
        self.cooldown=np.maximum(self.cooldown-1,0)
        if self.refractory:self.cooldown[chosen]=self.refractory
        self.clock+=1
        return chosen

def state_key(engine):
    active=tuple((i,float(a).hex()) for i,a in engine.activity.items())
    if isinstance(engine,BinaryPopulation):
        ids=np.flatnonzero(engine.cooldown)
        return active,tuple((int(i),int(engine.cooldown[i])) for i in ids)
    return active,()

def check():
    from hebb_text import CharacterHebb
    # Two simultaneous units can return through other units with zero diagonal.
    g=CharacterHebb('甲乙丙丁');g.weights[0,2:]=1;g.weights[1,2:]=1
    g.weights[2,:2]=1;g.weights[3,:2]=1
    e=BinaryPopulation(g,2,no_self=True);e.activity={0:1.,1:1.}
    e.advance(e.currents());assert set(e.activity)=={2,3}
    e.advance(e.currents());assert set(e.activity)=={0,1}
    # An incoming refractory gate must not erase already-fired outgoing spikes.
    e=BinaryPopulation(g,2,no_self=True,refractory=1);e.activity={0:1.,1:1.};e.cooldown[:2]=1
    raw=e.currents();assert raw[2]>0 and raw[3]>0
    e.advance(raw);assert set(e.activity)=={2,3}
    assert e.cooldown.tolist()==[0,0,1,1]
    assert not np.any(np.diag(g.weights))
    print('POPULATION INTEGRITY CHECK PASS')

if __name__=='__main__':check()
