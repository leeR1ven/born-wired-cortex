import sys, numpy as np
sys.path.insert(0, ".")
from tools import evolve_chase as E
rng = np.random.default_rng(7)
a = dict(route="motor", source="position", flip=True, pivot=9, gain=0.8, turn_gain=0.11, turn_time=0.3)
b = dict(route="orient", source="muscle", flip=False, pivot=6, gain=1.7, turn_gain=0.27, turn_time=0.13)
parents = [dict(chase=a), dict(chase=b)]
crossed = 0
bad = 0
for i in range(4000):
    got = E.pick_chase(parents, rng)
    if got.get("route") not in ("motor", "orient"): bad += 1
    if got.get("source") not in ("position", "muscle"): bad += 1
    if not isinstance(got.get("flip"), (bool, np.bool_)): bad += 1
    if not (5 <= int(got.get("pivot", 0)) <= 11): bad += 1
    for k in ("gain", "turn_gain", "turn_time"):
        if not (isinstance(got.get(k), float) and got[k] > 0): bad += 1
    if got["route"] != a["route"] and got["route"] != b["route"]: crossed += 1
print("4000 次抽查，不合格项:", bad)
print("没有爹的接法时（只能整套重摇）:", E.pick_chase([dict(genome={})], rng))
mid = [E.pick_chase(parents, rng) for _ in range(1)]
print("样例:", mid[0])
