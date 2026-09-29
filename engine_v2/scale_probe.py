"""One fixed workload, so several copies of it can be run at once.

    python scale_probe.py 300        # 300 controller steps, prints ms/step

Used to find out whether this machine is short of cores or short of memory
bandwidth: if N copies each take N times as long, the limit is shared (cache,
bandwidth, or hyperthreading) and adding workers buys nothing.
"""
import sys
import time

sys.path.insert(0, ".")
sys.path.insert(0, "tools")

from tools import taskbank as tb

steps = int(sys.argv[1]) if len(sys.argv) > 1 else 300
body = tb.clean_body(tb.ARENA)
ctx = tb.context({}, 0)
brain = tb.brain_for(ctx, body)
obs = body.observe()
env = tb.blank_environment()
for _ in range(20):
    brain.step(obs, environment=env, dt=tb.DT, learn=False)
started = time.perf_counter()
for _ in range(steps):
    brain.step(obs, environment=env, dt=tb.DT, learn=False)
elapsed = time.perf_counter() - started
print("%.4f" % (elapsed / steps * 1000.))