import sys
from pathlib import Path
import numpy as np
ROOT = Path(r"F:\born-wired-cortex\engine_v2")
for extra in (ROOT, ROOT / "tools"):
    sys.path.insert(0, str(extra))
from born_wired.embodied import EmbodiedController
from tools import taskbank as tb
import json
row = json.loads((ROOT / "artifacts" / "云端" / "题1f_演化_云_冠军.jsonl").read_text(encoding="utf-8").splitlines()[0])
ctx = tb.context(row["genome"], 0)
body = tb.clean_body(ctx["model_path"])
brain = EmbodiedController(body.home_angles, body.lower_limits, body.upper_limits, seed=0,
                           eye_limits=(body.eye_lower_limits, body.eye_upper_limits),
                           **ctx["parameters"])
names = sorted(brain.groups)
print("组数", len(names), " 神经元总数", brain.network.n_neurons)
import re
for key in ("red", "eye", "locomot", "walk", "gait", "rhythm", "turn", "steer", "forward",
            "drive", "association", "initiat", "fatigue", "com", "com_", "muscle", "hip"):
    hits = [n for n in names if key in n.lower()]
    if hits:
        print("\n[%s] %d 个：" % (key, len(hits)))
        for n in hits:
            print("   ", n, len(np.atleast_1d(brain.groups[n])))