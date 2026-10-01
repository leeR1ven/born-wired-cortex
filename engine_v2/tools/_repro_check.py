import json, sys
from pathlib import Path
ROOT = Path(r"F:\born-wired-cortex\engine_v2")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"tools"))
import numpy as np
from tools import taskbank as tb
from tools import evolve as ev
row = json.loads((ROOT/"artifacts"/"云端"/"题1f_演化_云_冠军.jsonl").read_text(encoding="utf-8").splitlines()[0])
ctx = tb.context(row["genome"], row["seed"])
res = tb.run_seed(seed=ctx["seed"], duration=10.0, model_path=ctx["model_path"],
                  scenario="autonomous", terrain_start="origin", props=ev.open_props(),
                  controller_parameters_for_seed=ctx["parameters"])
m = res["metrics"]
print("run_seed 本体：位移 %.3f 米  路程 %.3f  最低直立分量 %.3f  平均速度 %.3f  最快 %.3f"
      % (m["total_displacement_m"], m["horizontal_path_m"], m["minimum_up_z"],
         m["mean_speed_mps"], m["max_speed_mps"]))
print("检查项：", {k: v for k, v in res.items() if k not in ("metrics","timeline")} if len(str(res)) < 400 else "")