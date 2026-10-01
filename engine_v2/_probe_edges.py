import sys
from pathlib import Path
import numpy as np
ROOT = Path(r"F:\born-wired-cortex\engine_v2")
for extra in (ROOT, ROOT/"tools"):
    if str(extra) not in sys.path: sys.path.insert(0, str(extra))
import chase_red_ball as C
genome = dict(C.load_champion()["genome"])
for tg in (0.05, 0.2):
    chase = {"turn_gain": tg, "gain": 1., "source": "position", "flip": False}
    body, brain, eyes, geom = C.build(genome, chase, seed=0, spec=C.gaze_spec())
    syn = brain.synapses
    motor = brain.groups["motor"]; tl = brain.groups["chase_turn_left"]; tr = brain.groups["chase_turn_right"]
    mset = set(int(v) for v in motor)
    sel = np.isin(syn.dst, list(mset)) & np.isin(syn.src, [int(tl[0]), int(tr[0])])
    print("turn_gain=%.2f  span[:4]=%s motor_gain=%s" % (tg, np.round(brain.span[:4], 3), brain.motor_gain))
    print("   从转向细胞进髋关节的边 %d 条，权重和 %.3f，最大 %.3f" % (sel.sum(), syn.weights[sel].sum() if sel.any() else 0., syn.weights[sel].max() if sel.any() else 0.))
    print("   转向细胞总入边 %d 条，权重和 %.2f" % (np.isin(syn.dst, [int(tl[0])]).sum(), syn.weights[np.isin(syn.dst, [int(tl[0])])].sum()))
    # 髋关节那一排 motor 单元的总入边预算
    hip = brain.groups["motor"].reshape(12, -1)[0]
    inc = np.isin(syn.dst, hip)
    print("   左前髋那一排总入边 %d 条，权重和 %.2f，该排预算 %.3f" % (inc.sum(), syn.weights[inc].sum(), syn.budgets[hip[0]]))
    eyes.close()