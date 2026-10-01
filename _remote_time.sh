cd /hy-tmp/bwc/engine_v2
export PYTHONIOENCODING=utf-8
export MUJOCO_GL=osmesa
PY=/hy-tmp/bwc/.venv/bin/python
$PY - <<'PY'
import sys, time
from pathlib import Path
ROOT = Path('/hy-tmp/bwc/engine_v2')
for extra in (ROOT, ROOT/'tools'):
    if str(extra) not in sys.path: sys.path.insert(0, str(extra))
import chase_red_ball as C
import tools.chase_task as ct
genome = dict(C.load_champion()['genome'])
chase = dict(route='motor', source='position', flip=False, pivot=8, gain=1.0, turn_gain=0.1, turn_time=0.3)
t0=time.perf_counter(); spec = C.gaze_spec(); t1=time.perf_counter()
print('接线表 %.1f 秒' % (t1-t0))
t0=time.perf_counter(); body, brain, eyes, geom = C.build(genome, chase, seed=0, spec=spec); t1=time.perf_counter()
print('建一只狗 %.1f 秒' % (t1-t0))
t0=time.perf_counter(); got = C.measure(body, brain, eyes, geom, 6., .5, 3.0, seed=0); t1=time.perf_counter()
print('跑一趟 6 秒 %.1f 秒' % (t1-t0))
t0=time.perf_counter()
for _ in range(50):
    eyes.observe_raw()
t1=time.perf_counter(); print('眼睛拍 50 张 %.1f 秒（一张 %.3f 秒）' % (t1-t0, (t1-t0)/50))
PY