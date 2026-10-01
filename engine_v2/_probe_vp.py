import inspect, mujoco.viewer as V
src = inspect.getsource(V)
for i, line in enumerate(src.splitlines(), 1):
    if "viewport" in line and ("=" in line or "def " in line):
        print(i, line.rstrip()[:120])