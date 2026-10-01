import inspect, mujoco.viewer as V
src = inspect.getsource(V).splitlines()
for i, line in enumerate(src, 1):
    if "run_physics_thread" in line or "def launch_passive" in line or "def launch(" in line:
        print(i, line.rstrip()[:140])