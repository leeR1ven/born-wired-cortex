import inspect, mujoco.viewer as V
src = inspect.getsource(V).splitlines()
for i, line in enumerate(src, 1):
    if "GLFW" in line or "glfw.KEY" in line or ("self._key" in line) or "mjKEY" in line:
        print(i, line.rstrip()[:160])