import inspect, mujoco.viewer as V
src = inspect.getsource(V)
for i, line in enumerate(src.splitlines(), 1):
    low = line.lower()
    if "key" in low and ("callback" in low or "glfw_key" in low or "def " in low):
        print(i, line.rstrip()[:150])
print("set_texts", inspect.signature(V.Handle.set_texts))
print(inspect.getdoc(V.Handle.set_texts)[:400])