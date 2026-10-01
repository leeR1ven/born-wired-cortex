import inspect, mujoco.viewer as V
src = inspect.getsource(V).splitlines()
for i in range(100, 215):
    print(i+1, src[i].rstrip()[:150])