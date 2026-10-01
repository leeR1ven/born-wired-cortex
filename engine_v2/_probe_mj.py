import mujoco, inspect
print("mujoco", mujoco.__version__)
from mujoco import viewer as V
h = V.launch_passive
print("launch_passive", inspect.signature(h))
src = inspect.getsource(V.Handle.__init__)
print([n for n in dir(V.Handle) if not n.startswith("_")])