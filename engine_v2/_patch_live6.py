import io
p = r"F:\born-wired-cortex\engine_v2\tools\eye_live.py"
s = io.open(p, encoding="utf-8", newline="").read()
old = """    base = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_BODY, "base")
    origin = np.asarray(body.data.xpos[base], dtype=float)
    rotation = np.asarray(body.data.xmat[base], dtype=float).reshape(3, 3)"""
new = """    base = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_BODY, "base")
    left = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_BODY, "eye_left")
    right = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_BODY, "eye_right")
    rotation = np.asarray(body.data.xmat[base], dtype=float).reshape(3, 3)
    origin = (np.asarray(body.data.xpos[left], dtype=float)
              + np.asarray(body.data.xpos[right], dtype=float))/2."""
assert s.count(old) == 1
s = s.replace(old, new)
old2 = """    用 base 自己的位置和朝向算，所以狗身子就算挪了、转了，球也始终在它眼前，
    不会跑到旁边或者背后去。方位记法和扫场表一致（+ 方位 = 狗的左前方）。"""
new2 = """    原点是两只眼球的中点，所以「距离」就是球离眼睛多远；朝向用 dog 自己的朝向算，
    狗身子就算挪了、转了，球也始终在它眼前，不会跑到旁边或者背后去
    （+ 方位 = 狗的左前方，+ 高低 = 上方）。"""
assert s.count(old2) == 1
io.open(p, "w", encoding="utf-8", newline="").write(s.replace(old2, new2))
print("改好了")