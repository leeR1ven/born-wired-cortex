import io
p = r"F:\born-wired-cortex\engine_v2\tools\eye_geometry.py"
s = io.open(p, encoding="utf-8", newline="").read()
add = '''

def place_in_front(model, data, target, bearing, elevation, distance, base="base"):
    """把球摆在「两只眼正中间的正前方」的这个（方位，高低，距离）上。

    原点是两只眼球的中点，所以「距离」就是球离眼睛多远 —— 拿场地原点当原点会让球
    其实离眼睛只有「距离 - 0.30」米，眼球经常转不过去。朝向用 base 当前朝向算，
    所以狗身子就算挪了、转了，球也始终在它眼前，不会跑到旁边或者背后去。

    方位记法和位置野表一致：+ 方位 = 狗的左前方，+ 高低 = 上方。
    """
    left = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "eye_left")
    right = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "eye_right")
    frame = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base)
    if min(left, right, frame) < 0:
        raise ValueError("model needs base, eye_left and eye_right bodies")
    rotation = np.asarray(data.xmat[frame], dtype=float).reshape(3, 3)
    origin = (np.asarray(data.xpos[left], dtype=float)
              + np.asarray(data.xpos[right], dtype=float))/2.
    offset = np.array([distance*np.cos(elevation)*np.cos(bearing),
                       distance*np.cos(elevation)*np.sin(bearing),
                       distance*np.sin(elevation)])
    model.geom_pos[target] = origin + rotation @ offset
    mujoco.mj_forward(model, data)
'''
io.open(p, "w", encoding="utf-8", newline="").write(s.rstrip("\n") + "\n" + add)
print("eye_geometry.py 加了 place_in_front")