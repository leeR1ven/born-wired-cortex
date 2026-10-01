# -*- coding: utf-8 -*-
"""两只眼各自的坐标系 —— 接线和测试共用这一份几何。

位置野表（artifacts/红球位置野_红*.json）记的是「球摆在头坐标系的哪个方位、哪个高低」。
可是一只眼看着球要回答的是「球偏到我画面的哪一边」，那是这只眼自己的角度。同一个球摆在
正前方，左眼看到它偏右 0.18 弧度、右眼看到它偏左 0.18 弧度，两只眼都要往里转 —— 这就是
会聚。头坐标系里根本看不出这件事：球在正前方就是 0 度，两只眼于是谁也不动，看着像斜视。

换算要用到三个数，全都从模型里来：

- 两只眼球的位置（tools/build_arena.py 里 pos='.30 ±.11 .32'）；
- 球的摆放高度原点：扫场时球的高度是「base 高度 + .03」（tools/sweep_red_ball_features.py），
  也就是 0.27 + .03 = 0.30 米；
- 扫场时球离得多远：位置野表 options 里的 distance，默认 0.90 米。

tests/test_eye_geometry.py 会拿模型核这几个数，改了模型却忘了改这里会被测出来。
"""
import mujoco
import numpy as np

EYE_POS = {"left": np.array([.30, .11, .32]), "right": np.array([.30, -.11, .32])}
BALL_Z = .30               # 扫场时球的高度 = base 高度 .27 + .03
BALL_OVER_BASE = .03       # 上面那个「+.03」本身，测试拿它核模型
SWEEP_DISTANCE = .90       # 扫场时球离 base 多远（位置野表 options.distance）


def ball_world(bearing, elevation, distance=SWEEP_DISTANCE, ball_z=BALL_Z):
    """头坐标系里的（方位，高低）+ 距离 -> 球在世界里的位置。"""
    return np.array([distance*np.cos(elevation)*np.cos(bearing),
                     distance*np.cos(elevation)*np.sin(bearing),
                     ball_z + distance*np.sin(elevation)])


def ball_places(bearings, elevations, distance=SWEEP_DISTANCE, ball_z=BALL_Z):
    """一次算一批球的绝对位置，形状 (n, 3)。"""
    bearings = np.asarray(bearings, dtype=float)
    elevations = np.asarray(elevations, dtype=float)
    return np.stack([distance*np.cos(elevations)*np.cos(bearings),
                     distance*np.cos(elevations)*np.sin(bearings),
                     ball_z + distance*np.sin(elevations)], axis=1)


def place_ball(model, data, target, bearing, elevation, distance=SWEEP_DISTANCE,
               ball_z=BALL_Z):
    """把球摆到（方位，高低）上，并结算一次前向动力学。"""
    model.geom_pos[target] = ball_world(bearing, elevation, distance, ball_z)
    mujoco.mj_forward(model, data)


def seen_by(place, side, distance=SWEEP_DISTANCE, ball_z=BALL_Z):
    """球摆在这个位置时，side 那只眼看到它偏在自己坐标系的哪个角度。

    返回（方位，高低），正的方位 = 球在这只眼的左手边，正的高低 = 球在这只眼的上方，
    和位置野表里的记法同向，换的只是原点：从 base 换成这只眼。
    """
    place = np.asarray(place, dtype=float)
    taken = seen_by_many(place.reshape(1, 2), side, distance, ball_z)[0]
    return float(taken[0]), float(taken[1])


def seen_by_many(places, side, distance=SWEEP_DISTANCE, ball_z=BALL_Z):
    """一批球位置一次算完，形状 (n, 2)。"""
    places = np.asarray(places, dtype=float).reshape(-1, 2)
    offset = ball_places(places[:, 0], places[:, 1], distance, ball_z) - EYE_POS[side]
    yaw = np.arctan2(offset[:, 1], offset[:, 0])
    pitch = np.arctan2(offset[:, 2], np.hypot(offset[:, 0], offset[:, 1]))
    return np.stack([yaw, pitch], axis=1)


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
