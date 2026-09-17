# -*- coding: utf-8 -*-
"""身体_go2.py —— 身体换成现成的四足机器人（宇树 Go2），跑在 MuJoCo 上。

★ 为什么换（2026-09-14）

  之前那个身体是自己写的"质点 + 约束"简化物理。为了让四条腿站住，得自己
  补铰链面、腿根钉、翻面锁…… 每补一条，都是自己造的物理在漏。
  现成的机器人模型（MuJoCo Menagerie 的宇树 Go2）把这些全解决了：
    · 真物理：真铰链、真碰撞、真摩擦；
    · 自带 home 关键帧 = 现成的完美站姿，姿势直接抄，不用自己搜；
    · 12 个关节正好当 12 块肌肉，接口干净。

★ 身体和大脑怎么分工

  大脑只做一件事：每帧给出 12 个 0~1 的「发力」。
  这个文件负责把发力变成关节目标角度，再用一层 PD（这就是「肌肉」）出力矩，
  推进物理。动作全部来自神经元放电 —— 这里没有任何"做动作"的代码。

★ 发力怎么对应角度

  每块肌肉 [0,1] 线性对应一段角度范围，**并且刻意让站姿正好落在 0.5**：
  于是"站着"就是 12 个 0.5 的一串，本能写起来最干净（见 站姿发力）。
  0 和 1 两端分别是这条腿能弯到、能伸直到的极限。

★ 注意：MuJoCo 打不开中文路径
  所以模型必须放在英文目录（默认 C:/mujoco_models/unitree_go2）。
  放不进去也没关系：这个文件会自己从网上下载（见 保证模型存在）。
"""
from __future__ import annotations

import os
import pathlib
import urllib.request

import numpy as np

模型目录 = pathlib.Path(os.environ.get("GO2模型目录", r"C:\mujoco_models\unitree_go2"))
模型网址基 = "https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/main/unitree_go2/"

# ---------------- 12 块肌肉 ----------------
# 名字按身体方位起：左前/右前/左后/右后 + 髋/大腿/小腿
肌肉名 = [
    "左前髋", "左前大腿", "左前小腿",
    "右前髋", "右前大腿", "右前小腿",
    "左后髋", "左后大腿", "左后小腿",
    "右后髋", "右后大腿", "右后小腿",
]
关节名 = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
]
执行器名 = ["FL_hip", "FL_thigh", "FL_calf", "FR_hip", "FR_thigh", "FR_calf",
            "RL_hip", "RL_thigh", "RL_calf", "RR_hip", "RR_thigh", "RR_calf"]
肌肉数 = len(肌肉名)

# 每块肌肉对应的角度范围（度）。范围都落在 MuJoCo 给的关节限位以内，
# 并且让 home 站姿正好是 0.5 —— 所以 0.5 就是"天生就会站"。
角度范围 = np.array([
    [-60.0, 60.0], [-38.0, 142.0], [-156.0, -50.2],      # 左前
    [-60.0, 60.0], [-38.0, 142.0], [-156.0, -50.2],      # 右前
    [-60.0, 60.0], [-30.0, 133.2], [-156.0, -50.2],      # 左后
    [-60.0, 60.0], [-30.0, 133.2], [-156.0, -50.2],      # 右后
], dtype=float)

# ---------------- 肌肉参数 ----------------
# 2026-09-14 晚：这两个数是**试出来的最优一档**，不是猜的。
# 扫描过「节奏 30/50 Hz × 刚度 10~150 × 阻尼 0.5~8」（见 诊断_细扫2.py）：
#   50 Hz + 刚度 80 + 阻尼 4 时，抄下来的 26 个动作里 25 个都能自己做完；
#   原来的 30 Hz + 刚度 60 + 阻尼 3 只有 13 个走得出 0.25 米。
# 这两个数属于「肌肉/身体」的参数，和架构无关，可以按需要再调。
刚度 = 80.0        # PD 的 P：偏离目标角 1 弧度出 80 牛米
阻尼 = 4.0         # PD 的 D：抑制摆动，不然会一直晃
最大力矩 = 25.0    # 关节力矩上限（牛米），和真机一个量级
每帧秒 = 1.0 / 50.0   # 大脑一帧对应多少秒（= 20 毫秒，和训练好的那个模型一样）
物理步长 = 0.002      # MuJoCo 自带的时间步

# ---------------- 两个现成的姿势 ----------------
# 站姿：直接抄 Go2 自带的 home 关键帧（0 度 / 51.6 度 / -103.1 度）
站姿角 = np.array([0.0, 51.6, -103.1] * 4)
# 趴姿：大腿往前折、小腿收到最紧，身体落到地上（四处脚还在，肚子贴地）
趴姿角 = np.array([0.0, 95.0, -156.0] * 4)


def 角度转发力(角, 名=None):
    """角度（度）-> 0~1 发力。"""
    角 = np.asarray(角, dtype=float)
    return np.clip((角 - 角度范围[:, 0]) / (角度范围[:, 1] - 角度范围[:, 0]), 0.0, 1.0)


def 发力转角度(发力):
    """0~1 发力 -> 角度（度）。"""
    发力 = np.clip(np.asarray(发力, dtype=float), 0.0, 1.0)
    return 角度范围[:, 0] + 发力 * (角度范围[:, 1] - 角度范围[:, 0])


站姿发力 = 角度转发力(站姿角)
趴姿发力 = 角度转发力(趴姿角)


# ---------------- 模型文件 ----------------
下载清单 = ["scene.xml", "go2.xml"] + [
    "assets/" + n for n in (
        "base_0.obj", "base_1.obj", "base_2.obj", "base_3.obj", "base_4.obj",
        "hip_0.obj", "hip_1.obj", "thigh_0.obj", "thigh_1.obj", "thigh_mirror_0.obj",
        "thigh_mirror_1.obj", "calf_0.obj", "calf_1.obj", "calf_mirror_0.obj",
        "calf_mirror_1.obj", "foot.obj")
]


def 保证模型存在(说=True):
    """模型没下过就自己下。这也是"直接抄现成的"该有的样子。"""
    模型目录.mkdir(parents=True, exist_ok=True)
    缺 = [f for f in 下载清单 if not (模型目录 / f).exists()]
    if not 缺:
        return 模型目录
    if 说:
        print("第一次用：从 MuJoCo Menagerie 下载 Go2 模型（共 %d 个文件）" % len(缺))
    for f in 缺:
        目标 = 模型目录 / f
        目标.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(模型网址基 + f, timeout=60) as r:
            目标.write_bytes(r.read())
        if 说:
            print("  下载", f)
    return 模型目录


class 身体:
    """Go2 身体。大脑每帧调一次 步进(发力)。"""

    def __init__(self, 现场=True, 模型=None, 渲染宽=640, 渲染高=420):
        import mujoco
        self.mujoco = mujoco
        保证模型存在()
        模型文件 = 模型 or (模型目录 / ("scene.xml" if 现场 else "go2.xml"))
        self.模型 = mujoco.MjModel.from_xml_path(str(模型文件))
        self.数据 = mujoco.MjData(self.模型)
        self.每帧物理步 = max(1, int(round(每帧秒 / self.模型.opt.timestep)))
        self.发力 = np.full(肌肉数, 0.5)
        self.目标角 = 站姿角.copy()
        self.时间 = 0.0
        self._渲染 = None
        self._渲染宽, self._渲染高 = 渲染宽, 渲染高
        self.摆成站姿()

    # ---------------- 摆姿势 ----------------
    def 摆成站姿(self):
        self.数据.qpos[:] = 0.0
        self.数据.qpos[2] = self.模型.key_qpos[0][2]
        self.数据.qpos[3] = 1.0
        self.数据.qpos[7:] = np.radians(站姿角)
        self.数据.qvel[:] = 0.0
        self.发力 = 站姿发力.copy()
        self.目标角 = 站姿角.copy()
        self.mujoco.mj_forward(self.模型, self.数据)

    def 摆成趴姿(self):
        self.数据.qpos[:] = 0.0
        self.数据.qpos[2] = 0.12
        self.数据.qpos[3] = 1.0
        self.数据.qpos[7:] = np.radians(趴姿角)
        self.数据.qvel[:] = 0.0
        self.发力 = 趴姿发力.copy()
        self.目标角 = 趴姿角.copy()
        self.mujoco.mj_forward(self.模型, self.数据)

    # ---------------- 一帧 ----------------
    def 步进(self, 发力=None, 记录=True):
        if 发力 is not None:
            self.发力 = np.clip(np.asarray(发力, dtype=float), 0.0, 1.0)
        self.目标角 = 发力转角度(self.发力)
        目标弧 = np.radians(self.目标角)
        for _ in range(self.每帧物理步):
            角 = self.数据.qpos[7:]
            力矩 = 刚度 * (目标弧 - 角) - 阻尼 * self.数据.qvel[6:]
            self.数据.ctrl[:] = np.clip(力矩, -最大力矩, 最大力矩)
            self.mujoco.mj_step(self.模型, self.数据)
            self.时间 += self.模型.opt.timestep
        return self.状态()

    def 走(self, 秒, 发力=None):
        for _ in range(int(round(秒 / 每帧秒))):
            self.步进(发力)

    # ---------------- 身体读数 ----------------
    def 机身高度(self):
        return float(self.数据.qpos[2])

    def 站直程度(self):
        """0=趴在地上，1=站姿高度。大脑用这个当"身体感觉"。"""
        return float(np.clip(self.数据.qpos[2] / self.模型.key_qpos[0][2], 0.0, 1.5))

    def 机身歪了(self):
        """机身相对于竖直倾斜了多少度（耳石那种感觉）。"""
        四元数 = self.数据.qpos[3:7]
        w, x, y, z = 四元数
        上 = np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])
        return float(np.degrees(np.arccos(np.clip(上[2], -1.0, 1.0))))

    def 上方向(self):
        """机身"头顶"在世界坐标里指哪儿（单位向量）。耳石/前庭那一路读的就是它。"""
        w, x, y, z = self.数据.qpos[3:7]
        return np.array([2 * (x * z + w * y), 2 * (y * z - w * x),
                         1 - 2 * (x * x + y * y)])

    def 关节偏差(self):
        """12 个关节角相对**站姿**的偏差（弧度）。本体感觉那一路读的就是它。"""
        return self.数据.qpos[7:] - np.radians(站姿角)

    def 关节角速度(self):
        """12 个关节的角速度（弧度/秒）。"""
        return self.数据.qvel[6:]

    def 足高(self):
        """四只脚离地多高（米）。Go2 没有单独的脚 body，脚长在小腿末端，
        所以这里取每条腿所有几何体「最低点」的估计值（几何体位置 - 包围球半径）。"""
        高 = []
        for 脚 in ("FL_calf", "FR_calf", "RL_calf", "RR_calf"):
            b = self.mujoco.mj_name2id(self.模型, self.mujoco.mjtObj.mjOBJ_BODY, 脚)
            z = [self.数据.geom_xpos[g][2] - self.模型.geom_rbound[g]
                 for g in range(self.模型.ngeom) if self.模型.geom_bodyid[g] == b]
            高.append(float(min(z)) if z else np.nan)
        return np.array(高)

    def 触地(self):
        """四条腿哪几条真的踩在地上了（看真实碰撞接触，不是估的）。"""
        腿身 = {}
        for 腿 in ("左前", "右前", "左后", "右后"):
            pass
        接触 = [False] * 4
        组 = [("FL_calf", 0), ("FR_calf", 1), ("RL_calf", 2), ("RR_calf", 3)]
        编号 = {self.mujoco.mj_name2id(self.模型, self.mujoco.mjtObj.mjOBJ_BODY, n): i for n, i in 组}
        for c in self.数据.contact[:self.数据.ncon]:
            for g in (c.geom1, c.geom2):
                b = int(self.模型.geom_bodyid[g])
                if b in 编号:
                    接触[编号[b]] = True
        return np.array(接触, dtype=bool)

    def 状态(self):
        return {
            "时间": self.时间,
            "机身高度": self.机身高度(),
            "站直": self.站直程度(),
            "歪": self.机身歪了(),
            "脚高": self.足高(),
            "触地": self.触地(),
        }

    # ---------------- 画一张 ----------------
    def 画(self, 相机=85.0, 俯角=-18.0, 距离=1.3, 看向=(0.0, 0.0, 0.28)):
        if self._渲染 is None:
            self._渲染 = self.mujoco.Renderer(self.模型, self._渲染高, self._渲染宽)
        cam = self.mujoco.MjvCamera()
        self.mujoco.mjv_defaultCamera(cam)
        cam.lookat[:] = 看向
        cam.distance = 距离
        cam.azimuth = 相机
        cam.elevation = 俯角
        self._渲染.update_scene(self.数据, cam)
        return self._渲染.render()


if __name__ == "__main__":
    from PIL import Image

    身 = 身体()
    print("肌肉数 %d  每帧物理步 %d（%.0f 帧/秒）" % (肌肉数, 身.每帧物理步, 1.0 / 每帧秒))
    print("站姿发力:", np.round(站姿发力, 3))
    print("趴姿发力:", np.round(趴姿发力, 3))

    print("\n[1] 站姿发力，走 10 秒：")
    身.摆成站姿()
    for k in range(300):
        身.步进(站姿发力)
        if k % 60 == 0:
            状 = 身.状态()
            print("   %4.1f 秒 机身高度 %.3f 米 站直 %.2f 歪 %.1f 度" %
                  (状["时间"], 状["机身高度"], 状["站直"], 状["歪"]))
    Image.fromarray(身.画()).save("身体_go2_站姿.png")

    print("\n[2] 趴姿发力，走 5 秒：")
    身.摆成趴姿()
    for k in range(150):
        身.步进(趴姿发力)
        if k % 30 == 0:
            状 = 身.状态()
            print("   %4.1f 秒 机身高度 %.3f 米 站直 %.2f 脚高 %s" %
                  (状["时间"], 状["机身高度"], 状["站直"], np.round(状["脚高"], 3)))
    Image.fromarray(身.画(距离=1.1, 看向=(0.0, 0.0, 0.15))).save("身体_go2_趴姿.png")
    print("\n两张图：身体_go2_站姿.png / 身体_go2_趴姿.png")
