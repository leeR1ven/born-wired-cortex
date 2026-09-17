# -*- coding: utf-8 -*-
"""仿真桥：把 MuJoCo 里的小人接到关联神经元架构上。

为什么要这个文件：
  脑区网络（视觉/听觉/运动/前额叶）只认"神经元亮不亮"，
  仿真软件只认"肌肉发力多少、画面长什么样"。
  这个文件就干两件事：
    出：把仿真画面 → 变成视觉输入层能吃的信号
    进：把运动皮层亮起的神经元 → 变成 10 个电机的发力

一个坑：MuJoCo 的 C++ 打不开带中文的路径（本机实测），
  所以这里一律把 xml 读成字符串再载入，不走 from_xml_path。

相机方向：小人的"眼"朝 +x（走路方向），仿真里 -x 是背后。
"""

from __future__ import annotations

import pathlib
import numpy as np

根 = pathlib.Path(__file__).resolve().parent
模型文件 = 根 / "仿真_小人.xml"

电机名 = ["左髋", "左膝", "左踝", "右髋", "右膝", "右踝", "左肩", "右肩", "转腰", "转颈"]
电机数 = len(电机名)

# 每次 mj_step 是 5 毫秒仿真时间；视觉/皮层一拍按 20 毫秒算（≈ 50 Hz 行为频率）
每拍物理步数 = 4


def 载入模型(路径=None):
    """读出 xml 文本并载入。返回 (m, d)。"""
    import mujoco
    p = pathlib.Path(路径) if 路径 else 模型文件
    xml = p.read_text(encoding="utf-8")
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    return m, d


class 仿真:
    """一个小人 + 一台相机 + 10 个电机。"""

    def __init__(self, 宽=640, 高=480, 路径=None, 开渲染=True):
        import mujoco
        self.mujoco = mujoco
        self.m, self.d = 载入模型(路径)
        self.宽, self.高 = 宽, 高
        self.相机号 = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_CAMERA, "眼")
        self._r = mujoco.Renderer(self.m, 高, 宽) if 开渲染 else None
        self.每拍物理步数 = 每拍物理步数
        self.累计物理步 = 0

    # ---------- 身体 ----------
    def 复位(self):
        self.mujoco.mj_resetData(self.m, self.d)
        self.mujoco.mj_forward(self.m, self.d)

    def 步进(self, 物理步数=None):
        n = self.每拍物理步数 if 物理步数 is None else 物理步数
        for _ in range(n):
            self.mujoco.mj_step(self.m, self.d)
        self.累计物理步 += n

    def 写发力(self, 十路发力):
        """十路发力：长度 10 的数组，每个 -1~1（正负方向）。"""
        v = np.clip(np.asarray(十路发力, dtype=float).ravel(), -1.0, 1.0)
        if v.size != 电机数:
            raise ValueError(f"要有 {电机数} 路发力，收到 {v.size}")
        self.d.ctrl[:] = v

    def 电机角度(self):
        """每个电机对应关节当前的角度，用来判断"动作做出来没有"。"""
        return np.array([self.d.qpos[self.m.jnt_qposadr[
            self.mujoco.mj_name2id(self.m, self.mujoco.mjtObj.mjOBJ_JOINT, n)]] for n in 电机名])

    def 躯干位置(self):
        return self.d.qpos[0:3].copy()

    def 倒了没(self):
        """躯干中心离地高度；低了就是倒了。"""
        return float(self.d.qpos[2]) < 0.75

    # ---------- 眼睛 ----------
    def 看(self):
        """返回相机看到的画面 uint8 (高, 宽, 3)。"""
        if self._r is None:
            raise RuntimeError("建的时候开了 开渲染=False")
        self._r.update_scene(self.d, camera=self.相机号)
        return self._r.render()

    def 看别人的眼(self, 相机名="眼"):
        """用指定相机看（默认就是自己眼睛）。"""
        cid = self.mujoco.mj_name2id(self.m, self.mujoco.mjtObj.mjOBJ_CAMERA, 相机名)
        self._r.update_scene(self.d, camera=cid)
        return self._r.render()

    def 关(self):
        if self._r is not None:
            self._r.close()
            self._r = None


def 发力转十路(肌肉激活: np.ndarray, 每组=20):
    """运动皮层亮起的神经元 -> 10 路发力。

    运动区一共 肌肉数 块肌肉（每块 每档对数 对神经元）。
    这里把肌肉分成 10 组，每组 20 块；一组里平均发力多少就代表这一路出力多少，
    再乘 2 减 1 变成 -1~1（0.5 发力 = 不动）。
    """
    import importlib
    运动 = importlib.import_module("运动输出区_motor_output")
    每档 = 运动.每档对数
    正 = np.asarray(肌肉激活, dtype=bool)[0::2].reshape(运动.肌肉数, 每档)
    档 = 正.sum(axis=1) / 每档                      # 每块肌肉 0~1
    组数 = 运动.肌肉数 // 每组
    出力 = 档.reshape(组数, 每组).mean(axis=1)      # 每组 0~1
    if 组数 >= 10:
        出力 = 出力[:10]
    else:
        出力 = np.concatenate([出力, np.full(10 - 组数, 0.5)])
    return (出力 - 0.5) * 2.0


if __name__ == "__main__":
    import time
    s = 仿真()
    s.复位()
    print("电机数", s.m.nu, "相机", s.相机号)
    print("初始躯干高度 %.3f" % s.躯干位置()[2])
    t0 = time.perf_counter()
    for _ in range(200):
        s.步进(1)
    t1 = time.perf_counter()
    print("物理 %.4f ms/步 (%d 步/s)" % ((t1-t0)/200*1000, 200/(t1-t0)))
    t0 = time.perf_counter()
    for _ in range(100):
        img = s.看()
    t1 = time.perf_counter()
    print("渲染 %dx%d %.2f ms/帧 (%.0f fps)  画面均值 %.1f" % (s.宽, s.高, (t1-t0)/100*1000, 100/(t1-t0), img.mean()))
    s.关()
