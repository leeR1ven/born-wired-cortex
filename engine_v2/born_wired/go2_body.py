"""Named MuJoCo Go2 adapter: joint PD only, with no behaviour selection.

The twelve leg joints are torque motors driven by the caller. The four eye
muscles are position servos, so an eye target is an angle, not a torque;
they are commanded separately from the leg targets.
"""

import os
from pathlib import Path

import numpy as np
import mujoco


# Go2 那套文件跟着仓库走（models/unitree_go2），换台机器也一样。想用机器上另外一份，
# 就用环境变量 GO2模型目录 指过去。原来这里写死 C:\mujoco_models，非 Windows 的机器
# （比如云服务器）一加载就报找不到文件。
_MODEL_DIR = Path(os.environ.get("GO2模型目录") or Path(__file__).resolve().parents[1] / "models" / "unitree_go2")
DEFAULT_MODEL = str(_MODEL_DIR / "scene.xml")


def configured_model(value, fallback):
    """配置里写的模型路径，这台机器上没有就退回仓库那一份。

    live_config.json 是跟着仓库走的，里面那个 model 却是某台机器上的绝对路径，换台机器
    （比如云服务器）就变成解析 XML 失败。退回 fallback，并说一句用的是哪个。
    """
    if value:
        path = Path(value)
        if path.exists():
            return path
        print(f"配置里的模型不在这台机器上：{path} —— 改用 {fallback}")
    return fallback


def _scalar(value, name, positive=False):
    array = np.asarray(value)
    if array.shape or array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a real scalar")
    result = float(array)
    if not np.isfinite(result) or result < 0 or (positive and result == 0):
        raise ValueError(f"invalid {name}")
    return result


def _vector(value, length, name):
    array = np.asarray(value)
    if array.shape != (length,) or array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must have {length} real entries")
    result = np.array(array, dtype=float, copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


class Go2Body:
    """Joint order: FL, FR, RL, RR; each hip, thigh, calf. Angles are radians."""

    def __init__(self, model_path=DEFAULT_MODEL, timestep=0.002, kp=60, kd=3, torque_limit=25):
        self.kp = _scalar(kp, "kp")
        self.kd = _scalar(kd, "kd")
        self.torque_limit = _scalar(torque_limit, "torque_limit", positive=True)
        self.timestep = _scalar(timestep, "timestep", positive=True)
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.model.opt.timestep = self.timestep
        self.data = mujoco.MjData(self.model)
        legs = ("FL", "FR", "RL", "RR")
        motors = tuple(f"{leg}_{joint}" for leg in legs for joint in ("hip", "thigh", "calf"))
        self.joint_names = tuple(name + "_joint" for name in motors)
        self._joints = np.array([self._id(mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.joint_names])
        self._actuators = np.array([self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in motors])
        if (np.any(self.model.jnt_type[self._joints] != mujoco.mjtJoint.mjJNT_HINGE)
                or np.any(self.model.actuator_trnid[self._actuators, 0] != self._joints)
                or np.any(self.model.actuator_trntype[self._actuators] != mujoco.mjtTrn.mjTRN_JOINT)
                or not np.all(self.model.actuator_gear[self._actuators, 0] == 1)):
            raise ValueError("expected named hinge joints with direct unit-gear motors")
        self._qpos = self.model.jnt_qposadr[self._joints].copy()
        self._qvel = self.model.jnt_dofadr[self._joints].copy()
        self.lower_limits = self.model.jnt_range[self._joints, 0].copy()
        self.upper_limits = self.model.jnt_range[self._joints, 1].copy()
        self._base = self._id(mujoco.mjtObj.mjOBJ_BODY, "base")
        self._floor = self._id(mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self._feet = np.array([self._id(mujoco.mjtObj.mjOBJ_GEOM, leg) for leg in legs])
        base_joint = int(self.model.body_jntadr[self._base])
        if base_joint < 0 or self.model.jnt_type[base_joint] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("base must have a free joint")
        self._base_qpos = int(self.model.jnt_qposadr[base_joint])
        if not self.model.nkey:
            raise ValueError("stand reset requires keyframe 0")
        self.home_angles = self.model.key_qpos[0, self._qpos].copy()
        ranges = self.model.actuator_ctrlrange[self._actuators]
        limited = self.model.actuator_ctrllimited[self._actuators].astype(bool)
        self._ctrl_low = np.maximum(-self.torque_limit, np.where(limited, ranges[:, 0], -np.inf))
        self._ctrl_high = np.minimum(self.torque_limit, np.where(limited, ranges[:, 1], np.inf))
        # The bare robot has no eye bodies. The sensing arena adds four eye
        # muscles; a model without them simply has no eyes to command.
        eyes = tuple(f'eye_{side}_{dof}' for side in ('left', 'right') for dof in ('yaw', 'pitch'))
        present = [n for n in eyes if self.model.njnt and
                   mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n + '_joint') >= 0]
        if present and len(present) != len(eyes):
            raise ValueError("a model must have either all four eye muscles or none")
        self.eye_joint_names = tuple(name + '_joint' for name in eyes) if present else ()
        self.has_eyes = bool(present)
        eye_joints = (np.array([self._id(mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.eye_joint_names])
                      if present else np.empty(0, dtype=int))
        self._eye_actuators = (np.array([self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in eyes])
                               if present else np.empty(0, dtype=int))
        if present and (np.any(self.model.jnt_type[eye_joints] != mujoco.mjtJoint.mjJNT_HINGE)
                        or np.any(self.model.actuator_trnid[self._eye_actuators, 0] != eye_joints)
                        or np.any(self.model.actuator_biastype[self._eye_actuators] != mujoco.mjtBias.mjBIAS_AFFINE)
                        or not np.all(self.model.actuator_gainprm[self._eye_actuators, 0] > 0)):
            raise ValueError("expected four position-servo eye muscles")
        self._eye_qpos = self.model.jnt_qposadr[eye_joints].copy() if present else np.empty(0, dtype=int)
        self._eye_qvel = self.model.jnt_dofadr[eye_joints].copy() if present else np.empty(0, dtype=int)
        self.eye_lower_limits = self.model.jnt_range[eye_joints, 0].copy() if present else np.empty(0)
        self.eye_upper_limits = self.model.jnt_range[eye_joints, 1].copy() if present else np.empty(0)
        self._force_remaining = None
        self.reset()

    def _id(self, kind, name):
        index = mujoco.mj_name2id(self.model, kind, name)
        if index < 0:
            raise ValueError(f"model is missing {name}")
        return index

    def reset(self, pose="stand", seed=0, joint_noise=0, tilt=None):
        """Experiment initialization only. tilt is roll/pitch[/yaw] in radians."""
        if pose not in ("stand", "crouch"):
            raise ValueError("pose must be stand or crouch")
        noise = _scalar(joint_noise, "joint_noise")
        rng = np.random.default_rng(seed)
        rotation = None
        if tilt is not None:
            values = np.asarray(tilt)
            angles = _vector(values, 2 if values.shape == (2,) else 3, "tilt")
            if len(angles) == 2:
                angles = np.r_[angles, 0.0]
            cr, cp, cy = np.cos(angles / 2)
            sr, sp, sy = np.sin(angles / 2)
            rotation = np.array([cr*cp*cy + sr*sp*sy, sr*cp*cy - cr*sp*sy,
                                 cr*sp*cy + sr*cp*sy, cr*cp*sy - sr*sp*cy])
        target = self.home_angles.copy() if pose == "stand" else np.tile(np.deg2rad([0, 95, -156]), 4)
        with np.errstate(over="raise", invalid="raise"):
            target += rng.uniform(-1.0, 1.0, 12) * noise
        target = np.clip(target, self.lower_limits, self.upper_limits)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.data.ctrl[:] = 0
        self.data.qpos[self._qpos] = target
        if rotation is not None:
            q = self._base_qpos + 3
            home_rotation = self.data.qpos[q:q+4].copy()
            mujoco.mju_mulQuat(self.data.qpos[q:q+4], rotation, home_rotation)
        self.clear_body_force()
        mujoco.mj_forward(self.model, self.data)
        if pose == "crouch":
            bottom = np.min(self.data.geom_xpos[self._feet, 2] - self.model.geom_size[self._feet, 0])
            self.data.qpos[self._base_qpos + 2] += self.data.geom_xpos[self._floor, 2] + 0.003 - bottom
            mujoco.mj_forward(self.model, self.data)
        return self.observe()

    @property
    def eye_angles(self):
        """Eye muscle angles in radians: left yaw, left pitch, right yaw, right pitch."""
        result = self.data.qpos[self._eye_qpos].copy()
        result.flags.writeable = False
        return result

    @property
    def eye_velocities(self):
        result = self.data.qvel[self._eye_qvel].copy()
        result.flags.writeable = False
        return result

    def command_eyes(self, angles):
        """Set the eye muscle targets; they persist until changed."""
        if not self.has_eyes:
            raise RuntimeError("this model has no eye muscles")
        target = _vector(angles, 4, "angles")
        if np.any(target < self.eye_lower_limits) or np.any(target > self.eye_upper_limits):
            raise ValueError("eye angles exceed joint limits")
        self.data.ctrl[self._eye_actuators] = target
        return target

    def step(self, target_angles, duration=0.02, activation=None):
        target = _vector(target_angles, 12, "target_angles")
        recruitment = np.ones(12) if activation is None else _vector(activation, 12, "activation")
        if np.any(recruitment < 0) or np.any(recruitment > 1):
            raise ValueError("activation must lie in [0, 1]")
        if np.any(target < self.lower_limits) or np.any(target > self.upper_limits):
            raise ValueError("target_angles exceed joint limits")
        duration = _scalar(duration, "duration", positive=True)
        ratio = duration / self.timestep
        if not np.isfinite(ratio):
            raise ValueError("duration/timestep must be finite")
        steps = round(ratio)
        if steps < 1 or not np.isclose(ratio, steps, rtol=0, atol=1e-9):
            raise ValueError("duration must be a positive integer multiple of timestep")
        for _ in range(steps):
            with np.errstate(over="raise", invalid="raise"):
                torque = self.kp * (target - self.data.qpos[self._qpos]) - self.kd * self.data.qvel[self._qvel]
                torque *= recruitment
            if not np.isfinite(torque).all():
                raise FloatingPointError("non-finite actuator torque")
            self.data.ctrl[self._actuators] = np.clip(torque, self._ctrl_low, self._ctrl_high)
            before = self.data.time
            mujoco.mj_step(self.model, self.data)
            if self.data.time <= before or not (np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all()):
                raise FloatingPointError("MuJoCo state failed during physics stepping")
            if self._force_remaining is not None:
                self._force_remaining -= self.timestep
                if self._force_remaining <= self.timestep * 1e-9:
                    self.clear_body_force()
        mujoco.mj_forward(self.model, self.data)
        return self.observe()

    def set_body_force(self, force_xyz):
        """Set a persistent world-frame force at the base centre of mass; no time advance."""
        force = _vector(force_xyz, 3, "force_xyz")
        self.data.xfrc_applied[self._base] = np.r_[force, np.zeros(3)]
        self._force_remaining = None

    def clear_body_force(self):
        self.data.xfrc_applied[self._base] = 0
        self._force_remaining = None

    def apply_force(self, force_xyz, duration):
        """Apply during subsequent steps, clearing after duration rounded up to one timestep."""
        force = _vector(force_xyz, 3, "force_xyz")
        duration = _scalar(duration, "duration", positive=True)
        self.set_body_force(force)
        self._force_remaining = duration

    def observe(self):
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self._base, velocity, 0)
        rotation = self.data.xmat[self._base].reshape(3, 3)
        contact = np.zeros(4, dtype=bool)
        for item in self.data.contact[:self.data.ncon]:
            if item.dist <= 0:
                for i, foot in enumerate(self._feet):
                    if {int(item.geom1), int(item.geom2)} == {int(foot), self._floor}:
                        contact[i] = True
        # Rotating and re-expressing the world vertical by hand can land a
        # component a few ulps outside [-1, 1] in an exactly axis-aligned pose.
        # These two entries are unit vectors, so clip the rounding away here
        # instead of failing a validity check downstream.
        observation = dict(time=float(self.data.time), joint_position=self.data.qpos[self._qpos].copy(),
                           joint_velocity=self.data.qvel[self._qvel].copy(),
                           body_up=np.clip(rotation[:, 2], -1., 1.).copy(),
                           gravity_direction=np.clip(rotation.T @ np.array([0., 0., -1.]), -1., 1.),
                           angular_velocity_local=rotation.T @ velocity[:3],
                           body_angular_velocity=velocity[:3].copy(), body_linear_velocity=velocity[3:].copy(),
                           body_height=float(self.data.xpos[self._base, 2]), foot_contact=contact,
                           foot_position=self.data.geom_xpos[self._feet].copy(),
                           eye_position=self.data.qpos[self._eye_qpos].copy(),
                           eye_velocity=self.data.qvel[self._eye_qvel].copy(),
                           has_eyes=self.has_eyes)
        if not all(np.isfinite(value).all() for value in observation.values()):
            raise FloatingPointError("non-finite body observation")
        return observation
