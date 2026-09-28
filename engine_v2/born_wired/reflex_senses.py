"""Read-only MuJoCo environment senses; no action selection or physics step."""

import copy

import mujoco
import numpy as np

from .synapses import _scalar, _vector


class ReflexSenses:
    """Rays L/C/R; legs FL/FR/RL/RR; body sectors front/left/back/right.

    Normalized scales: body 80 N, obstacle 40 N, load 80 N, slip .30 m/s.
    Support requires positive solved normal force and a force-normal/world-up
    cosine >= .5 (default), on a named foot geom and any nonrobot geom.
    """

    def __init__(self, body, *, max_range=2., sensor_offset=(.35, 0, .03), support_cosine=.5,
                 include_ranges=False):
        if not isinstance(include_ranges, (bool, np.bool_)):
            raise ValueError('include_ranges must be boolean')
        self.include_ranges = bool(include_ranges)
        self.max_range = _scalar(max_range, "max_range", positive=True)
        self.sensor_offset = _vector(sensor_offset, 3, "sensor_offset")
        self.support_cosine = _scalar(support_cosine, "support_cosine", positive=True)
        if self.support_cosine > 1:
            raise ValueError("support_cosine must be in (0, 1]")
        if not isinstance(getattr(body, "model", None), mujoco.MjModel) or not isinstance(getattr(body, "data", None), mujoco.MjData):
            raise ValueError("body must expose a MuJoCo model and data")
        self.body, self.model = body, body.model
        self._base = self._id(mujoco.mjtObj.mjOBJ_BODY, "base")
        self._robot_bodies = self._subtree(self._base)
        self._robot_geoms = self._robot_bodies[self.model.geom_bodyid]
        self._feet = np.array([self._id(mujoco.mjtObj.mjOBJ_GEOM, name) for name in ("FL", "FR", "RL", "RR")])
        if not self._robot_geoms[self._feet].all():
            raise ValueError("foot geoms must belong to the robot subtree")
        self._leg_for_geom = np.full(self.model.ngeom, -1, dtype=int)
        for index, name in enumerate(("FL_hip", "FR_hip", "RL_hip", "RR_hip")):
            limb = self._subtree(self._id(mujoco.mjtObj.mjOBJ_BODY, name))
            self._leg_for_geom[limb[self.model.geom_bodyid]] = index
        static = np.zeros(self.model.nbody, dtype=bool)
        static[0] = True
        for index in range(1, self.model.nbody):
            static[index] = (static[self.model.body_parentid[index]] and self.model.body_jntnum[index] == 0
                             and self.model.body_mocapid[index] < 0)
        self._static_environment = static[self.model.geom_bodyid] & ~self._robot_geoms
        # mj_ray excludes only one body ID, not a subtree. A private model
        # copy supplies the exclusion mask without changing live render groups.
        # Geometry poses and contact state always come from current body.data.
        self._ray_model = copy.copy(self.model)
        self._ray_model.geom_group[:] = 5
        self._ray_model.geom_group[self._static_environment] = 0
        self._ray_groups = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
        angles = np.deg2rad([35., 0, -35.])
        self._ray_local = np.column_stack((np.cos(angles), np.sin(angles), np.zeros(3)))

    def _id(self, kind, name):
        result = mujoco.mj_name2id(self.model, kind, name)
        if result < 0:
            raise ValueError(f"missing named object: {name}")
        return result

    def _subtree(self, root):
        included = np.zeros(self.model.nbody, dtype=bool)
        included[root] = True
        for index in range(root + 1, self.model.nbody):
            included[index] = included[self.model.body_parentid[index]]
        return included

    def _rays(self, rotation, base_position, data):
        # Keep relevant geometry properties current; only geom_group differs.
        for name in ("geom_type", "geom_size", "geom_dataid", "geom_rbound", "geom_rgba", "geom_matid",
                     "mat_rgba", "hfield_data", "hfield_size", "mesh_vert"):
            np.copyto(getattr(self._ray_model, name), getattr(self.model, name))
        origin = base_position + rotation @ self.sensor_offset
        directions = self._ray_local @ rotation.T
        raw = np.empty(3)
        geom_ids = np.full(3, -1, dtype=np.int32)
        for index, direction in enumerate(directions):
            hit = np.full(1, -1, dtype=np.int32)
            raw[index] = mujoco.mj_ray(self._ray_model, data, origin, direction,
                                       self._ray_groups, True, -1, hit)
            geom_ids[index] = hit[0]
        if not np.isfinite(raw).all():
            raise FloatingPointError("non-finite ray distance")
        hit = (raw >= 0) & (raw <= self.max_range)
        distance = np.where(hit, raw, self.max_range)
        return dict(ray_distance=distance, ray_proximity=1-distance/self.max_range,
                    ray_raw_distance_m=raw, ray_hit=hit, ray_geom_id=geom_ids,
                    ray_origin_world=origin, ray_direction_world=directions)

    def observe(self):
        """Return fresh arrays; caller must have forwarded the current MuJoCo state."""
        if self.body.model is not self.model:
            raise ValueError("construct new senses after replacing the body's model")
        data = self.body.data
        rotation = data.xmat[self._base].reshape(3, 3)
        position = data.xpos[self._base]
        gravity = self.model.opt.gravity
        if not (np.isfinite(rotation).all() and np.isfinite(position).all() and np.isfinite(gravity).all()):
            raise FloatingPointError("non-finite body pose or gravity")
        magnitude = np.linalg.norm(gravity)
        up = -gravity / magnitude if magnitude > 0 else np.array([0., 0., 1.])
        result = self._rays(rotation, position, data) if self.include_ranges else {}
        touch, obstacle, load = np.zeros(4), np.zeros(4), np.zeros(4)
        support, foot_contact = np.zeros(4, dtype=bool), np.zeros(4, dtype=bool)
        foot_lookup = {int(geom): index for index, geom in enumerate(self._feet)}
        for index, contact in enumerate(data.contact[:data.ncon]):
            g1, g2 = int(contact.geom1), int(contact.geom2)
            if g1 < 0 or g2 < 0 or self._robot_geoms[g1] == self._robot_geoms[g2] or contact.efc_address < 0:
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, data, index, force)
            if not (np.isfinite(force).all() and np.isfinite(contact.frame).all() and np.isfinite(contact.pos).all()):
                raise FloatingPointError("non-finite contact state")
            if force[0] <= 0:
                continue
            robot_geom = g1 if self._robot_geoms[g1] else g2
            normal_on_robot = contact.frame[:3] * (-1 if robot_geom == g1 else 1)
            normal_force = float(force[0])
            strength = float(np.linalg.norm(force[:3]))
            foot_index = foot_lookup.get(robot_geom)
            is_support = foot_index is not None and float(normal_on_robot @ up) >= self.support_cosine
            if foot_index is not None:
                foot_contact[foot_index] = True
                if is_support:
                    support[foot_index] = True
                    load[foot_index] += normal_force
            else:
                local_point = rotation.T @ (contact.pos-position)
                sector = int(np.argmax([local_point[0], local_point[1], -local_point[0], -local_point[1]]))
                touch[sector] += strength
            leg = self._leg_for_geom[robot_geom]
            if leg >= 0 and not is_support:
                obstacle[leg] += strength
        velocities = np.empty((4, 3))
        for index, geom in enumerate(self._feet):
            velocity = np.zeros(6)
            mujoco.mj_objectVelocity(self.model, data, mujoco.mjtObj.mjOBJ_GEOM, int(geom), velocity, 0)
            velocities[index] = velocity[3:]
        horizontal = velocities - np.outer(velocities @ up, up)
        slip = np.where(foot_contact, np.linalg.norm(horizontal, axis=1), 0)
        result.update(body_touch=np.clip(touch/80., 0, 1), foot_obstacle=np.clip(obstacle/40., 0, 1),
                      foot_load=np.clip(load/80., 0, 1), foot_slip=np.clip(slip/.30, 0, 1),
                      body_touch_force_n=touch, foot_obstacle_force_n=obstacle, foot_load_force_n=load,
                      foot_slip_mps=slip, foot_velocity_world_mps=velocities,
                      foot_support=support, foot_contact=foot_contact, world_up=up.copy(),
                      time=float(data.time))
        mechanical_power = float(np.sum(np.abs(data.actuator_force * data.actuator_velocity)))
        # Explicit effort proxy, not battery charge or a biological energy unit.
        effort_proxy = mechanical_power + .015 * float(np.sum(data.actuator_force ** 2))
        result.update(mechanical_power_w=mechanical_power, effort_proxy_w=effort_proxy,
                      motor_effort=float(np.clip(effort_proxy / 30., 0, 1)))
        if not all(np.isfinite(value).all() for value in result.values()):
            raise FloatingPointError("non-finite environment observation")
        return result
