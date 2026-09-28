"""Offline Go2 leg geometry audit: forward kinematics only, no controller policy."""

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(r"C:\mujoco_models\unitree_go2\scene.xml")
sys.path.insert(0, str(ROOT))


def stance_diagnosis():
    """One requested fixed-drive diagnostic on an independent simulated body."""
    sources = ["born_wired/innate.py", "born_wired/go2_body.py", "born_wired/reflex_senses.py",
               "born_wired/adaptive.py", "born_wired/regulation.py", "born_wired/encoding.py"]
    before_hashes = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources}
    from born_wired.go2_body import Go2Body
    from born_wired.innate import InnateController
    from born_wired.reflex_senses import ReflexSenses
    body = Go2Body()
    home = np.tile([0., .75, -1.5], 4)
    joint_ids = np.array([body.model.joint(name).id for name in body.joint_names])
    qpos_ids = body.model.jnt_qposadr[joint_ids]
    actuator_ids = np.array([body.model.actuator(name.removesuffix("_joint")).id for name in body.joint_names])
    feet = np.array([body.model.geom(name).id for name in ("FL", "FR", "RL", "RR")])
    base = body.model.body("base").id
    free_joint = int(body.model.body_jntadr[base])
    root_qpos = int(body.model.jnt_qposadr[free_joint])
    body.data.qpos[qpos_ids] = home
    mujoco.mj_forward(body.model, body.data)
    bottom = np.min(body.data.geom_xpos[feet, 2]-body.model.geom_size[feet, 0])
    body.data.qpos[root_qpos+2] += .005-bottom
    mujoco.mj_forward(body.model, body.data)
    brain = InnateController(home, body.lower_limits, body.upper_limits, seed=0,
                             balanced_gait=True, lift_gain=.45, gait_gain=.6, rhythm_speed=2.5)
    senses = ReflexSenses(body)
    observation = body.observe()
    dt = .01
    records = []
    started = time.perf_counter()
    for index in range(2300):
        drive = 0 if index < 300 else .65
        target, activation = brain.step(observation, dt=dt, locomotion=drive, learn=False)
        observation = body.step(target, duration=dt, activation=activation)
        environmental = senses.observe()
        rotation = body.data.xmat[base].reshape(3, 3)
        phase = brain.network.activity[brain.groups["phase"]]
        records.append(dict(time=float(body.data.time), drive=drive,
                            old_contact=observation["foot_contact"].tolist(),
                            support=environmental["foot_support"].tolist(),
                            load=environmental["foot_load_force_n"].tolist(),
                            clearance=(body.data.geom_xpos[feet, 2]-body.model.geom_size[feet, 0]
                                       -body.data.geom_xpos[body.model.geom("floor").id, 2]).tolist(),
                            slip=environmental["foot_slip_mps"].tolist(),
                            speed=np.linalg.norm(environmental["foot_velocity_world_mps"][:, :2], axis=1).tolist(),
                            target=target.reshape(4, 3).tolist(), actual=observation["joint_position"].reshape(4, 3).tolist(),
                            torque=body.data.ctrl[actuator_ids].reshape(4, 3).tolist(), phase=phase.tolist(),
                            height=observation["body_height"], up_z=float(observation["body_up"][2]),
                            com_body=(rotation.T@(body.data.subtree_com[base]-body.data.xpos[base])).tolist(),
                            foot_body=((body.data.geom_xpos[feet]-body.data.xpos[base])@rotation).tolist()))
    walking = records[300:]
    arrays = {key:np.asarray([row[key] for row in walking]) for key in walking[0]}
    per_leg = []
    for leg, name in enumerate(("FL", "FR", "RL", "RR")):
        own = 0 if leg in (0, 3) else 1
        lift_phase = arrays["phase"][:, own] > arrays["phase"][:, 1-own]
        load, contact = arrays["load"][:, leg], arrays["old_contact"][:, leg]
        clearance = arrays["clearance"][:, leg]
        error = arrays["actual"][:, leg] - arrays["target"][:, leg]
        control_limit = np.minimum(body.torque_limit, body.model.actuator_ctrlrange[actuator_ids[3*leg:3*leg+3], 1])
        per_leg.append(dict(leg=name, old_contact_fraction=float(contact.mean()),
                            support_fraction=float(arrays["support"][:, leg].mean()),
                            load_over_1N_fraction=float(np.mean(load>1)),
                            old_contact_but_load_le_1N_fraction=float(np.mean(contact & (load<=1))),
                            load_n=dict(min=float(load.min()), mean=float(load.mean()), median=float(np.median(load)), max=float(load.max())),
                            lift_phase_load_mean_n=float(load[lift_phase].mean()),
                            other_phase_load_mean_n=float(load[~lift_phase].mean()),
                            clearance_m=dict(min=float(clearance.min()), median=float(np.median(clearance)), max=float(clearance.max()),
                                             over_2mm_fraction=float(np.mean(clearance>.002))),
                            grounded_horizontal_speed_mean_mps=float(arrays["slip"][:, leg].mean()),
                            horizontal_speed_peak_mps=float(arrays["speed"][:, leg].max()),
                            mean_target_rad=arrays["target"][:, leg].mean(axis=0).tolist(),
                            mean_actual_rad=arrays["actual"][:, leg].mean(axis=0).tolist(),
                            joint_tracking_rmse_rad=np.sqrt(np.mean(error**2,axis=0)).tolist(),
                            motor_limit_fraction=np.mean(np.abs(arrays["torque"][:, leg])>=control_limit-1e-8,axis=0).tolist()))
    after_hashes = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources}
    return dict(config=dict(home_angles_rad=home.tolist(), balanced_gait=True, lift_gain=.45, gait_gain=.6,
                            rhythm_speed=2.5, locomotion=.65, standing_seconds=3, walking_seconds=20,
                            dt=dt, learn=False, seed=0, motor_units=100, floor="flat plane", sensors_read_only=True),
                source_sha256=before_hashes, source_sha256_after=after_hashes, source_unchanged=before_hashes==after_hashes,
                per_leg=per_leg, height_range_m=[float(arrays["height"].min()),float(arrays["height"].max())],
                minimum_up_z=float(arrays["up_z"].min()), mean_com_body_m=arrays["com_body"].mean(axis=0).tolist(),
                mean_foot_body_m=arrays["foot_body"].mean(axis=0).tolist(),
                sampled_records_10hz=records[::10], simulated_seconds=float(body.data.time),
                physics_steps=round(body.data.time/body.timestep), wall_seconds=time.perf_counter()-started,
                scope="One isolated 23s diagnostic with learning disabled and original contact feedback unchanged. Clearance is sphere-bottom height above this flat floor only; fixed drive is a diagnostic input, not a deployed policy.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stance-check", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    legs = ("FL", "FR", "RL", "RR")
    base = model.body("base").id
    for leg in legs:
        for joint, angle in zip(("hip", "thigh", "calf"), (0., .75, -1.5)):
            joint_id = model.joint(f"{leg}_{joint}_joint").id
            data.qpos[model.jnt_qposadr[joint_id]] = angle
    mujoco.mj_forward(model, data)
    base_qpos = data.qpos.copy()
    initial_time = float(data.time)
    eps = 1e-6

    def foot_position(geom):
        return data.xmat[base].reshape(3, 3).T @ (data.geom_xpos[geom]-data.xpos[base])

    cases = []
    for leg in legs:
        data.qpos[:] = base_qpos
        mujoco.mj_forward(model, data)
        geom = model.geom(leg).id
        joints = [model.joint(f"{leg}_{part}_joint").id for part in ("thigh", "calf")]
        addresses = np.array([model.jnt_qposadr[joint] for joint in joints])
        dofs = np.array([model.jnt_dofadr[joint] for joint in joints])
        position = foot_position(geom).copy()
        jac_world = np.zeros((3, model.nv))
        mujoco.mj_jacGeom(model, data, jac_world, None, geom)
        jacobian = data.xmat[base].reshape(3, 3).T @ jac_world[:, dofs]
        numeric = np.zeros((3, 2))
        for column, address in enumerate(addresses):
            data.qpos[:] = base_qpos
            data.qpos[address] += eps
            mujoco.mj_forward(model, data)
            plus = foot_position(geom).copy()
            data.qpos[address] -= 2*eps
            mujoco.mj_forward(model, data)
            numeric[:, column] = (plus-foot_position(geom)) / (2*eps)
        proposals = {}
        for name, desired in (("forward_10mm", np.array([.01, 0, 0])),
                              ("lift_10mm", np.array([0, 0, .01]))):
            delta = np.linalg.solve(jacobian[[0, 2]], desired[[0, 2]])
            data.qpos[:] = base_qpos
            data.qpos[addresses] += delta
            mujoco.mj_forward(model, data)
            actual = foot_position(geom)-position
            proposals[name] = dict(desired_body_displacement_m=desired.tolist(),
                                   joint_delta_rad=delta.tolist(), joint_delta_deg=np.rad2deg(delta).tolist(),
                                   linear_prediction_m=(jacobian@delta).tolist(),
                                   measured_displacement_m=actual.tolist(),
                                   nonlinear_error_m=float(np.linalg.norm(actual-desired)))
        # Small proposed -0.5 compensation is checked geometrically, not run
        # as a feedback controller or inverse-kinematics policy.
        compensated = np.array([.035, -.070])
        proposals["thigh_minus_half_calf"] = dict(joint_delta_rad=compensated.tolist(),
                                                    predicted_body_displacement_m=(jacobian@compensated).tolist())
        cases.append(dict(leg=leg, joint_order=["thigh", "calf"], foot_body_position_m=position.tolist(),
                          jacobian_body_m_per_rad=jacobian.tolist(), central_difference_m_per_rad=numeric.tolist(),
                          max_jacobian_error=float(np.max(np.abs(jacobian-numeric))),
                          pure_lift_thigh_per_calf=float(-jacobian[0, 1]/jacobian[0, 0]),
                          proposals=proposals))
    data.qpos[:] = base_qpos
    mujoco.mj_forward(model, data)
    jacobians = np.asarray([case["jacobian_body_m_per_rad"] for case in cases])
    report = dict(command="python tools/probe_leg_jacobian.py", python=platform.python_version(),
                  numpy=np.__version__, mujoco=mujoco.__version__, model_path=str(MODEL_PATH),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  model_xml_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest()
                                    for path in (MODEL_PATH, MODEL_PATH.parent/"go2.xml")},
                  config=dict(all_leg_angles_rad=[0, .75, -1.5], finite_difference_eps_rad=eps,
                              base_fixed=True, coordinates="body +x forward, +y left, +z up"),
                  cases=cases, max_inter_leg_jacobian_difference=float(np.max(np.abs(jacobians-jacobians[0]))),
                  time_before=initial_time, time_after=float(data.time), physics_steps=0,
                  wall_seconds=time.perf_counter()-started,
                  scope="Local forward-kinematics audit only. Inverse increments are diagnostic numbers, not a runtime controller, gait sequence or locomotion guarantee.")
    if args.stance_check:
        report["command"] += " --stance-check"
        report["stance_diagnosis"] = stance_diagnosis()
        report["total_physics_steps_including_diagnosis"] = report["stance_diagnosis"]["physics_steps"]
        report["scope"] += " An explicitly requested separate short physical stance diagnosis is attached."
    (ROOT/"artifacts/leg_jacobian.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    for case in cases:
        print(dict(leg=case["leg"], jacobian=case["jacobian_body_m_per_rad"],
                   lift_delta=case["proposals"]["lift_10mm"]["joint_delta_rad"],
                   forward_delta=case["proposals"]["forward_10mm"]["joint_delta_rad"],
                   compensation_ratio=case["pure_lift_thigh_per_calf"], error=case["max_jacobian_error"]))
    print("max_inter_leg_difference",report["max_inter_leg_jacobian_difference"],"physics_steps",report["physics_steps"])
    if args.stance_check:
        for leg in report["stance_diagnosis"]["per_leg"]:
            print(leg)
        print("diagnostic_source_unchanged",report["stance_diagnosis"]["source_unchanged"])


if __name__ == "__main__":
    main()
