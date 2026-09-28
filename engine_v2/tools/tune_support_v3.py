"""Bounded offline parameter comparison; never changes the live controller."""
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOURCES = ["born_wired/innate.py", "born_wired/go2_body.py", "born_wired/reflex_senses.py",
           "born_wired/adaptive.py", "born_wired/regulation.py", "born_wired/synapses.py",
           "born_wired/encoding.py", "tools/tune_support_v3.py"]
IMPORT_HASHES = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCES}
from born_wired.go2_body import Go2Body
from born_wired.innate import InnateController
from born_wired.reflex_senses import ReflexSenses
from born_wired.regulation import RegulatedSynapses

OUTPUT = ROOT / "artifacts/support_tuning_v3.json"
LEGS = ("FL", "FR", "RL", "RR")
DT, STAND_SECONDS, WALK_SECONDS = .01, 3, 20
SCORE_SPEC = dict(forward_saturation_mps=.10, rear_air_target=.25,
                  weights=dict(forward=2., rear_air=1.5, slip=-1.5, air_imbalance=-.75,
                               load_imbalance=-.5, pitch=-.35, target_jump=-.25),
                  normalizers=dict(slip_mps=.05, air_imbalance=.15, load_imbalance=.10,
                                   pitch_rad=.10, target_jump_rad=.05),
                  eligibility=dict(min_forward_mps=.01, min_late_forward_mps=.01,
                                   min_path_efficiency=.75, min_height_m=.20,
                                   min_up_z=.866, max_target_jump_rad=.35))


def hashes():
    return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCES}


def candidates():
    base = dict(thigh_offset=0., rear_scale=1., lift_gain=.45, gait_gain=.6, rhythm_speed=2.5)
    result = [dict(name="baseline", **base)]
    for offset in (.05, .10, .15):
        result.append(dict(base, name=f"offset_{offset:.2f}", thigh_offset=offset))
    for scale in (1.2, 1.4, 1.6, 1.8):
        result.append(dict(base, name=f"rear_{scale:.1f}", thigh_offset=.10, rear_scale=scale))
    for field, values in (("lift_gain", (.35, .55)), ("gait_gain", (.45, .75)),
                          ("rhythm_speed", (2., 3.))):
        for value in values:
            item = dict(base, name=f"{field}_{value:g}", thigh_offset=.10, rear_scale=1.4)
            item[field] = value
            result.append(item)
    assert len(result) == 14
    return result


def scale_rear_lift(brain, factor):
    """Rebuild this instance's graph once, preserving topology and neural state.

    Direct phase E/I paths to rear thigh AND calf scale together; delayed swing
    stays untouched. Developmental bounds and tether anchors follow the scale,
    while signs, plasticity, incoming budgets and learning rates stay unchanged.
    """
    old = brain.synapses
    sources = np.r_[brain.groups["phase"], brain.groups["phase_inhibitory"]]
    targets = brain.groups["motor"].reshape(12, brain.motor_units)[[7, 8, 10, 11]].ravel()
    selected = np.isin(old.src, sources) & np.isin(old.dst, targets)
    assert int(selected.sum()) == 8 * brain.motor_units
    multiplier = np.where(selected, factor, 1.)
    proposed = old.weights * multiplier
    new = RegulatedSynapses(old.src, old.dst, proposed, old.signs, old.n_neurons,
                            lower=old.lower*multiplier, upper=old.w_max*multiplier,
                            budgets=old.budgets, plasticity=old.plasticity,
                            learning_rate=old.learning_rate, tether=old.tether,
                            target_activity=old.target_activity)
    new.anchor = old.anchor * multiplier
    new.anchor.flags.writeable = False
    assert np.array_equal(new.weights, proposed), "budget projection changed the proposed genotype"
    brain.synapses = new
    brain.network.synapses = new
    brain.initial_weights = new.weights
    return dict(selected_edge_count=int(selected.sum()), changed_edge_count=int(selected.sum()) if factor != 1 else 0,
                scale=factor, neuron_count=old.n_neurons, edge_count=len(proposed),
                scaled_fields=["weights", "anchor", "lower", "upper"],
                unmodified="phase_delay paths, topology, Dale signs, budgets, plasticity, tether, neural states/parameters")


def score(metrics):
    m = metrics
    components = dict(forward=min(max(m["forward_mps"], 0)/.10, 1),
                      rear_air=min(m["rear_air_fraction"]/.25, 1),
                      slip=m["load_weighted_foot_speed_mps"]/.05,
                      air_imbalance=abs(m["front_air_fraction"]-m["rear_air_fraction"])/.15,
                      load_imbalance=abs(m["rear_load_share"]-.5)/.10,
                      pitch=m["pitch_abs_p95_rad"]/.10,
                      target_jump=m["target_jump_p99_rad"]/.05)
    components = {key: float(value*SCORE_SPEC["weights"][key]) for key, value in components.items()}
    checks = dict(forward=m["forward_mps"]>.01, late_forward=m["late_forward_mps"]>.01,
                  path_efficiency=m["path_efficiency"]>.75, height=m["minimum_height_m"]>.20,
                  upright=m["minimum_up_z"]>.866, jump=m["target_jump_max_rad"]<.35)
    return dict(score=float(sum(components.values())), components=components,
                eligible=all(checks.values()), eligibility_checks=checks)


def run_case(config, learn):
    started, before = time.perf_counter(), hashes()
    body = Go2Body()
    home = np.tile([0., .75+config["thigh_offset"], -1.5], 4)
    joint_ids = np.array([body.model.joint(name).id for name in body.joint_names])
    feet = np.array([body.model.geom(name).id for name in LEGS])
    base = body.model.body("base").id
    floor = body.model.geom("floor").id
    root_qpos = body.model.jnt_qposadr[body.model.body_jntadr[base]]
    body.data.qpos[body.model.jnt_qposadr[joint_ids]] = home
    mujoco.mj_forward(body.model, body.data)
    bottom = np.min(body.data.geom_xpos[feet, 2]-body.model.geom_size[feet, 0])
    body.data.qpos[root_qpos+2] += .005-bottom
    mujoco.mj_forward(body.model, body.data)
    brain = InnateController(home, body.lower_limits, body.upper_limits, seed=0,
                             balanced_gait=True, lift_gain=config["lift_gain"],
                             gait_gain=config["gait_gain"], rhythm_speed=config["rhythm_speed"])
    modification = scale_rear_lift(brain, config["rear_scale"])
    senses = ReflexSenses(body)
    observation, records, samples = body.observe(), [], []
    previous_target = home.copy()
    positions = []
    for index in range(round((STAND_SECONDS+WALK_SECONDS)/DT)):
        drive = 0. if index < round(STAND_SECONDS/DT) else .65
        target, activation = brain.step(observation, dt=DT, locomotion=drive, learn=learn)
        jump = float(np.max(np.abs(target-previous_target)))
        previous_target = target.copy()
        observation = body.step(target, duration=DT, activation=activation)
        env = senses.observe()
        rotation = body.data.xmat[base].reshape(3, 3)
        position = body.data.xpos[base].copy()
        if index == round(STAND_SECONDS/DT)-1:
            origin = position.copy()
            heading = rotation[:, 0].copy()
            heading[2] = 0
            heading /= np.linalg.norm(heading)
            left = np.array([-heading[1], heading[0], 0.])
            positions.append(position.copy())
        row = dict(time=float(body.data.time), drive=drive,
                   load=env["foot_load_force_n"].tolist(), support=env["foot_support"].tolist(),
                   old_contact=observation["foot_contact"].tolist(),
                   clearance=(body.data.geom_xpos[feet, 2]-body.model.geom_size[feet, 0]
                              -body.data.geom_xpos[floor, 2]).tolist(),
                   slip=env["foot_slip_mps"].tolist(),
                   speed=np.linalg.norm(env["foot_velocity_world_mps"][:, :2], axis=1).tolist(),
                   height=float(observation["body_height"]), up_z=float(observation["body_up"][2]),
                   pitch=float(np.arcsin(np.clip(-rotation[2, 0], -1, 1))),
                   target_jump=jump, target=target.reshape(4, 3).tolist(),
                   actual=observation["joint_position"].reshape(4, 3).tolist(),
                   position=position.tolist(), effort_proxy_w=env["effort_proxy_w"])
        if index % 100 == 0:
            samples.append(row)
        if drive:
            records.append(row)
            positions.append(position.copy())
    a = {key: np.asarray([row[key] for row in records]) for key in records[0]}
    positions = np.asarray(positions)
    displacement = positions[-1]-origin
    path_length = float(np.linalg.norm(np.diff(positions[:, :2], axis=0), axis=1).sum())
    load, speed, air = a["load"], a["speed"], a["clearance"]>.002
    weighted = lambda values, weights: float(np.sum(values*weights)/max(np.sum(weights), 1e-12))
    per_leg = []
    for leg, name in enumerate(LEGS):
        per_leg.append(dict(leg=name, load_mean_n=float(load[:, leg].mean()),
                            load_p95_n=float(np.quantile(load[:, leg], .95)),
                            support_fraction=float(a["support"][:, leg].mean()),
                            air_over_2mm_fraction=float(air[:, leg].mean()),
                            clearance_p95_m=float(np.quantile(a["clearance"][:, leg], .95)),
                            clearance_max_m=float(a["clearance"][:, leg].max()),
                            load_weighted_foot_speed_mps=weighted(speed[:, leg], load[:, leg]),
                            grounded_speed_time_mean_mps=float(a["slip"][:, leg].mean()),
                            false_old_contact_fraction=float(np.mean(a["old_contact"][:, leg] & (load[:, leg]<=1))),
                            joint_tracking_rmse_rad=np.sqrt(np.mean((a["actual"][:, leg]-a["target"][:, leg])**2, axis=0)).tolist()))
    metrics = dict(forward_mps=float(displacement@heading/WALK_SECONDS),
                   forward_m=float(displacement@heading), lateral_m=float(displacement@left),
                   late_forward_mps=float((positions[-1]-positions[-1001])@heading/10),
                   path_length_m=path_length, path_efficiency=float(displacement@heading/max(path_length, 1e-12)),
                   front_air_fraction=float(air[:, :2].mean()), rear_air_fraction=float(air[:, 2:].mean()),
                   rear_load_share=float(load[:, 2:].sum()/max(load.sum(), 1e-12)),
                   load_weighted_foot_speed_mps=weighted(speed, load),
                   rear_load_weighted_foot_speed_mps=weighted(speed[:, 2:], load[:, 2:]),
                   front_load_weighted_foot_speed_mps=weighted(speed[:, :2], load[:, :2]),
                   pitch_mean_rad=float(a["pitch"].mean()), pitch_abs_p95_rad=float(np.quantile(np.abs(a["pitch"]), .95)),
                   pitch_abs_max_rad=float(np.abs(a["pitch"]).max()),
                   minimum_height_m=float(a["height"].min()), minimum_up_z=float(a["up_z"].min()),
                   target_jump_max_rad=float(a["target_jump"].max()),
                   target_jump_p99_rad=float(np.quantile(a["target_jump"], .99)),
                   mean_effort_proxy_w=float(a["effort_proxy_w"].mean()))
    after = hashes()
    return dict(config=config, learn=learn, seed=0, status="completed", home_angles_rad=home.tolist(),
                source_sha256_before=before, source_sha256_after=after, source_unchanged=before==after==IMPORT_HASHES,
                modification=modification, metrics=metrics, ranking=score(metrics), per_leg=per_leg,
                diagnostics=brain.diagnostics(), samples_1hz=samples,
                simulated_seconds=float(body.data.time), wall_seconds=time.perf_counter()-started)


def main():
    started = time.perf_counter()
    model_path = Path(r"C:\mujoco_models\unitree_go2\scene.xml")
    report = dict(command="python tools/tune_support_v3.py", python=platform.python_version(),
                  numpy=np.__version__, mujoco=mujoco.__version__, source_sha256=IMPORT_HASHES,
                  model_xml_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                    for path in (model_path, model_path.parent/"go2.xml")},
                  common_config=dict(seed=0, dt=DT, physics_dt=.002, stand_seconds=STAND_SECONDS,
                                     move_seconds=WALK_SECONDS, locomotion=.65, balanced_gait=True,
                                     motor_units=100, proprio_units=32, association_units=128,
                                     balance_gain=.22, swing_delay=.16, scaffold_fraction=.03,
                                     kp=60, kd=3, torque_limit=25, floor="flat plane"),
                  candidate_grid=candidates(), score_specification=SCORE_SPEC, cases=[],
                  scope="Offline parameter experiment only. Fixed drive is an experimental input; all joint commands still come from the existing neural motor populations. No live model/controller is touched.",
                  metric_limits="Foot speed is the horizontal foot geom-centre velocity, load weighted, not exact material-point friction slip; clearance uses this flat plane only. Eligibility thresholds and multiobjective weights are diagnostic choices, not proven safety or global optimality.")

    def execute(config, learn):
        try:
            case = run_case(config, learn)
        except Exception as exc:
            case = dict(config=config, learn=learn, seed=0, status="failed", error=repr(exc))
        report["cases"].append(case)
        OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        if case["status"] == "completed":
            m = case["metrics"]
            print(json.dumps(dict(name=config["name"], learn=learn, score=round(case["ranking"]["score"], 3),
                                  eligible=case["ranking"]["eligible"], forward=round(m["forward_mps"], 4),
                                  air=[round(m["front_air_fraction"], 3), round(m["rear_air_fraction"], 3)],
                                  slip=round(m["load_weighted_foot_speed_mps"], 4),
                                  rear_load=round(m["rear_load_share"], 3),
                                  pitch=round(m["pitch_abs_p95_rad"], 4), jump=round(m["target_jump_max_rad"], 4))), flush=True)
        else:
            print(json.dumps(case), flush=True)

    for config in report["candidate_grid"]:
        execute(config, False)
    complete = [case for case in report["cases"] if case["status"] == "completed"]
    ranked = sorted(complete, key=lambda item: (item["ranking"]["eligible"], item["ranking"]["score"]), reverse=True)
    report["coarse_ranking"] = [case["config"]["name"] for case in ranked]
    report["selected_for_learning"] = [case["config"]["name"] for case in ranked[:2]]
    for case in ranked[:2]:
        execute(case["config"], True)
    report["source_sha256_after"] = hashes()
    report["source_unchanged"] = report["source_sha256_after"] == IMPORT_HASHES and all(
        case.get("source_unchanged", False) for case in report["cases"])
    report["total_simulated_seconds"] = sum(case.get("simulated_seconds", 0) for case in report["cases"])
    report["wall_seconds"] = time.perf_counter()-started
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(dict(selected=report["selected_for_learning"], source_unchanged=report["source_unchanged"],
                          total_simulated_seconds=report["total_simulated_seconds"], wall_seconds=report["wall_seconds"])), flush=True)


if __name__ == "__main__":
    main()
