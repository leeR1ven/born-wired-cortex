# -*- coding: utf-8 -*-
"""眼睛第一关（红球注视）：随机接线 -> 成批生成 -> 按「眼睛有没有朝球转」挑爹。

接线只从 retinal_opponent 的红色通道来（球是红的，这层只认红色），位置野用实测表
artifacts/红球位置野_红.json —— 列号到方位不是直线，中间密边缘疏，用一条拟合直线
在边缘能差好几列。每根线的权重是

    权重 = 该方向的总增益 * (|角度| / 满量程) * 每细胞随机抖动

一根线从「球落在某个位置时最亮的那个红细胞」出发，走到「那个方向的那一档力度」的
指令细胞上。方位为正（机器人左侧）走 +yaw 那一档，方位为负走 -yaw 那一档；高低同理，
只是 +pitch 是往下看（探针 tools/probe_eye_axis.py 量的：把眼球转 +0.30 弧度，
画面里的亮块行号往下走）。离正前方 3 度以内的细胞不接，球已经在正中了不该再动。

力度两档（用户要的「两种力度」）：离正中远的细胞进「移动」档（推力大，眼睛转过去），
离得近的进「保持」档（推力小，眼睛轻轻挪一下然后靠这股力停住）。没有信号时两档都不
放电，眼肌自己的静息电流把眼睛带回正前方（本来就接着，不用另加）。

测试：把红球摆在 9 个位置（左右 3 x 上下 3），每个位置跑 `--steps` 步，读最后
`--tail` 步的平均指令角，和球的真实方位/高低比。命中 = 两个轴都在 `--tolerance` 以内。

    python tools/evolve_eye.py --candidates 200 --generations 3 --workers 16 \
        --out-prefix artifacts/眼睛_红球注视
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_name, "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np                                                      # noqa: E402

from born_wired.embodied import EmbodiedController                    # noqa: E402
from born_wired.stereo_senses import RawEyes                          # noqa: E402
from tools import taskbank as tb                                      # noqa: E402

DT = .01
TABLE = ROOT / "artifacts" / "红球位置野_红.json"
DEAD = np.radians(3.)              # 死区：球离正视方向这么近就不许再动
FULL = {"yaw": .45, "pitch": .36}  # 扫场表里方位/高低的满量程（弧度）
SILENT = 1e-9   # 把别的看东西的通路掐掉，单看红色这一条；它们各自要求正数，所以给约等于 0 的数
MUTED = ("eye_track_gain", "eye_pitch_gain", "eye_gaze_inhibition", "eye_change_gain",
         "eye_change_relay_gain", "eye_change_common", "eye_sound_gain", "eye_memory_gain",
         "eye_orient_gain", "eye_row_relay_gain", "eye_row_gain", "eye_row_common",
         "eye_band_gain", "eye_band_common", "eye_band_push", "eye_vergence_gain",
         "eye_vergence_steps", "eye_vergence_baseline", "eye_vergence_inhibition",
         "eye_near_gain", "eye_fusion_gain", "eye_wall_gain", "eye_loom_gain",
         "eye_stereo_gain", "eye_distance_slope")

DEFAULTS = dict(red_gain_left=.30, red_gain_right=.30, red_gain_up=.30, red_gain_down=.30,
                red_tier_split=.12, red_flip=.25, red_jitter=.35, red_keep=.85,
                red_hold_share=.30, red_move_push=.45, red_hold_push=.10,
                red_seed=0)
BOUNDS = dict(red_gain_left=(.003, 8.), red_gain_right=(.003, 8.),
              red_gain_up=(.003, 8.), red_gain_down=(.003, 8.),
              red_tier_split=(.03, .45), red_flip=(0., 1.), red_jitter=(0., 1.2),
              red_keep=(.10, 1.), red_hold_share=(.02, 1.),
              red_move_push=(.02, .9), red_hold_push=(.005, .5))
SIGMA = dict(red_gain_left=.6, red_gain_right=.6, red_gain_up=.6, red_gain_down=.6,
             red_tier_split=.05, red_flip=.15, red_jitter=.15, red_keep=.08,
             red_hold_share=.10, red_move_push=.10, red_hold_push=.04)
NOISE_GENES = ("red_gain_left", "red_gain_right", "red_gain_up", "red_gain_down")


def default_genome(seed=0):
    genome = dict(DEFAULTS)
    genome["red_seed"] = int(seed)
    return genome


def clamp(name, value):
    low, high = BOUNDS[name]
    return float(min(high, max(low, value)))


def mutate(genome, rng, genes=6, sigma=1.):
    """几个数字动一下；动到哪个由 rng 决定。seed 变了就是换一张全新的接线。"""
    out = dict(genome)
    out["red_seed"] = int(genome.get("red_seed", 0))
    names = [name for name in DEFAULTS if name != "red_seed"]
    for name in rng.choice(names, size=min(int(genes), len(names)), replace=False):
        step = SIGMA[name]*float(sigma)
        if name in NOISE_GENES:
            out[name] = clamp(name, float(genome[name])*float(np.exp(rng.normal(0., step))))
        else:
            out[name] = clamp(name, float(genome[name]) + float(rng.normal(0., step)))
    if rng.random() < .25:
        out["red_seed"] = int(rng.integers(0, 2**31 - 1))
    return {name: (int(value) if name == "red_seed" else float(value))
            for name, value in out.items()}


def read_jsonl(path):
    rows = []
    if path is not None and Path(path).exists():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append(row, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_table(path=TABLE):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def wires(table, genome):
    """按位置野表 + 这份基因，把每个红细胞该接到哪一个方向、哪一档、多粗，算出来。"""
    places = np.asarray(table["places"], dtype=float)
    layer = table["cells"]["retinal_opponent"]
    best = np.asarray(layer["best_place"], dtype=int)
    peak = np.asarray(layer["peak"], dtype=float)
    if "red_peak" in layer:  # 兼容以后可能加的一列
        peak = np.asarray(layer["red_peak"], dtype=float)
    red = np.arange(len(best))[np.arange(len(best)) % 3 == 0]     # 红色通道：每 3 个取 1 个
    red = red[peak[red] > .15]                                    # 球从没扫亮过的不接
    angles = places[best[red]]
    rng = np.random.default_rng(int(genome["red_seed"]))
    kept = rng.random(len(red)) < float(genome["red_keep"])
    jitter = rng.normal(0., float(genome["red_jitter"]), size=len(red)) \
        if genome["red_jitter"] > 0 else np.zeros(len(red))
    flip = rng.random(len(red))
    out = []
    for index, (cell, (bearing, elevation)) in enumerate(zip(red, angles)):
        if not kept[index]:
            continue
        noise = float(np.exp(jitter[index]))
        for axis, angle in (("yaw", bearing), ("pitch", elevation)):
            if abs(angle) < DEAD:
                continue
            direction = (("left" if angle > 0 else "right") if axis == "yaw"
                         else ("up" if angle > 0 else "down"))
            tier = "move" if abs(angle) > float(genome["red_tier_split"]) else "hold"
            if flip[index] < float(genome["red_flip"]):
                tier = "hold" if tier == "move" else "move"
            weight = float(genome["red_gain_" + direction]) * min(1., abs(angle)/FULL[axis])*noise
            out.append([int(cell), direction, tier, weight])
    # 两档各自按总增益归一：一档里细胞多少不改变总力度，只有球的位置决定走哪一档。
    for tier, share in (("move", 1.), ("hold", float(genome["red_hold_share"]))):
        for direction in ("left", "right", "up", "down"):
            mine = [edge for edge in out if edge[1] == direction and edge[2] == tier]
            total = sum(edge[3] for edge in mine)
            if total <= 0:
                continue
            for edge in mine:
                edge[3] = edge[3]/total*float(genome["red_gain_" + direction])*share
    return out, len(red)


def spec_of(table, genome):
    edges, wired = wires(table, genome)
    return dict(edges=[tuple(edge) for edge in edges], wired=int(wired),
                kept=len(edges),
                move_push=float(genome["red_move_push"]),
                hold_push=float(genome["red_hold_push"]))


def build(ctx, spec, ball_red=True, distance=.90, size=.06):
    import mujoco
    body = tb.clean_body(ctx["model_path"])
    target = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_GEOM, "green_target")
    body.model.geom_size[target] = [size]*3
    if ball_red:
        body.model.geom_rgba[target] = [1., 0., 0., 1.]
    parameters = dict(ctx["parameters"])
    for name in MUTED:
        if name in parameters:
            parameters[name] = SILENT
    brain = controller(body, parameters, ctx, spec)
    eyes = RawEyes(body, width=brain.eye_width, height=brain.eye_height)
    return body, brain, eyes, target


def controller(body, parameters, ctx, spec):
    return EmbodiedController(body.home_angles, body.lower_limits, body.upper_limits,
                              seed=ctx["seed"],
                              eye_limits=(body.eye_lower_limits, body.eye_upper_limits),
                              eye_red=spec, **parameters)


def places_of(spread=.30, lift=.20):
    return [(b, e) for e in (-lift, 0., lift) for b in (-spread, 0., spread)]


def evaluate(brain, body, eyes, target, aim, places, steps, tail):
    """把球摆在每个位置上，看眼睛最终朝哪儿。返回每个位置的误差和成绩。"""
    environment = tb.blank_environment()
    observation = body.observe()
    rows = []
    for bearing, elevation in places:
        true_bearing, true_elevation = aim(bearing, elevation)
        yaws, pitches = [], []
        for _ in range(int(steps)):
            activation = brain.step(observation, environment=environment,
                                    eye_pixels=eyes.observe_raw(), dt=DT, learn=False)[1]
            body.command_eyes(brain.eye_command())
            observation = body.step(np.asarray(body.home_angles), duration=DT,
                                    activation=activation)
            command = brain.eye_command()
            yaws.append(.5*(float(command[0]) + float(command[2])) - true_bearing)
            pitches.append(.5*(float(command[1]) + float(command[3])) + true_elevation)
        rows.append(dict(bearing=round(true_bearing, 4), elevation=round(true_elevation, 4),
                         yaw_error=round(float(np.mean(yaws[-int(tail):])), 4),
                         pitch_error=round(float(np.mean(pitches[-int(tail):])), 4)))
    for row in rows:
        row["error"] = round(max(abs(row["yaw_error"]), abs(row["pitch_error"])), 4)
    return rows


def run_one(job):
    import mujoco
    started = time.perf_counter()
    table = load_table(job["table"])
    ctx = tb.context({}, job["seed"])
    spec = spec_of(table, job["genome"])
    row = dict(kind="eye", gen=job["gen"], index=job["index"], seed=job["seed"],
               genome=job["genome"], revision=tb.revision(),
               edges=len(spec["edges"]), wired=spec["wired"])
    try:
        body, brain, eyes, target = build(ctx, spec, ball_red=job["red_ball"])
        base = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_BODY, "base")

        def aim(bearing, elevation):
            x = job["distance"]*np.cos(elevation)*np.cos(bearing)
            y = job["distance"]*np.cos(elevation)*np.sin(bearing)
            z = .32 + job["distance"]*np.sin(elevation)
            body.model.geom_pos[target] = [x, y, z]
            mujoco.mj_forward(body.model, body.data)
            offset = np.asarray(body.data.geom_xpos[target]) - np.asarray(body.data.xpos[base])
            rotation = np.asarray(body.data.xmat[base]).reshape(3, 3)
            return (float(np.arctan2(offset @ rotation[:, 1], offset @ rotation[:, 0])),
                    float(np.arctan2(offset @ rotation[:, 2],
                                     np.linalg.norm(offset @ rotation[:, :2]))))
        rows = evaluate(brain, body, eyes, target, aim, job["places"],
                        job["steps"], job["tail"])
        eyes.close()
    except Exception as exc:
        row.update(status="refused", error="%s: %s" % (type(exc).__name__, exc),
                   wall_seconds=time.perf_counter() - started)
        return row
    errors = np.array([entry["error"] for entry in rows])
    row.update(status="ok", error=None, places=rows,
               accuracy=float((errors <= job["tolerance"]).mean()),
               mean_error=float(errors.mean()), worst_error=float(errors.max()),
               wall_seconds=time.perf_counter() - started)
    return row


def rank(row):
    return (-float(row["accuracy"]), float(row["mean_error"]))


def pick_parents(rows, keep):
    ok = [row for row in rows if row.get("status") == "ok"]
    ok.sort(key=rank)
    seen, out = set(), []
    for row in ok:
        mark = json.dumps(row["genome"], sort_keys=True)
        if mark in seen:
            continue
        seen.add(mark)
        out.append(row)
        if len(out) >= keep:
            break
    return out


def make_child(parents, rng, args):
    if len(parents) >= 2 and rng.random() < args.cross_rate:
        first, second = (parents[int(v)] for v in rng.choice(len(parents), size=2, replace=False))
        genome = {}
        for name in DEFAULTS:
            if name == "red_seed":
                genome[name] = int(first["genome"].get(name, 0))
                continue
            mine = float(first["genome"].get(name, DEFAULTS[name]))
            theirs = float(second["genome"].get(name, DEFAULTS[name]))
            roll = rng.random()
            genome[name] = mine if roll < .4 else (theirs if roll < .8 else .5*(mine + theirs))
        return mutate(genome, rng, genes=args.genes, sigma=args.sigma)
    return mutate(dict(parents[int(rng.integers(len(parents)))]["genome"]), rng,
                  genes=args.genes, sigma=args.sigma)


def generation_line(gen, rows, seconds):
    ok = [row for row in rows if row.get("status") == "ok"]
    if not ok:
        return "第 %d 代：%d 只一只都没建出来" % (gen, len(rows))
    best = min(ok, key=rank)
    hit = sorted(row["accuracy"] for row in ok)
    return ("第 %d 代收工：%d 只里建出来 %d 只；命中率中位 %.2f、最好 %.2f；"
            "平均误差中位 %.3f、最好 %.3f 弧度；本代花了 %.1f 分钟"
            % (gen, len(rows), len(ok), hit[len(hit)//2], hit[-1],
               float(np.median([row["mean_error"] for row in ok])), best["mean_error"],
               seconds/60.))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--parents", default=None, help="头一代的爹从这份台账里挑")
    ap.add_argument("--candidates", type=int, default=200)
    ap.add_argument("--generations", type=int, default=2)
    ap.add_argument("--keep", type=int, default=10)
    ap.add_argument("--genes", type=int, default=6)
    ap.add_argument("--sigma", type=float, default=1.)
    ap.add_argument("--cross-rate", type=float, default=.5)
    ap.add_argument("--steps", type=int, default=50, help="每个位置跑多少步（0.01 秒一步）")
    ap.add_argument("--tail", type=int, default=15, help="读最后多少步的平均")
    ap.add_argument("--tolerance", type=float, default=.10, help="命中容差（弧度）")
    ap.add_argument("--distance", type=float, default=.90)
    ap.add_argument("--spread", type=float, default=.30, help="左右最远多少个弧度")
    ap.add_argument("--lift", type=float, default=.20, help="上下最远多少个弧度")
    ap.add_argument("--red-ball", dest="red_ball", action="store_true", default=True)
    ap.add_argument("--green-ball", dest="red_ball", action="store_false",
                    help="对照：球是绿的，红色那一路不应该有反应")
    ap.add_argument("--table", default=str(TABLE))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--start-gen", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-prefix", default=str(ROOT / "artifacts" / "眼睛_红球注视"))
    args = ap.parse_args(argv)

    places = places_of(args.spread, args.lift)
    rng = np.random.default_rng(args.seed)
    if args.parents:
        parents = pick_parents(read_jsonl(args.parents), args.keep)
    else:
        parents = [dict(genome=default_genome(int(rng.integers(0, 2**31 - 1))))]
    if not parents:
        print("这份台账里挑不出爹：%s" % args.parents)
        return 1
    print("位置 %d 个：%s" % (len(places), "、".join("(%+.2f,%+.2f)" % p for p in places)), flush=True)
    print("头一代的爹 %d 只：%s" % (len(parents), "、".join(
        "命中 %.2f 误差 %.3f" % (row.get("accuracy", float("nan")), row.get("mean_error", float("nan")))
        for row in parents)), flush=True)

    champion = None
    with ProcessPoolExecutor(args.workers) as pool:
        for gen in range(args.start_gen, args.start_gen + args.generations):
            jobs = []
            for index in range(args.candidates):
                genome = make_child(parents, rng, args)
                jobs.append(dict(gen=gen, index=index, seed=int(rng.integers(1000000)),
                                 genome=genome, places=places, steps=args.steps, tail=args.tail,
                                 tolerance=args.tolerance, distance=args.distance,
                                 red_ball=args.red_ball, table=args.table))
            ledger = "%s_g%02d.jsonl" % (args.out_prefix, gen)
            print("第 %d 代：生 %d 只、每只 %d 个位置 x %d 步、%d 个进程"
                  % (gen, len(jobs), len(places), args.steps, args.workers), flush=True)
            started = time.perf_counter()
            rows = []
            for row in pool.map(run_one, jobs):
                rows.append(row)
                append(row, ledger)
                if row["status"] == "ok" and (champion is None or rank(row) < rank(champion)):
                    champion = row
                    print("  跨代冠军易主：%d 代第 %d 只  命中 %.2f  平均误差 %.3f 弧度  最差 %.3f"
                          % (row["gen"], row["index"], row["accuracy"], row["mean_error"],
                             row["worst_error"]), flush=True)
            print(generation_line(gen, rows, time.perf_counter() - started), flush=True)
            parents = pick_parents(rows, args.keep)
            keep = "%s_g%02d_keep.jsonl" % (args.out_prefix, gen)
            Path(keep).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                          for row in parents),
                                  encoding="utf-8", newline="\n")
            print("  下一代的爹 %d 只（%s）：%s" % (len(parents), keep, "、".join(
                "%.2f/%.3f" % (row["accuracy"], row["mean_error"]) for row in parents)), flush=True)
            if champion is not None:
                Path("%s_冠军.jsonl" % args.out_prefix).write_text(
                    json.dumps(champion, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    if champion is not None:
        print("跑完。跨代冠军：%d 代第 %d 只，命中 %.2f，平均误差 %.3f 弧度"
              % (champion["gen"], champion["index"], champion["accuracy"],
                 champion["mean_error"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())