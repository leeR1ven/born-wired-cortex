# -*- coding: utf-8 -*-
from pathlib import Path

p = Path("tools/wire_red_gaze.py")
t = p.read_text(encoding="utf-8", newline="")

old_measure = '''def measure(brain, body, eyes, aim, places, steps, tail):
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
        rows.append(dict(bearing=true_bearing, elevation=true_elevation,
                         yaw_error=float(np.mean(yaws[-int(tail):])),
                         pitch_error=float(np.mean(pitches[-int(tail):])),
                         yaw_wander=float(np.std(yaws[-int(tail):])),
                         pitch_wander=float(np.std(pitches[-int(tail):]))))
    for row in rows:
        row["error"] = max(abs(row["yaw_error"]), abs(row["pitch_error"]))
        row["wander"] = max(row["yaw_wander"], row["pitch_wander"])
    return rows
'''

new_measure = '''def required(body, target):
    """每只眼各自该转到哪儿才算正对球（肌肉的记法：+yaw 往左、+pitch 往下）。"""
    _, rotation = head_frame(body)
    ball = np.asarray(body.data.geom_xpos[target], dtype=float)
    out = []
    for position in eye_world(body):
        v = rotation.T @ (ball - position)
        out.append((float(np.arctan2(v[1], v[0])),
                    -float(np.arctan2(v[2], np.hypot(v[0], v[1])))))
    return out


def measure(brain, body, eyes, aim, target, places, steps, tail):
    """每只眼对着自己的目标量，不用两眼平均。

    球摆在一边时，离得远的那只眼本来就该看到更大的偏角 —— 更远的那些位置它干脆看不见
    （相机上下 60 度，半角 0.52 弧度）。所以「两眼命令的平均 vs 头坐标系方位」这把旧尺子
    在这里量不出东西，它把歪的那只眼摊平到平均值里。看不见球的那只眼不判。
    """
    environment = tb.blank_environment()
    observation = body.observe()
    rows = []
    for bearing, elevation in places:
        true_bearing, true_elevation = aim(bearing, elevation)
        ask = required(body, target)
        asked = [abs(a[0]) <= VIEW and abs(a[1]) <= VIEW for a in ask]
        yaws, pitches = [], []
        for _ in range(int(steps)):
            activation = brain.step(observation, environment=environment,
                                    eye_pixels=eyes.observe_raw(), dt=DT, learn=False)[1]
            body.command_eyes(brain.eye_command())
            observation = body.step(np.asarray(body.home_angles), duration=DT,
                                    activation=activation)
            command = brain.eye_command()
            yaws.append([float(command[0]) - ask[0][0], float(command[2]) - ask[1][0]])
            pitches.append([float(command[1]) - ask[0][1], float(command[3]) - ask[1][1]])
        closes, trims = yaws[-int(tail):], pitches[-int(tail):]
        rows.append(dict(bearing=true_bearing, elevation=true_elevation, asked=asked,
                         yaw_error=[float(np.mean([one[k] for one in closes])) for k in (0, 1)],
                         pitch_error=[float(np.mean([one[k] for one in trims])) for k in (0, 1)],
                         yaw_wander=[float(np.std([one[k] for one in closes])) for k in (0, 1)],
                         pitch_wander=[float(np.std([one[k] for one in trims])) for k in (0, 1)]))
    for row in rows:
        row["error"] = max([max(abs(row["yaw_error"][k]), abs(row["pitch_error"][k]))
                            for k in (0, 1) if row["asked"][k]] or [0.])
        row["wander"] = max([max(row["yaw_wander"][k], row["pitch_wander"][k])
                             for k in (0, 1) if row["asked"][k]] or [0.])
    return rows
'''

old_call = """    rows = measure(brain, body, eyes, aim, places, args.steps, args.tail)
    eyes.close()
    print("\\n%8s %8s | %9s %9s | %8s %8s" % ("球方位", "球高低", "yaw 误差", "pitch 误差",
                                             "最大", "后段抖动"))
    for row in rows:
        print("%8.3f %8.3f | %+9.3f %+9.3f | %+8.3f %5.3f/%5.3f"
              % (row["bearing"], row["elevation"], row["yaw_error"], row["pitch_error"],
                 row["error"], row["yaw_wander"], row["pitch_wander"]))
    hits = [row for row in rows if row["error"] <= args.tolerance]
    mid = [row for row in rows
           if max(abs(row["bearing"]), abs(row["elevation"])) <= args.tolerance]
    print("\\n命中 %d/%d = %.2f（其中球本来就摆在正中的 %d 个，不动眼睛也算命中）"
          % (len(hits), len(rows), len(hits)/float(len(rows)), len(mid)))
    print("误差：中位 %.3f、平均 %.3f、最差 %.3f 弧度（容差 %.2f）"
          % (float(np.median([row["error"] for row in rows])),
             float(np.mean([row["error"] for row in rows])),
             float(np.max([row["error"] for row in rows])), args.tolerance))
    print("后段抖动：yaw 最大 %.4f、pitch 最大 %.4f 弧度（抖得大 = 眼睛在来回撞，不是稳在一个角度）"
          % (float(np.max([row["yaw_wander"] for row in rows])),
             float(np.max([row["pitch_wander"] for row in rows]))))
"""

new_call = """    rows = measure(brain, body, eyes, aim, target, places, args.steps, args.tail)
    eyes.close()
    print("\\n%8s %8s | %13s | %13s | %8s" % ("球方位", "球高低", "左眼 yaw/pitch",
                                              "右眼 yaw/pitch", "后段抖动"))
    for row in rows:
        print("%8.3f %8.3f | %+5.3f/%+5.3f %s | %+5.3f/%+5.3f %s | %5.3f"
              % (row["bearing"], row["elevation"],
                 row["yaw_error"][0], row["pitch_error"][0],
                 "看" if row["asked"][0] else "盲",
                 row["yaw_error"][1], row["pitch_error"][1],
                 "看" if row["asked"][1] else "盲", row["wander"]))
    hits = [row for row in rows if row["error"] <= args.tolerance]
    errors = [max(abs(row["yaw_error"][k]), abs(row["pitch_error"][k]))
              for row in rows for k in (0, 1) if row["asked"][k]]
    print("\\n命中 %d/%d = %.2f（每只眼对着自己的目标算；标「盲」的那只眼看不见球，不判它）"
          % (len(hits), len(rows), len(hits)/float(len(rows))))
    print("误差：中位 %.3f、平均 %.3f、最差 %.3f 弧度（容差 %.2f）"
          % (float(np.median(errors)), float(np.mean(errors)), float(np.max(errors)),
             args.tolerance))
    print("后段抖动：最大 %.4f 弧度（抖得大 = 眼睛在来回撞，不是稳在一个角度）"
          % float(np.max([row["wander"] for row in rows])))
"""

old_out = """    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            dict(options=vars(args), edges=len(spec["edges"]), places=rows,
                 accuracy=float(len(hits)/float(len(rows))),
                 mean_error=float(np.mean([row["error"] for row in rows])),
                 worst_error=float(np.max([row["error"] for row in rows]))),"""
new_out = """    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            dict(options=vars(args), edges=len(spec["edges"]), places=rows,
                 accuracy=float(len(hits)/float(len(rows))),
                 mean_error=float(np.mean(errors)), worst_error=float(np.max(errors))),"""

old_dt = "DT = .01\n"
new_dt = "DT = .01\n# 相机上下 60 度，半角 0.52 弧度：偏得比这还多的球，这只眼根本看不见。\nVIEW = .52\n"

for old, new in ((old_dt, new_dt), (old_measure, new_measure), (old_call, new_call), (old_out, new_out)):
    assert old in t, "没找到：%r" % old[:70]
    assert t.count(old) == 1, "出现多次：%r" % old[:70]
    t = t.replace(old, new)
p.write_text(t, encoding="utf-8", newline="")
print("wire_red_gaze.py 改完")