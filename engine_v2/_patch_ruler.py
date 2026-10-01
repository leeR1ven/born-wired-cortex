# -*- coding: utf-8 -*-
import io, sys
p = r"F:\born-wired-cortex\engine_v2\tools\wire_red_gaze.py"
s = io.open(p, encoding="utf-8", newline="").read()

old = "DT = .01\n# \u76f8\u673a\u4e0a\u4e0b 60 \u5ea6\uff0c\u534a\u89d2 0.52 \u5f27\u5ea6\uff1a\u504f\u5f97\u6bd4\u8fd9\u8fd8\u591a\u7684\u7403\uff0c\u8fd9\u53ea\u773c\u6839\u672c\u770b\u4e0d\u89c1\u3002\nVIEW = .52\n"
new = "DT = .01\n"
assert s.count(old) == 1, "A"
s = s.replace(old, new)

old = '''def mark(row, which):
    """\u8fd9\u4e00\u683c\u8fd9\u53ea\u773c\u7b97\u4e0d\u7b97\u6570\uff1a\u770b/\u76f2\uff08\u770b\u4e0d\u89c1\u7403\uff09/\u4e0d\u7ba1\uff08\u8fd9\u4e00\u8f6e\u53ea\u505a\u53e6\u4e00\u53ea\u773c\uff09\u3002"""
    if row.get("only") is not None and row["only"] != which:
        return "\u4e0d\u7ba1"
    return "\u770b" if row["asked"][which] else "\u76f2"
'''
new = '''def view_half_angles(body):
    """\u6bcf\u53ea\u773c\u7684\u753b\u9762\u6709\u591a\u5bbd\uff08\u5f27\u5ea6\uff09\uff0c\u4ece\u6a21\u578b\u91cc\u7684\u76f8\u673a\u8bfb\uff1a\u7ad6\u76f4\u534a\u89d2 fovy/2\uff0c\u6c34\u5e73\u534a\u89d2
    atan(tan(\u7ad6\u76f4\u534a\u89d2) * \u5bbd/\u9ad8)\u3002

    2026-09-30 \u4e4b\u524d\u8fd9\u91cc\u5199\u6b7b 0.52\uff08\u7ad6\u76f4\u534a\u89d2\uff09\u53bb\u5361\u4e24\u4e2a\u65b9\u5411\uff0c\u4e8e\u662f 0.9 \u7c73\u5916\u504f 0.35 \u5f27\u5ea6\u7684
    \u7403\u88ab\u8bb0\u6210\u300c\u770b\u4e0d\u89c1\u300d\u2014\u2014\u5176\u5b9e\u5b83\u5728\u6c34\u5e73\u65b9\u5411\uff08\u534a\u89d2 0.656 \u5f27\u5ea6\uff09\u91cc\uff0c\u53ea\u662f\u90a3\u53ea\u773c\u8f6c\u4e0d\u8fc7\u53bb\u3002
    """
    camera = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_CAMERA, "eye_left")
    half = np.radians(float(body.model.cam_fovy[camera]))/2.
    wide = float(np.arctan(np.tan(half)*float(body.eye_width)/float(body.eye_height)))
    return wide, half


def eye_reach(body, which):
    """\u8fd9\u53ea\u773c\u7684\u4e24\u6839\u8f74\u5404\u80fd\u8f6c\u5230\u54ea\u513f\uff08\u5f27\u5ea6\uff0c\u4e0b\u9650 \u4e0a\u9650\uff09\u3002"""
    return (float(body.eye_lower_limits[2*which]), float(body.eye_upper_limits[2*which]),
            float(body.eye_lower_limits[2*which + 1]), float(body.eye_upper_limits[2*which + 1]))


def mark(row, which):
    """\u8fd9\u4e00\u683c\u8fd9\u53ea\u773c\u7b97\u4e0d\u7b97\u6570\uff1a\u770b / \u76f2\uff08\u7403\u4e0d\u5728\u5b83\u753b\u9762\u91cc\uff09/ \u591f\u4e0d\u7740\uff08\u8f6c\u8fc7\u53bb\u4e5f\u5230\u4e0d\u4e86\uff09/ \u4e0d\u7ba1\u3002"""
    if row.get("only") is not None and row["only"] != which:
        return "\u4e0d\u7ba1"
    if not row["visible"][which]:
        return "\u76f2"
    if not row["reachable"][which]:
        return "\u591f\u4e0d\u7740"
    return "\u770b"
'''
assert s.count(old) == 1, "B"
s = s.replace(old, new)

old = '''def measure(brain, body, eyes, aim, target, places, steps, tail, only=None):
    """\u6bcf\u53ea\u773c\u5bf9\u7740\u81ea\u5df1\u7684\u76ee\u6807\u91cf\uff0c\u4e0d\u7528\u4e24\u773c\u5e73\u5747\u3002

    \u7403\u6446\u5728\u4e00\u8fb9\u65f6\uff0c\u79bb\u5f97\u8fdc\u7684\u90a3\u53ea\u773c\u672c\u6765\u5c31\u8be5\u770b\u5230\u66f4\u5927\u7684\u504f\u89d2 \u2014\u2014 \u66f4\u8fdc\u7684\u90a3\u4e9b\u4f4d\u7f6e\u5b83\u5e72\u8106\u770b\u4e0d\u89c1
    \uff08\u76f8\u673a\u4e0a\u4e0b 60 \u5ea6\uff0c\u534a\u89d2 0.52 \u5f27\u5ea6\uff09\u3002\u6240\u4ee5\u300c\u4e24\u773c\u547d\u4ee4\u7684\u5e73\u5747 vs \u5934\u5750\u6807\u7cfb\u65b9\u4f4d\u300d\u8fd9\u628a\u65e7\u5c3a\u5b50
    \u5728\u8fd9\u91cc\u91cf\u4e0d\u51fa\u4e1c\u897f\uff0c\u5b83\u628a\u6b6a\u7684\u90a3\u53ea\u773c\u644a\u5e73\u5230\u5e73\u5747\u503c\u91cc\u3002\u770b\u4e0d\u89c1\u7403\u7684\u90a3\u53ea\u773c\u4e0d\u5224\u3002
    """
    environment = tb.blank_environment()
    observation = body.observe()
    rows = []
    for bearing, elevation in places:
        true_bearing, true_elevation = aim(bearing, elevation)
        ask = required(body, target)
        asked = [abs(a[0]) <= VIEW and abs(a[1]) <= VIEW for a in ask]
        if only is not None:
            asked = [want and which == int(only) for which, want in enumerate(asked)]
'''
new = '''def measure(brain, body, eyes, aim, target, places, steps, tail, only=None, view=None):
    """\u6bcf\u53ea\u773c\u5bf9\u7740\u81ea\u5df1\u7684\u76ee\u6807\u91cf\uff0c\u4e0d\u7528\u4e24\u773c\u5e73\u5747\u3002

    \u7403\u6446\u5728\u4e00\u8fb9\u65f6\uff0c\u79bb\u5f97\u8fdc\u7684\u90a3\u53ea\u773c\u672c\u6765\u5c31\u8be5\u770b\u5230\u66f4\u5927\u7684\u504f\u89d2\u3002\u7403\u4e0d\u5728\u5b83\u753b\u9762\u91cc\uff08\u76f2\uff09\u3001\u6216\u8005\u8f6c\u5230\u5e95\u4e5f\u5230\u4e0d\u4e86\uff08\u591f\u4e0d\u7740\uff09
    \u7684\u90a3\u51e0\u4e2a\u4f4d\u7f6e\u4e0d\u5224\u5b83 \u2014\u2014 \u62ff\u4e00\u6746\u5e73\u5747\u5c3a\u5b50\u4f1a\u628a\u6b6a\u7684\u90a3\u53ea\u773c\u644a\u5e73\u5230\u5e73\u5747\u503c\u91cc\u3002
    """
    environment = tb.blank_environment()
    observation = body.observe()
    rows = []
    for bearing, elevation in places:
        true_bearing, true_elevation = aim(bearing, elevation)
        ask = required(body, target)
        visible = [True, True] if view is None else [abs(a[0]) <= view[0] and abs(a[1]) <= view[1]
                                                    for a in ask]
        reachable = []
        for which in (0, 1):
            low_y, high_y, low_p, high_p = eye_reach(body, which)
            reachable.append(low_y <= ask[which][0] <= high_y and low_p <= ask[which][1] <= high_p)
        asked = [see and can for see, can in zip(visible, reachable)]
        if only is not None:
            asked = [want and which == int(only) for which, want in enumerate(asked)]
'''
assert s.count(old) == 1, "C"
s = s.replace(old, new)

old = '''        rows.append(dict(bearing=true_bearing, elevation=true_elevation, asked=asked,
                         only=None if only is None else int(only),'''
new = '''        rows.append(dict(bearing=true_bearing, elevation=true_elevation, asked=asked,
                         visible=visible, reachable=reachable,
                         only=None if only is None else int(only),'''
assert s.count(old) == 1, "D"
s = s.replace(old, new)

old = '''    print("\u9759\u6b62\u65f6\u7ed9\u51fa\u7684\u89d2\u5ea6\uff08\u7403\u8fd8\u6ca1\u6446\uff09\uff1a%s" % np.round(brain.eye_command(), 4))
    rows = measure(brain, body, eyes, aim, target, places, args.steps, args.tail, only)'''
new = '''    view = view_half_angles(body)
    print("\u6bcf\u53ea\u773c\u7684\u753b\u9762\uff1a\u6c34\u5e73\u534a\u89d2 %.3f \u5f27\u5ea6\u3001\u7ad6\u76f4\u534a\u89d2 %.3f \u5f27\u5ea6\uff1b"
          "\u5de6\u773c\u884c\u7a0b yaw %+.2f~%+.2f\u3001pitch %+.2f~%+.2f"
          % (view[0], view[1]) + " \u5f27\u5ea6" % () if False else
          "\u6bcf\u53ea\u773c\u7684\u753b\u9762\uff1a\u6c34\u5e73\u534a\u89d2 %.3f\u3001\u7ad6\u76f4\u534a\u89d2 %.3f \u5f27\u5ea6\uff1b\u5de6\u773c\u884c\u7a0b %s"
          % (view[0], view[1], np.round(np.asarray(eye_reach(body, 0)), 2)))
    print("\u9759\u6b62\u65f6\u7ed9\u51fa\u7684\u89d2\u5ea6\uff08\u7403\u8fd8\u6ca1\u6446\uff09\uff1a%s" % np.round(brain.eye_command(), 4))
    rows = measure(brain, body, eyes, aim, target, places, args.steps, args.tail, only, view)'''
assert s.count(old) == 1, "E"
s = s.replace(old, new)

old = '''    hits = [row for row in rows if row["error"] <= args.tolerance]
    errors = [max(abs(row["yaw_error"][k]), abs(row["pitch_error"][k]))
              for row in rows for k in (0, 1) if row["asked"][k]]
    print("\\n\u547d\u4e2d %d/%d = %.2f\uff08\u6bcf\u53ea\u773c\u5bf9\u7740\u81ea\u5df1\u7684\u76ee\u6807\u7b97\uff1b\u6807\u300c\u76f2\u300d\u7684\u90a3\u53ea\u773c\u770b\u4e0d\u89c1\u7403\uff0c\u4e0d\u5224\u5b83\uff09"
          % (len(hits), len(rows), len(hits)/float(len(rows))))'''
new = '''    judged = [(row, k) for row in rows for k in (0, 1) if row["asked"][k]]
    hits = [(row, k) for row, k in judged
            if max(abs(row["yaw_error"][k]), abs(row["pitch_error"][k])) <= args.tolerance]
    errors = [max(abs(row["yaw_error"][k]), abs(row["pitch_error"][k])) for row, k in judged]
    print("\\n\u547d\u4e2d %d/%d = %.2f\uff08\u53ea\u7b97\u7403\u80fd\u770b\u89c1\u3001\u53c8\u8f6c\u5f97\u5230\u7684\u90a3\u51e0\u683c\uff09"
          % (len(hits), len(judged), len(hits)/float(max(1, len(judged)))))'''
assert s.count(old) == 1, "F"
s = s.replace(old, new)

old = '''            dict(options=vars(args), edges=len(spec["edges"]), places=rows,
                 accuracy=float(len(hits)/float(len(rows))),'''
new = '''            dict(options=vars(args), edges=len(spec["edges"]), places=rows,
                 accuracy=float(len(hits)/float(max(1, len(judged)))),'''
assert s.count(old) == 1, "G"
s = s.replace(old, new)

io.open(p, "w", encoding="utf-8", newline="").write(s)
print("ok")