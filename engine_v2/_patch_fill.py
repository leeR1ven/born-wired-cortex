# -*- coding: utf-8 -*-
"""补缺口：原来的位置野表哪里是 nan（左眼最左 2 列、右眼最右 2 列），就用宽表的值补上；
其余一切照旧。这样「线接全了」，但原来的标定不被挪动。"""
import io

p = "tools/wire_red_gaze.py"
text = io.open(p, encoding="utf-8", newline="").read()
edits = [
    ('TABLE = ROOT / "artifacts" / "红球位置野_红.json"\n',
     'TABLE = ROOT / "artifacts" / "红球位置野_红.json"\n'
     '# 补缺口用的表：扫得更宽（方位 ±0.62），把每只眼画面最边上那两列也标定到了。\n'
     '# 只在主表是空的地方用它 —— 主表其他列的标定一个字都不动。\n'
     'WIDE = ROOT / "artifacts" / "红球位置野_红_宽.json"\n'),

    ('''def red_cells(table, floor=.15):
    """红色通道里值得接线的细胞。"""
    peak = np.asarray(table["cells"]["retinal_opponent"]["peak"], dtype=float)
    index = np.arange(len(peak))
    red = index[index % 3 == 0]                      # 红色通道：每 3 个取 1 个
    return red[peak[red] > floor]                    # 球从没扫亮过的不接''',
     '''def red_cells(table, floor=.15, fill=None):
    """红色通道里值得接线的细胞。fill 那张表里扫亮过的，也算上。"""
    peak = np.asarray(table["cells"]["retinal_opponent"]["peak"], dtype=float)
    index = np.arange(len(peak))
    red = index[index % 3 == 0]                      # 红色通道：每 3 个取 1 个
    keep = peak[red] > floor                         # 球从没扫亮过的不接
    if fill is not None and fill is not table:
        more = np.asarray(fill["cells"]["retinal_opponent"]["peak"], dtype=float)
        if more.shape == peak.shape:
            keep = keep | (more[red] > floor)
    return red[keep]'''),

    ('def retina_angles(table, floor=.15):\n',
     'def retina_angles(table, floor=.15, fill=None):\n'),

    ('''        for number in range(rows):
            taken = seen[row[mine] == number, 1]
            if len(taken):
                pitch[which, number] = float(np.median(taken))
    return yaw, pitch''',
     '''        for number in range(rows):
            taken = seen[row[mine] == number, 1]
            if len(taken):
                pitch[which, number] = float(np.median(taken))
    if fill is not None and fill is not table:
        # 主表没标定到的列（左眼最左、右眼最右）才问补表。
        other_yaw, other_pitch = retina_angles(fill, floor)
        yaw = np.where(np.isfinite(yaw), yaw, other_yaw)
        pitch = np.where(np.isfinite(pitch), pitch, other_pitch)
    return yaw, pitch'''),

    ('''def cell_angles(table, index, group, per_eye=True, floor=.15):''',
     '''def cell_angles(table, index, group, per_eye=True, floor=.15, fill=None):'''),

    ('''    yaw, pitch = retina_angles(table, floor)
    row, column = cell_rows_columns(index, table)
    return np.stack([yaw[group, column], pitch[group, row]], axis=1)''',
     '''    yaw, pitch = retina_angles(table, floor, fill)
    row, column = cell_rows_columns(index, table)
    return np.stack([yaw[group, column], pitch[group, row]], axis=1)'''),

    ('''          block=None, per_eye=True, only=None):
    """算每一根线：从哪个细胞出发、接到哪个方向、哪一档、多粗。"""
    block = dict(BLOCK if block is None else block)
    hold_top = float(split if hold_top is None else hold_top)
    red = red_cells(table)
    group = eye_of(red, table)
    sight = cell_angles(table, red, group, per_eye)''',
     '''          block=None, per_eye=True, only=None, fill=None):
    """算每一根线：从哪个细胞出发、接到哪个方向、哪一档、多粗。"""
    block = dict(BLOCK if block is None else block)
    hold_top = float(split if hold_top is None else hold_top)
    red = red_cells(table, fill=fill)
    group = eye_of(red, table)
    sight = cell_angles(table, red, group, per_eye, fill=fill)'''),

    ('''    yaw_map, pitch_map = retina_angles(table) if per_eye else (None, None)''',
     '''    yaw_map, pitch_map = retina_angles(table, fill=fill) if per_eye else (None, None)'''),

    ('''    direct = bool(kwargs.pop("direct", True))''',
     '''    direct = bool(kwargs.pop("direct", True))
    fill = kwargs.pop("fill", None)'''),

    ('''                  dead=dead, per_eye=per_eye, only=only, gain=gain, **kwargs)''',
     '''                  dead=dead, per_eye=per_eye, only=only, gain=gain, fill=fill, **kwargs)'''),

    ('''    ap.add_argument("--table", default=str(TABLE))''',
     '''    ap.add_argument("--table", default=str(TABLE))
    ap.add_argument("--fill", default=str(WIDE),
                    help="补缺口用的表：主表没标定到的列（左眼最左、右眼最右各 2 列）用它补上")'''),

    ('''                   rest_speed=args.rest_speed)''',
     '''                   rest_speed=args.rest_speed,
                   fill=None if args.fill in ("", "none") else load_table(args.fill))'''),
]
for old, new in edits:
    assert text.count(old) == 1, old[:70]
    text = text.replace(old, new)
io.open(p, "w", encoding="utf-8", newline="").write(text)
print("补缺口改好")