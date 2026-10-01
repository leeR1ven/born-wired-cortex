# -*- coding: utf-8 -*-
from pathlib import Path
p = Path("tools/wire_red_gaze.py")
t = p.read_text(encoding="utf-8", newline="")

old_block = '''# 红球 0.06 米、0.9 米远，在视网膜上亮的是一片约 5.6 x 6.2 个细胞的方块（实测），
# 换成角度就是下面这两个半宽 —— 归一化要知道「球摆在那儿时亮多大一片」。
BLOCK = {"yaw": np.radians(3.3), "pitch": np.radians(3.0)}
'''
new_block = '''# 红球 0.06 米、0.9 米远，在视网膜上亮的是一片约 5.6 x 6.2 个细胞的方块（实测，
# tools/measure_red_ball_features.py）。下面两个是半宽，单位是**细胞**，不是角度 ——
# 视网膜中央密、边上疏，同样 ±3.3 度在中央圈到七列、在边上只圈到两三列，归一化的分母
# 于是差两三倍，两只眼、两个方向都不一致（2026-09-30 实测：同一套线，左眼只到目标的
# 79%、右眼只到 56%；把推力上限抬到 0.60 后左眼到位 21/25 但抖动涨到 0.21 弧度，
# 根源都是分母在边上算小了）。球的像有这么大一团是球自己的事，跟它落在视网膜哪儿无关，
# 所以归一化按细胞圈，不按角度圈。
BLOB = {"yaw": 2.8, "pitch": 3.1}
BLOCK = {"yaw": np.radians(3.3), "pitch": np.radians(3.0)}   # 不分眼的老做法按角度圈
'''
assert t.count(old_block) == 1
t = t.replace(old_block, new_block)

old_helper = '''def spec_of(table, split, move_push, hold_push, dead, **kwargs):'''
new_helper = '''def nearest(values, target):
    """这一列（这一行）里，答话角度最接近 target 的那一个编号。"""
    values = np.asarray(values, dtype=float)
    good = np.isfinite(values)
    if not good.any():
        return None
    numbers = np.arange(len(values))[good]
    return int(numbers[np.argmin(np.abs(values[good] - target))])


def spec_of(table, split, move_push, hold_push, dead, **kwargs):'''
assert t.count(old_helper) == 1
t = t.replace(old_helper, new_helper)

old_wires = '''    red = red_cells(table)
    group = eye_of(red, table)
    sight = cell_angles(table, red, group, per_eye)
    bearing, elevation = sight[:, 0], sight[:, 1]
    out = []
    for eye in ((0, 1) if per_eye else (None,)):
        mine = np.ones(len(red), bool) if eye is None else (group == eye)
'''
new_wires = '''    red = red_cells(table)
    group = eye_of(red, table)
    sight = cell_angles(table, red, group, per_eye)
    bearing, elevation = sight[:, 0], sight[:, 1]
    row, column = cell_rows_columns(red, table)
    yaw_map, pitch_map = retina_angles(table) if per_eye else (None, None)
    out = []
    for eye in ((0, 1) if per_eye else (None,)):
        mine = np.ones(len(red), bool) if eye is None else (group == eye)
'''
assert t.count(old_wires) == 1
t = t.replace(old_wires, new_wires)

old_blob = '''                # 球摆在「这一档最远处、正前方那个高度」时点亮的那片细胞。球在视网膜上
                # 亮的是一个团，不是一个竖列，所以两个方向都要卡：只按方位卡，会把所有
                # 高度的细胞都算进分母，接线就细了十倍。per_eye 时分母只算这一只眼的。
                blob = (mine & (np.abs(seen - band_top) <= block[axis])
                        & (np.abs(other) <= block[cross_axis]))
'''
new_blob = '''                # 球摆在「这一档最远处、正前方那个高度」时点亮的那片细胞。球在视网膜上
                # 亮的是一个团，不是一个竖列，所以两个方向都要卡：只按方位卡，会把所有
                # 高度的细胞都算进分母，接线就细了十倍。per_eye 时分母只算这一只眼的，
                # 而且按细胞圈（BLOB）：中央密、边上疏，按角度圈会让边缘的分母小两三倍。
                if per_eye:
                    centre = nearest(yaw_map[eye], band_top if axis == "yaw" else 0.)
                    row_here = nearest(pitch_map[eye], band_top if axis == "pitch" else 0.)
                    if centre is None or row_here is None:
                        continue
                    blob = (mine & (np.abs(column - centre) <= BLOB["yaw"])
                            & (np.abs(row - row_here) <= BLOB["pitch"]))
                else:
                    blob = (mine & (np.abs(seen - band_top) <= block[axis])
                            & (np.abs(other) <= block[cross_axis]))
'''
assert t.count(old_blob) == 1
t = t.replace(old_blob, new_blob)
p.write_text(t, encoding="utf-8", newline="")
print("归一化改成按细胞圈")