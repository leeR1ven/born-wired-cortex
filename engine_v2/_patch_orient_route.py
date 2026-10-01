import io
p = r"F:\born-wired-cortex\engine_v2\born_wired\embodied.py"
s = io.open(p, encoding="utf-8", newline="").read()
CR = "\r\n"

start = "            turn_left = cells('chase_turn_left', 1, time=turn_time)[0]"
end = "                    edge(unit, cell, each)"
i = s.index(start)
j = s.index(end) + len(end)
block = s[i:j]
assert 20 < block.count("\n") < 40, block.count("\n")

indented = "\r\n".join(("    " + line) if line.strip() else line for line in block.split("\r\n"))
wrapped = ("            if make_turn_cells:" + CR + indented)

# 在 flip 之后插入 orient 那条路 + make_turn_cells 开关
anchor = ("            if chase.get('flip'):" + CR
          + "                units['left'], units['right'] = units['right'], units['left']" + CR)
assert anchor in s
branch = anchor + (
    "            # 2026-09-30 第二条路：把「眼睛偏在哪一边」直接报给模型本来那个" + CR
    + "            # 「转向、凑过去看」的细胞（看到绿色的东西时用的就是它）。狗拐弯的" + CR
    + "            # 力不是这里给的，是基因 avoidance_gain（steering→髋 的那根线）；" + CR
    + "            # 这里只决定「往哪边拐」。这条路上的细胞带疲劳，推一下自己会收，" + CR
    + "            # 所以不会像恒定拧髋那样把步态顶死。" + CR
    + "            route = str(chase.get('route', 'motor'))" + CR
    + "            make_turn_cells = route != 'orient'" + CR
    + "            if not make_turn_cells:" + CR
    + "                orient = self.groups['orienting']" + CR
    + "                weight = float(chase.get('gain', 1.))" + CR
    + "                for side, key in (('left', 0), ('right', 1)):" + CR
    + "                    for unit in units[side]:" + CR
    + "                        edge(unit, orient[key], weight)" + CR)
s = s.replace(anchor, branch, 1)
s = s.replace(block, wrapped, 1)
s = s.replace("            turn_gain = float(chase.get('turn_gain', .25))" + CR,
              "            turn_gain = float(chase.get('turn_gain', .25)) if make_turn_cells else 0." + CR, 1)
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("patched embodied.py")
