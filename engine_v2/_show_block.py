import io
p = r"F:\born-wired-cortex\engine_v2\born_wired\embodied.py"
s = io.open(p, encoding="utf-8", newline="").read()
start = "            turn_left = cells('chase_turn_left', 1, time=turn_time)[0]"
end = "                    edge(unit, cell, each)"
i = s.index(start); j = s.index(end) + len(end)
print(repr(s[i:j]))
print("---- 之后 200 字 ----")
print(repr(s[j:j+200]))