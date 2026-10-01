import io
p = r"F:\born-wired-cortex\engine_v2\tools\evolve_chase.py"
s = io.open(p, encoding="utf-8", newline="").read()
old = '''               shifted_m=max(item["shifted_m"] for item in got["rows"]
                             if item["shifted_m"] == item["shifted_m"]),'''
new = '''               shifted_m=max([item["shifted_m"] for item in got["rows"]
                              if item["shifted_m"] == item["shifted_m"]] or [float("nan")]),'''
assert old in s
io.open(p, "w", encoding="utf-8", newline="").write(s.replace(old, new, 1))
print("patched")