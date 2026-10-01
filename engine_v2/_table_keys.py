import json, io
t = json.load(io.open(r"F:\born-wired-cortex\engine_v2\artifacts\红球位置野_红.json", encoding="utf-8"))
print(list(t.keys()))
print(list(t["cells"].keys()))
c = t["cells"]["retinal_opponent"]
print({k: (len(v) if hasattr(v, "__len__") else v) for k, v in c.items()})
print("options:", t.get("options"))