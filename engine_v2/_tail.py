import io
for name in (r"F:\born-wired-cortex\engine_v2\_冒烟_追球_三批.log", r"F:\born-wired-cortex\engine_v2\_冒烟_追球_三批.err"):
    print("=== " + name)
    try:
        print(io.open(name, encoding="utf-8", errors="replace").read())
    except FileNotFoundError:
        print("(还没有)")