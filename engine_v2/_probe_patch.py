import io
s = io.open(r"born_wired\stereo_senses.py", encoding="utf-8").read().splitlines()
start = None
for i, line in enumerate(s):
    if "_patches" in line or "_sampling" in line:
        if start is None:
            start = i
for i in range(start-2, min(start+70, len(s))):
    print(i+1, s[i][:130])