import io
p = r"F:\born-wired-cortex\engine_v2\_patch_chase3.py"
s = io.open(p, encoding="utf-8", newline="").read()
print("CRLF" if "\r\n" in s else "LF only")
old = "    hits = [i for i, line in enumerate(lines) if line == marker]\n    hits = [i for i in hits if i >= start]\n"
new = ("    hits = [i for i, line in enumerate(lines) if line == marker]\n"
       "    if not hits:\n"
       "        hits = [i for i, line in enumerate(lines) if line.startswith(marker)]\n"
       "    hits = [i for i in hits if i >= start]\n")
assert old in s, "no find block"
s = s.replace(old, new, 1)
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("patched find")