import io, os
src = r"F:\born-wired-cortex\engine_v2\_smoke_raw.log"
out = r"F:\born-wired-cortex\engine_v2\_smoke_utf8.log"
raw = io.open(src, "rb").read()
for enc in ("utf-8", "gbk"):
    try:
        text = raw.decode(enc)
        break
    except UnicodeDecodeError:
        continue
io.open(out, "w", encoding="utf-8", newline="\n").write(text)
print("wrote", out, "as", enc)