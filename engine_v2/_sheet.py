import sys
from pathlib import Path
from PIL import Image
g = Image.open(r"F:\born-wired-cortex\engine_v2\artifacts\眼睛追球.gif")
n = g.n_frames
print("frames", n, "size", g.size)
picks = [0, 30, 60, 90, 120, 150, 180, 210, 250]
ims = []
for i in picks:
    g.seek(i)
    ims.append(g.convert("RGB").copy())
w, h = ims[0].size
cols = 3
rows = 3
sheet = Image.new("RGB", (w*cols, h*rows), "white")
for k, im in enumerate(ims):
    sheet.paste(im, ((k%cols)*w, (k//cols)*h))
out = r"F:\born-wired-cortex\engine_v2\artifacts\_movie_sheet.png"
sheet.save(out)
print("saved", out, sheet.size)