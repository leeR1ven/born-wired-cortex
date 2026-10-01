from PIL import Image
src = r"F:\born-wired-cortex\engine_v2\artifacts\眼睛追球.gif"
dst = r"F:\born-wired-cortex\engine_v2\artifacts\眼睛追球_小.gif"
g = Image.open(src)
keep = list(range(0, g.n_frames, 2))
frames = []
for i in keep:
    g.seek(i)
    frames.append(g.convert("RGB").quantize(colors=64, method=Image.MEDIANCUT, dither=Image.NONE))
frames[0].save(dst, save_all=True, append_images=frames[1:], duration=100, loop=0, optimize=True)
import os
print("%s  %d 帧  %.1f MB" % (dst, len(frames), os.path.getsize(dst)/1e6))