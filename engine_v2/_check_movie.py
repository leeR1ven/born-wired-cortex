import numpy as np
from PIL import Image
g = Image.open(r"F:\born-wired-cortex\engine_v2\artifacts\眼睛追球.gif")
PW, PH = 240, 180
# 每只眼画面 240x180，水平视场 = 2*atan(tan(30deg)*4/3)
hfov = np.degrees(2*np.arctan(np.tan(np.radians(30.))*PW/PH))
deg_per_px = hfov/PW
print("水平视场 %.1f 度，每像素 %.3f 度" % (hfov, deg_per_px))
rows = []
for i in range(0, g.n_frames):
    g.seek(i)
    im = np.asarray(g.convert("RGB").crop((PW, 0, PW*3, PH))).astype(int)
    out = []
    for k in range(2):
        p = im[:, k*PW:(k+1)*PW]
        mask = (p[:,:,0] > 140) & (p[:,:,1] < 110) & (p[:,:,2] < 110)
        if mask.sum() < 5:
            out.append(None); continue
        xs = np.nonzero(mask)[1]
        out.append((xs.mean() - PW/2.)*deg_per_px)
    rows.append((i*0.05, out[0], out[1]))
good_l = [r[1] for r in rows if r[1] is not None]
good_r = [r[2] for r in rows if r[2] is not None]
def stat(v):
    a = np.abs(np.asarray(v))
    return "n=%d 中位|偏心|=%.1f度 最大=%.1f度  >10度占比=%.0f%%" % (len(a), np.median(a), a.max(), 100*np.mean(a > 10))
print("左眼:", stat(good_l))
print("右眼:", stat(good_r))
print("球在画面里找不到的帧数: 左 %d 右 %d" % (sum(r[1] is None for r in rows), sum(r[2] is None for r in rows)))
print("\n关键时刻：")
for r in rows[::20]:
    print("  t=%4.1f  左眼偏心 %s  右眼偏心 %s" % (r[0],
        "----" if r[1] is None else "%+6.1f度" % r[1],
        "----" if r[2] is None else "%+6.1f度" % r[2]))