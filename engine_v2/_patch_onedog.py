import io
p = r"F:\born-wired-cortex\engine_v2\tools\chase_red_ball.py"
s = io.open(p, encoding="utf-8", newline="").read()
anchor = "def measure(body, brain, eyes, geom, seconds, bearing, distance, elevation=0., seed=0,"
assert anchor in s
helpers = '''def snapshot(brain):
    """把这只狗脑子的细胞状态存一份（刚建出来那一刻的）。

    用户 2026-09-30：同一只狗要测两遍，球分别放左边和右边。两趟之间要把脑子复位，
    不然第二趟是从第一趟跑完的状态起跑，两趟就没法比了。
    """
    net = brain.network
    saved = {"voltage": np.array(net._voltage, copy=True),
             "adaptation": np.array(net._adaptation, copy=True)}
    device = getattr(net, "_device", None)
    if device is not None:
        saved["voltage_device"] = device.voltage.clone()
        saved["adaptation_device"] = device.adaptation.clone()
    return saved


def restore(brain, saved):
    """把脑子放回 snapshot 那一刻。"""
    net = brain.network
    net._voltage[:] = saved["voltage"]
    net._adaptation[:] = saved["adaptation"]
    device = getattr(net, "_device", None)
    if device is not None and "voltage_device" in saved:
        torch = device.torch
        with torch.no_grad():
            device.voltage.copy_(saved["voltage_device"])
            device.adaptation.copy_(saved["adaptation_device"])
        device.stale = False
    auditory = getattr(brain, "auditory", None)
    if auditory is not None and hasattr(auditory, "reset"):
        auditory.reset()


'''
s = s.replace(anchor, helpers + anchor, 1)
io.open(p, "w", encoding="utf-8", newline="").write(s)

p2 = r"F:\born-wired-cortex\engine_v2\tools\chase_task.py"
t = io.open(p2, encoding="utf-8", newline="").read()
old = '''    spec = gaze() if spec is None else spec
    rows = []
    for distance in DISTANCES:
        for bearing in BEARINGS:
            got = _trip(genome, seed, spec, bearing, distance, seconds, True, chase)
            row = _row(got, bearing, distance, 0.)
            if aware:
                free = _trip(genome, seed, spec, bearing, distance, seconds, False, chase)
                row["shifted_m"] = float(np.linalg.norm(np.asarray(got["end_xy"])
                                                        - np.asarray(free["end_xy"])))
                row["blind"] = bool(row["shifted_m"] < BLIND_BAR)
            rows.append(row)
            if loud:'''
new = '''    spec = gaze() if spec is None else spec
    rows = []
    # 用户 2026-09-30：同一只狗要测试两遍，球分别放左边和右边。所以这里按距离
    # 建一次狗，同一只身上先考左边那趟、把脑子复位、再考右边那趟。
    for distance in DISTANCES:
        body, brain, eyes, geom = C.build(genome, chase, seed=seed, spec=spec)
        saved = C.snapshot(brain)
        for bearing in BEARINGS:
            C.restore(brain, saved)
            got = C.measure(body, brain, eyes, geom, seconds, bearing, distance, seed=seed)
            row = _row(got, bearing, distance, 0.)
            if aware:
                C.restore(brain, saved)
                free = C.measure(body, brain, eyes, geom, seconds, bearing, distance,
                                 seed=seed, ball=False)
                row["shifted_m"] = float(np.linalg.norm(np.asarray(got["end_xy"])
                                                        - np.asarray(free["end_xy"])))
                row["blind"] = bool(row["shifted_m"] < BLIND_BAR)
            rows.append(row)
            if loud:'''
assert old in t
t = t.replace(old, new, 1)
old2 = '''                         "  (走起来了)" if row["moving"] else "  (没怎么动)"), flush=True)
    flee_rows = []'''
new2 = '''                         "  (走起来了)" if row["moving"] else "  (没怎么动)"), flush=True)
        eyes.close()
    flee_rows = []'''
assert old2 in t
t = t.replace(old2, new2, 1)
io.open(p2, "w", encoding="utf-8", newline="").write(t)
print("patched")