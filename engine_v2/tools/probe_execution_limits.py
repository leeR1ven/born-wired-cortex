"""做多大？每步的时间花在哪、有多少细胞真的在工作。

  python tools/probe_execution_limits.py            # 出货配置
  python tools/probe_execution_limits.py --grow     # 再加一张各区能长到多大的表

一步的开销 = 细胞数 x f + 连接数 x g，两个系数都由"每步把所有细胞和所有连接都算
一遍"决定，所以规模就是时间的线性函数。这里量三件事：规模、真正被点亮的比例、
以及时间落在那几个数组运算上。--grow 再逐个区试"还能不能建起来"（有几个区的连线
写死了尺寸，细胞一多，同一个目标细胞的输入下界总和就会超过它的预算）。
"""
import argparse, cProfile, io, json, pstats, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from born_wired.embodied import EmbodiedController     # noqa: E402
from born_wired.go2_body import Go2Body                # noqa: E402
from born_wired.stereo_senses import RawEyes           # noqa: E402

ARENA = ROOT / "models" / "reflex_arena.xml"
CONFIG = ROOT / "live_config.json"
ENVIRONMENT = ("body_touch", "foot_obstacle", "foot_load", "foot_slip")


def shipped():
    parameters = dict(json.loads(CONFIG.read_text(encoding="utf-8-sig"))["parameters"])
    return parameters


def build(body, parameters):
    units = {k: parameters[k] for k in ("motor_units", "proprio_units", "association_units")
             if k in parameters}
    return EmbodiedController(body.home_angles, body.lower_limits, body.upper_limits,
                              eye_limits=(body.eye_lower_limits, body.eye_upper_limits),
                              **parameters), units


def activity_report(brain, pixels_of, steps=16, warmup=8):
    rate = None
    for step in range(steps):
        target, activation = brain.step(_observation[0], environment=_environment,
                                        eye_pixels=pixels_of(), ear_waveform=_ear,
                                        dt=.01, learn=True, locomotion=.6)
        _observation[0] = _body.step(target, duration=.01, activation=activation)
        brain.eye_command() if brain.eye_encoder is not None else None
        if step >= warmup:
            now = np.asarray(brain.network.activity)
            rate = now if rate is None else rate + now
    return rate / float(steps - warmup)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grow", action="store_true")
    options = parser.parse_args(argv)
    parameters = shipped()
    width, height = parameters["eye_width"], parameters["eye_height"]
    global _body, _environment, _ear, _observation
    _body = Go2Body(model_path=ARENA)
    _observation = [_body.reset(seed=0)]
    _environment = {name: np.zeros(4) for name in ENVIRONMENT}
    _environment["foot_support"] = np.zeros(4, dtype=bool)
    _ear = np.zeros((2, 160))
    brain, units = build(_body, parameters)
    eyes = RawEyes(_body, width=width, height=height)
    print("出货配置：每只眼 %dx%d，运动/本体/联想 %d/%d/%d，%d 细胞，%d 连接"
          % (width, height, units.get("motor_units", 200), units.get("proprio_units", 64),
             units.get("association_units", 256), brain.network.n_neurons, len(brain.synapses.src)))

    rate = activity_report(brain, eyes.observe_raw)
    groups = brain.groups
    retina = np.concatenate([groups[name] for name in
                             ("photoreceptors", "retinal_interneurons", "retinal_opponent",
                              "retinal_contrast", "binocular")])
    mask = np.zeros(brain.network.n_neurons, dtype=bool)
    mask[retina] = True
    print("视网膜一侧 %d 细胞（%.1f%%），其余 %d"
          % (mask.sum(), 100*mask.mean(), (~mask).sum()))
    for level in (.01, .1, .2, .3, .5):
        lit = rate > level
        print("  活动 > %.2f：全部 %5.2f%%   视网膜一侧 %5.2f%%   其余 %5.2f%%"
              % (level, 100*lit.mean(), 100*lit[mask].mean(), 100*lit[~mask].mean()))
    print("  连接里有活动源的：%.2f%%" % (100*(rate[brain.synapses.src] > .01).mean()))

    calls = dict(eye_pixels=np.zeros(brain.eye_shape, dtype=np.uint8), ear_waveform=_ear,
                 environment=_environment, dt=.01)
    for learn in (True, False):
        for _ in range(3):
            brain.step(_observation[0], learn=learn, **calls)
        started = time.perf_counter()
        for _ in range(10):
            brain.step(_observation[0], learn=learn, **calls)
        print("大脑一步（learning=%s）：%.2f ms" % (learn, (time.perf_counter()-started)/10*1000))
    started = time.perf_counter()
    for _ in range(10):
        eyes.observe_raw()
    print("眼睛渲染：%.2f ms" % ((time.perf_counter()-started)/10*1000))
    profile = cProfile.Profile()
    profile.enable()
    for _ in range(10):
        brain.step(_observation[0], learn=True, **calls)
    profile.disable()
    stream = io.StringIO()
    pstats.Stats(profile, stream=stream).sort_stats("tottime").print_stats(6)
    print("\n".join(stream.getvalue().splitlines()[4:12]))
    eyes.close()

    if options.grow:
        print("\n各个区还能长到多大（每只眼 %dx%d 不变）：" % (width, height))
        plans = [("运动单位", "motor_units", (800, 1600, 3200, 6400)),
                 ("本体单位", "proprio_units", (256, 512, 1024)),
                 ("联想单位", "association_units", (1024, 2048, 4096, 8192))]
        for label, key, values in plans:
            for value in values:
                trial = dict(parameters)
                trial[key] = value
                try:
                    grown, _ = build(_body, trial)
                    print("  %s %5d：建起来了，%d 细胞 %d 连接"
                          % (label, value, grown.network.n_neurons, len(grown.synapses.src)))
                except Exception as error:
                    print("  %s %5d：建不起来 - %s" % (label, value, error))
                    break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
