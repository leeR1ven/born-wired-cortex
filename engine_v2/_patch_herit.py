# -*- coding: utf-8 -*-
"""让「追球接法」也能遗传（根因修复）。

原来每一代每只候选的 chase 接法都是 random_chase 重新摇的：爹身上那套接线传不下去，
所以「生一代、考一代、挑爹再生」根本攒不出本事 —— 挑爹挑得再准，下一代也是从零开始
乱摇。这就是上一批 8 代追球能力没长进的原因。

改法：跟基因一样遗传 —— 从爹身上抄一份接法，每一格有概率重新摇。
"""
import io

p = r"F:\born-wired-cortex\engine_v2\tools\evolve_chase.py"
s = io.open(p, encoding="utf-8", newline="").read()

anchor = '''def random_avoidance(rng):
    """拐弯的力：avoidance_gain（steering→髋 那根线）。0.001 到 0.5，取对数均匀。"""
    return float(10 ** rng.uniform(-3., -.52))
'''
new_block = anchor + '''

def mutate_chase(base, rng):
    """追球接法：从爹那套抄一份，每一格有概率重新摇；没有爹就整套重新摇。

    这样「会追球的接法」才传得下去、也才有机会越攒越准（用户 2026-10-01：每代少生、
    多迭代）。数值格子按倍数抖（gain / turn_gain / turn_time），不是加减，
    这样 0.02 和 0.3 这种量级不会被一下子抹平。
    """
    fresh = random_chase(rng)
    if not isinstance(base, dict) or not base:
        return fresh
    out = dict(fresh)
    for key in ("route", "source", "flip", "pivot"):
        if key in base and rng.random() < .75:
            out[key] = base[key]
    for key, span in (("gain", .5), ("turn_gain", 1.2), ("turn_time", 1.2)):
        if key in base and rng.random() < .7:
            out[key] = float(np.round(max(.001, float(base[key])
                                           * float(np.exp(rng.normal(0., span)))), 5))
    return out


def pick_chase(parents, rng):
    """从一个爹身上抄追球接法回来（没有爹带着接法，就整套重新摇）。"""
    donors = [row for row in parents if isinstance(row.get("chase"), dict)]
    if not donors:
        return mutate_chase(None, rng)
    return mutate_chase(donors[int(rng.integers(len(donors)))]["chase"], rng)
'''
assert anchor in s, "no random_avoidance"
s = s.replace(anchor, new_block, 1)

old_job = '''                genome = dict(genome)
                genome["avoidance_gain"] = random_avoidance(rng)
                touched = sorted(set(list(touched) + ["avoidance_gain"]))
'''
new_job = '''                genome = dict(genome)
                if rng.random() < .5:
                    # 一半的孩子把拐弯的力重新摇一个量级（原来那支被压到 0.0003，
                    # 不重摇就跳不出来）；另一半留着爹的值，让好的量级攒得住。
                    genome["avoidance_gain"] = random_avoidance(rng)
                touched = sorted(set(list(touched) + ["avoidance_gain"]))
'''
assert old_job in s, "no avoidance block"
s = s.replace(old_job, new_job, 1)

old_call = "flee=args.flee, chase=random_chase(rng)))"
new_call = "flee=args.flee, chase=pick_chase(parents, rng)))"
assert old_call in s, "no chase call"
s = s.replace(old_call, new_call, 1)

io.open(p, "w", encoding="utf-8", newline="").write(s)
print("patched: chase wiring is now inherited")