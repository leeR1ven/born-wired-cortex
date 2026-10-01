from pathlib import Path
p = Path("tools/evolve_chase.py")
s = p.read_text(encoding="utf-8")
old = '''def pick_chase(parents, rng):
    """从一个爹身上抄追球接法回来（没有爹带着接法，就整套重新摇）。"""
    donors = [row for row in parents if isinstance(row.get("chase"), dict)]
    if not donors:
        return mutate_chase(None, rng)
    return mutate_chase(donors[int(rng.integers(len(donors)))]["chase"], rng)
'''
new = '''def cross_chase(first, second, rng):
    """两个爹的追球接法各出一半 —— 跟基因那边的两爹融合是一个道理。

    用户 2026-10-01：孩子应该是「之前最好的模型融合后随机生成」。基因那边一直是
    两爹融合（tools/evolve.py 的 make_child），但追球接法原来只从**一个**爹那儿抄，
    少了一半的融合。这里补上：数值格子取 40%/40%/20%（谁/A+B 平均），
    离散格子（走哪条路、往哪边拐）各 50%。
    """
    out = {}
    for key in set(first) | set(second):
        mine, theirs = first.get(key), second.get(key)
        if mine is None or theirs is None:
            out[key] = theirs if mine is None else mine
            continue
        roll = rng.random()
        if isinstance(mine, float) and isinstance(theirs, float):
            out[key] = mine if roll < .4 else (theirs if roll < .8 else .5 * (mine + theirs))
        else:
            out[key] = mine if roll < .5 else theirs
    return out


def pick_chase(parents, rng):
    """给孩子一套追球接法：一半的孩子拿两个爹的接法融合，一半从一个爹那儿抄；
    之后一样要过 mutate_chase 的「按概率重摇」。

    没有爹带着接法，就整套重新摇。
    """
    donors = [row for row in parents if isinstance(row.get("chase"), dict)]
    if not donors:
        return mutate_chase(None, rng)
    if len(donors) >= 2 and rng.random() < .5:
        i, j = (int(v) for v in rng.choice(len(donors), size=2, replace=False))
        base = cross_chase(donors[i]["chase"], donors[j]["chase"], rng)
    else:
        base = donors[int(rng.integers(len(donors)))]["chase"]
    return mutate_chase(base, rng)
'''
n = s.count(old)
print("命中旧 pick_chase:", n)
assert n == 1, "没找到唯一的旧 pick_chase"
s = s.replace(old, new)
p.write_text(s, encoding="utf-8", newline="\n")
print("写回完成")
