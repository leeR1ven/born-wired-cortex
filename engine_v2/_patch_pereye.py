import ast
p = "born_wired/embodied.py"
s = open(p, encoding="utf-8", newline="").read()

start = "        if red is not None:\r\n            wanted = {}\r\n"
end = "                     command[(str(tier), str(direction))], float(weight))\r\n"
i = s.find(start)
j = s.find(end)
assert i > 0 and j > i, (i, j)
j += len(end)

new = """        if red is not None:
            # One set of command cells per eye, fed only by that eye's own red
            # cells and pushing only that eye's own muscles.  With one shared set
            # the two eyes are told the same thing and can only stare in the same
            # direction - which is wrong for anything closer than the horizon.
            # Each eye turning until its own picture is centred puts both eyes on
            # the ball, and the nearer the ball the more they turn toward each
            # other: convergence falls out of the wiring instead of being a
            # separate reflex.  Which eye a cell belongs to is not a guess, it is
            # where the cell sits in the retinal layer.
            per_eye = bool(red.get('per_eye', False))
            stride = int(np.prod(self.eye_shape[1:]))

            def whose(cell):
                return int(cell)//stride if per_eye else None

            wanted = {}
            for cell, direction, tier, weight in red['edges']:
                key = (str(tier), str(direction), whose(cell))
                wanted[key] = wanted.get(key, 0.) + float(weight)
            latch = max(0., float(red.get('latch', 0.)))
            cross = max(0., float(red.get('cross', 0.)))
            command, mirrored = {}, {}
            for key, total in wanted.items():
                name = 'eye_red_' + '_'.join(str(part) for part in key)
                command[key] = cells(name, 1, 1, time=float(red.get('cell_time', .03)),
                                     budget=1. + total + latch)[0]
                mirrored[key] = cells(name + '_inhibition', 1, -1, time=.01,
                                      budget=1. + cross)[0]
            # A cell may not reach itself, so the latch is a pair that excites the
            # pair: whatever fires, keeps firing.
            for key, cell in command.items():
                edge(cell, mirrored[key], 1.)
                if latch > 0:
                    relay = cells('eye_red_' + '_'.join(str(part) for part in key) + '_relay',
                                  1, 1, time=.03, budget=1. + latch)[0]
                    edge(cell, relay, latch)
                    edge(relay, cell, latch)
            if cross > 0:
                for key in list(command):
                    tier, side, eye = key
                    for here, other in (('left', 'right'), ('right', 'left'),
                                        ('up', 'down'), ('down', 'up')):
                        if side == here and (tier, other, eye) in mirrored:
                            edge(command[key], mirrored[(tier, other, eye)], cross)
            pushes = red.get('push')
            for tier, rank in (('move', 0), ('hold', 1)):
                for side, axis in (('left', 0), ('right', 0), ('down', 1), ('up', 1)):
                    for eye in ((0, 1) if per_eye else (None,)):
                        key = (tier, side, eye)
                        cell = command.get(key)
                        if cell is None:
                            continue
                        source = mirrored[key] if side in ('right', 'up') else cell
                        if pushes is None:
                            strength = float(red['move_push'] if tier == 'move'
                                             else red['hold_push'])
                        else:
                            strength = float(pushes[side][rank])
                        joints = ((2*eye + axis,) if per_eye
                                  else ((0, 2) if axis == 0 else (1, 3)))
                        for joint in joints:
                            push(joint, source, strength)
            for cell, direction, tier, weight in red['edges']:
                edge(self.groups['retinal_opponent'][cell],
                     command[(str(tier), str(direction), whose(cell))], float(weight))
""".replace("\n", "\r\n")

s = s[:i] + new + s[j:]
open(p, "w", encoding="utf-8", newline="").write(s)
ast.parse(open(p, encoding="utf-8").read())
print("embodied per-eye ok")