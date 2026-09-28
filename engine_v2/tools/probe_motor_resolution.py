"""Isolate the old encoder without importing or changing the old application."""
import ast
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OLD = Path(r'F:\born-wired-cortex\model')
source = (OLD / '运动输出区_motor_output.py').read_text(encoding='utf-8-sig')
tree = ast.parse(source)
names = {'肌肉段', '发力转激活', '解码发力'}
selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
assert len(selected) == len(names)
isolated = compile(ast.Module(body=selected, type_ignores=[]), '<old motor encoder>', 'exec')
actions = json.loads((OLD / '动作库.json').read_text(encoding='utf-8'))
stand = np.asarray(actions['站姿']['拍'][0])
sit = np.asarray(actions['坐']['拍'][0])
levels = []
for size in (10, 20, 50, 100):
    space = {'np': np, '肌肉数': 12, '每块肌肉神经元': size,
             '腿区宽度': 12 * size, '运动区宽度': 12 * size + 40}
    exec(isolated, space)
    encode, decode = space['发力转激活'], space['解码发力']
    a, b = encode(stand), encode(sit)
    probes = []
    baseline = encode(np.full(12, .5))
    for amount in (.50, .51, .52, .54, .55):
        inp = np.full(12, .5); inp[0] = amount
        out = encode(inp)
        probes.append({'first_channel_input': amount,
                       'decoded_first_channel': float(decode(out)[0]),
                       'same_pattern_as_0_50': bool(np.array_equal(out, baseline))})
    grid = np.linspace(0, 1, 10001)
    numerical_error = np.abs(np.rint(grid * size) / size - grid)
    levels.append({'neurons_per_channel': size, 'theoretical_output_levels': size + 1,
                   'step_size': 1 / size, 'max_grid_roundtrip_error': float(numerical_error.max()),
                   'stand_active': int(a.sum()), 'sit_active': int(b.sum()),
                   'stand_sit_differing_cells': int(np.count_nonzero(a != b)),
                   'stand_sit_jaccard': float(np.count_nonzero(a & b) / np.count_nonzero(a | b)),
                   'micro_force_probes': probes})
report = {'source': str(OLD / '运动输出区_motor_output.py'),
          'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
          'method': 'AST-extracted original encode/decode functions; only channel width varied',
          'actual_current_neurons_per_channel': 10,
          'results': levels,
          'scope': 'Static encoding resolution only; no PFC rerun, no robot, no neural/actuator noise.',
          'units': 'Normalized actuator command, not measured physical muscle force.'}
(ROOT / 'artifacts' / 'motor_resolution_probe.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
