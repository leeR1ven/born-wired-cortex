"""Inspect persistent character groups; default is silent association."""
import argparse
import json
from pathlib import Path
import numpy as np
from hebb_text import CharacterHebb
from persistent_candidates import PersistentRecall, RecallConfig

HERE = Path(__file__).resolve().parent


def label(character):
    return {' ': '空格', '\n': '换行', '\t': '制表'}.get(character, character)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prompt')
    parser.add_argument('--mode', choices=('silent','speak'), default='silent')
    parser.add_argument('--steps', type=int, default=12)
    parser.add_argument('--top', type=int, default=8)
    parser.add_argument('--k', type=int)
    parser.add_argument('--budget', type=float)
    parser.add_argument('--model', type=Path, help='Optional alternative CharacterHebb checkpoint; no weights are modified')
    args = parser.parse_args()
    if not 1 <= args.steps <= 2048 or not 1 <= args.top <= 64:
        parser.error('steps must be 1..2048; top must be 1..64')
    summary = json.loads((HERE/'candidate_budget_v1/summary.json').read_text(encoding='utf-8'))
    settings = dict(summary['protocol']['configs'][summary['selected']])
    if args.k is not None:
        settings['candidate_count'] = args.k
    if args.budget is not None:
        settings['candidate_budget'] = args.budget
    model_path = args.model if args.model is not None else HERE/summary['protocol']['model']
    graph = CharacterHebb.load(model_path)
    original_hash = graph.weight_hash()
    recall = PersistentRecall(graph, RecallConfig(**settings))
    recall.warm(args.prompt)
    print(f'输入：{args.prompt}')
    print(f'模式：{args.mode}；候选数：{settings["candidate_count"]}；候选总活动预算：{settings["candidate_budget"]}')
    unknown = ''.join(dict.fromkeys(ch for ch in args.prompt if ch not in graph.ids))
    if unknown:
        print(f'输入中未登录的字符：{unknown}（推进时间，不赋予新字符身份）')
    emitted = []
    for step in range(args.steps):
        raw = recall.currents()
        activity = recall.activity_summary()
        active = sorted(activity['all'], key=lambda item: (-item['amplitude'], graph.ids[item['character']]))
        print(f'{step:02d} 活跃：' + ' '.join(f'{label(x["character"])}({x["amplitude"]:.3f})' for x in active[:args.top]))
        chosen = graph.vocabulary[int(np.argmax(raw))] if np.any(raw > 0) else None
        if args.mode == 'speak' and chosen is not None:
            emitted.append(chosen)
        recall.advance(chosen if args.mode == 'speak' else None, raw)
    if args.mode == 'speak':
        print('续写：' + ''.join(emitted))
    assert graph.weight_hash() == original_hash


if __name__ == '__main__':
    main()
