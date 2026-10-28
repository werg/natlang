"""Write exact, custody-checked privileged references for the controller's training roots."""
import argparse
import hashlib
import json
from pathlib import Path

from natlang_neuralese.common.custody import ProtectedWorlds
from natlang_neuralese.rl.privileged_context import case_facts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--controller', required=True)
    parser.add_argument('--examples', required=True)
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    controller = Path(args.controller)
    split = json.loads((controller / '.natlang/tune/splits.json').read_text())
    if split.get('schema') != 'natlang.tune-split/v1':
        raise ValueError('controller split must be persisted before preparing references')
    custody = ProtectedWorlds.load(root=args.root, registry=Path(args.root) / 'training/protected-worlds.json')
    rows, seen = [], {}
    for line in Path(args.examples).read_text().splitlines():
        if not line.strip():
            continue
        example = json.loads(line)
        group = example['group']
        if group not in split['components']:
            raise ValueError(f'no persisted partition for {group}')
        if split['components'][group] != 'train' or example.get('protected') or 'expected' not in example:
            continue
        key = json.dumps([group, example['function'], example['input']], sort_keys=True)
        if key in seen:
            if seen[key] != example['expected']:
                raise ValueError('conflicting exact controller labels for the same invocation')
            continue
        seen[key] = example['expected']
        identity = hashlib.sha256(key.encode()).hexdigest()
        case = {'id': 'metalearner:' + identity, 'split': 'train', 'source_groups': [group],
                'semantics': {'root': example['function'], 'inputs': example['input'], 'expected': example['expected']}}
        receipt, facts = case_facts(case, custody=custody)
        rows.append({'schema': 'natlang.tune-privileged/2', 'function': example['function'], 'group': group,
                     'input': example['input'], 'result': example['expected'], 'case_id': case['id'], 'custody': receipt,
                     'facts': [{'id': fact.id, 'kind': fact.kind, 'text': fact.text, 'match': list(fact.match),
                                'sha256': fact.digest} for fact in facts], 'source': 'measured counterfactual meta examples'})
    path = controller / 'tune.privileged.jsonl'
    content = ''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows)
    if path.exists() and path.read_text() != content:
        raise ValueError('controller references changed; use a new operation')
    path.write_text(content)
    print(json.dumps({'command': 'controller-supply', 'training_roots': len(rows)}))


if __name__ == '__main__':
    main()
