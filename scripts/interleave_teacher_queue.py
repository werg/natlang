"""Write a new queue rotating pending cases across families and original source groups.

Only order changes: keys, IR indexes, budgets and evidence paths are preserved.
Completed entries retain their original order and remain skipped by the journal.
Apply source quarantine policy before this operation.
"""
import argparse
from collections import Counter, deque
import json
from pathlib import Path


def interleave_entries(entries, finished, describe):
    keys = [entry['key'] for entry in entries]
    if len(set(keys)) != len(keys):
        raise ValueError('queue has duplicate attempt keys')
    completed, families = [], {}
    for entry in entries:
        if entry['key'] in finished:
            completed.append(entry)
            continue
        family, group = describe(entry)
        families.setdefault(family, {}).setdefault(group, deque()).append(entry)
    family_rotation = deque(families)
    group_rotations = {family: deque(groups) for family, groups in families.items()}
    ordered = []
    for _ in range(len(entries) - len(completed)):
        family = family_rotation.popleft()
        groups = group_rotations[family]
        group = groups.popleft()
        queue = families[family][group]
        ordered.append(queue.popleft())
        if queue:
            groups.append(group)
        if groups:
            family_rotation.append(family)
    result = completed + ordered
    assert Counter(entry['key'] for entry in result) == Counter(keys)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--journal', required=True, type=Path)
    parser.add_argument('--ledger', required=True, type=Path)
    args = parser.parse_args()
    entries = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    finished = {event['key'] for event in map(json.loads, args.journal.read_text().splitlines())
                if event.get('event') == 'finish'}
    sources = {}

    def describe(entry):
        source = entry['source']
        if source not in sources:
            sources[source] = [json.loads(line) for line in Path(source).read_text().splitlines() if line.strip()]
        index = entry['index']
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError('queue index must be a nonnegative integer')
        record = sources[source][index]
        family = record.get('curriculum', {}).get('family') or record.get('family') or 'other'
        group = tuple(sorted(record.get('source_groups') or [record['id']]))
        return family, group

    ordered = interleave_entries(entries, finished, describe)
    # Never overwrite a queue a live collector may already have loaded.
    with args.output.open('x') as stream:
        stream.write(''.join(json.dumps(entry) + '\n' for entry in ordered))
    positions = {entry['key']: i for i, entry in enumerate(entries)}
    with args.ledger.open('x') as stream:
        for i, entry in enumerate(ordered):
            family, group = describe(entry)
            stream.write(json.dumps(dict(key=entry['key'], original_order=positions[entry['key']],
                                        next_order=i, family=family, source_groups=group,
                                        finished=entry['key'] in finished)) + '\n')
    print(json.dumps(dict(entries=len(ordered), pending=sum(e['key'] not in finished for e in ordered),
                          output=str(args.output))))


if __name__ == '__main__':
    main()
