#!/usr/bin/env python3
"""Print the verified train selectors that still need a teacher capture."""

import argparse
import json
import sys
from pathlib import Path

from audit_open_goal_references import fingerprint, read_catalog

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'neuralese_data'))
from sft_sources import live_manifest_paths  # noqa: E402
from live_teacher_sources import load_episode  # noqa: E402


def pending_cases(catalog_path, selector_path, capture_roots):
    _, catalog = read_catalog(catalog_path)
    selected = selector_path.read_text().split()
    if len(set(selected)) != len(selected):
        raise ValueError('The selector list contains duplicate goals')
    if invalid := [case for case in selected if catalog.get(case, {}).get('split') != 'train']:
        raise ValueError(f'Selectors must be train goals in the exact catalog: {", ".join(invalid)}')
    by_id = {catalog[case]['taskId']: case for case in selected}
    if len(by_id) != len(selected):
        raise ValueError('Train selectors must identify distinct tasks')
    completed = set()
    for manifest in live_manifest_paths([], capture_roots, allow_empty=True):
        path = Path(manifest)
        spec = json.loads(path.read_text())
        case = by_id.get(spec.get('id'))
        if case is None or case in completed or spec.get('split') != 'train':
            continue
        row = catalog[case]
        task = load_episode(path)['task']
        interaction = task.get('interaction') or {}
        if (interaction.get('kind') == 'open-user-goal' and task['family'] == row['family'] and
                set(task['source_groups']) == set(row['sourceGroups']) and
                fingerprint(interaction['goal']) == row['goalSha256']):
            completed.add(case)
    return [case for case in selected if case not in completed]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('catalog', type=Path)
    parser.add_argument('selectors', type=Path, help='accepted train selectors from the reference audit')
    parser.add_argument('--capture-root', action='append', default=[], type=Path,
                        help='read-only prior capture directory, repeatable')
    args = parser.parse_args()
    for case in pending_cases(args.catalog, args.selectors, args.capture_root):
        print(case)


if __name__ == '__main__':
    main()
