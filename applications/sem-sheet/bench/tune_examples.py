"""Generate `natlang tune` examples from sem-sheet train bench rows."""


def examples(row, _defaults):
    group = row['split_group']
    cells = row['arguments']['cells']
    expected_labels = row['expected'].get('expectedLabels') or {}
    out = []
    for coord, label in expected_labels.items():
        cell = cells.get(coord) or {}
        deps = cell.get('deps') or []
        input_text = '\n'.join(f"{dep}: {cells.get(dep, {}).get('value', '')}" for dep in deps)
        out.append({
            'name': f"{row['id']} {coord}",
            'function': 'classifyCell.nl',
            'group': group,
            'inputs': {
                'prompt': cell.get('prompt', 'Classify row content'),
                'inputText': input_text,
                'columnName': cell.get('column', coord),
            },
            'expected': label,
        })
    return out
