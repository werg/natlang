"""Generate `natlang tune` examples from semantic-etl train bench rows."""


def examples(row, _defaults):
    if 'entityLink' not in row.get('arguments', {}):
        return []
    group = row['split_group']
    link = row['arguments']['entityLink']
    if not link.get('left') or not link.get('right'):
        return []
    return [{
        'name': f"{row['id']} sameEntity",
        'function': 'sameEntity.nl',
        'group': group,
        'inputs': {'left': link['left'][0], 'right': link['right'][0]},
        'expected': True,
    }]
