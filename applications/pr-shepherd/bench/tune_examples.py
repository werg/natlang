"""Generate `natlang tune` examples from pr-shepherd train bench rows."""


def examples(row, _defaults):
    group = row['split_group']
    hunks = row['arguments']['hunks']
    expected = row['expected']
    out = []
    if expected['findingCount'] == 0:
        out.append({
            'name': f"{row['id']} mergeVerdict",
            'function': 'mergeVerdict.nl',
            'group': group,
            'inputs': {'findings': []},
            'expected': 'approve',
        })
    if expected['semanticHunks'] == 0 and hunks:
        out.append({
            'name': f"{row['id']} trivial",
            'function': 'isTrivialHunk.nl',
            'group': group,
            'inputs': {'hunk': hunks[0]},
            'expected': True,
        })
    return out
