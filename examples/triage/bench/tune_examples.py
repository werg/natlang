"""``natlang tune`` root examples from one triage fixture row: the two decisions the program makes about the ticket
(``classify.nl`` always, ``is_urgent.nl`` for a ticket that is not spam), in the ``examples.json`` shape. The row's split
group is the example's group, so a ticket's examples stay on one side of tune's own splits."""


def examples(row, defaults):
    ticket = row['arguments']['tickets'][0]
    label, urgent = row['expected']['label'], row['expected']['urgent']
    group = row['split_group']
    out = [{'name': f"{row['id']} classify", 'function': 'classify.nl', 'group': group,
            'inputs': {'ticket': ticket, 'rubric': defaults['rubric']}, 'expected': label}]
    if label != 'spam':
        out.append({'name': f"{row['id']} urgent", 'function': 'is_urgent.nl', 'group': group,
                    'inputs': {'ticket': ticket}, 'expected': bool(urgent)})
    return out
