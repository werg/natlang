from collections import Counter, defaultdict

from scripts.rl_tasks.sources.design_brief_gates import cases


def test_exact_gate_packet_becomes_separated_rl_cases():
    rows = cases()
    assert len(rows) == 300
    assert Counter(row['split'] for row in rows) == {
        'train': 180, 'dev': 60, 'confirmation': 60,
    }
    assert len({row['id'] for row in rows}) == 300
    assert all(row['semantics']['oracle'] == 'exact' for row in rows)
    assert all(row['generation']['teacher_generation'] is False for row in rows)
    assert all(any(group.startswith('ext-unit:design-brief-gates:') for group in row['source_groups'])
               for row in rows)
    app_splits = {}
    for row in rows:
        app = row['curriculum']['external']['application']
        assert app_splits.setdefault(app, row['split']) == row['split']
    assert all(row['curriculum']['external']['world_group'] for row in rows)
    assert all(row['semantics']['root'] in row['semantics']['files'] for row in rows)
    assert all(row['semantics']['expected'] not in row['curriculum']['external']['checks']['wrong_answers']
               for row in rows)
    by_split = defaultdict(Counter)
    for row in rows:
        by_split[row['split']][row['semantics']['expected']] += 1
    assert by_split == {
        'train': Counter({'pass': 60, 'fail': 60, 'needs-review': 60}),
        'dev': Counter({'pass': 20, 'fail': 20, 'needs-review': 20}),
        'confirmation': Counter({'pass': 20, 'fail': 20, 'needs-review': 20}),
    }
