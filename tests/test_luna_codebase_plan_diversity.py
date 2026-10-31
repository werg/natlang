"""Keep diverse applications represented even when their API sizes differ."""
from collections import Counter

import pytest

from scripts.rl_tasks.build_reviewed_luna_plan import select_balanced_train, _primary_world_group


class Protection:
    def __init__(self):
        self.checked = []

    def check(self, row):
        self.checked.append(row['id'])
        return {'case_id': row['id']}


def case(app, operation, index):
    return {'id': f'{app}:{operation}:{index}', 'split': 'train',
            'source_groups': [f'codebase:{app}:v1', f'upstream:{app}:world-{index}'],
            'semantics': {'oracle': 'exact'},
            'curriculum': {'external': {'program_id': app, 'operation': operation}}}


def test_large_api_does_not_drown_different_application():
    # Unequal API breadth and invocation counts must not become sampling votes.
    rows = [case('wide-api', f'operation-{op}.nl', i) for op in range(12) for i in range(4)]
    rows += [case('small-api', 'solve.nl', i) for i in range(8)]
    protected = Protection()
    selected, receipts = select_balanced_train(rows, max_cases=8, seed=13, protected=protected)
    assert Counter(row['curriculum']['external']['program_id'] for row in selected) == {
        'wide-api': 4, 'small-api': 4}
    assert len({row['curriculum']['external']['operation'] for row in selected
                if row['curriculum']['external']['program_id'] == 'wide-api'}) == 4
    assert len(protected.checked) == len(rows)
    assert len(receipts) == 8
    assert len({row['id'] for row in selected}) == 8


def test_world_diversity_does_not_count_generated_codebase_alias_as_world():
    row = case('app', 'solve.nl', 1)
    assert _primary_world_group(row) == 'upstream:app:world-1'


def test_duplicate_case_identity_is_not_extra_generation_supply():
    row = case('app', 'solve.nl', 1)
    with pytest.raises(ValueError, match='duplicate training case identity'):
        select_balanced_train([row, dict(row)], max_cases=2, seed=1, protected=Protection())


def test_codebase_intake_preserves_domain_without_breaking_collector_schema(tmp_path):
    from test_natlang_codebase_intake import _package
    from scripts.rl_tasks.intake_natlang_codebase import intake
    import json
    package, _ = _package(tmp_path)
    output = tmp_path / 'cases.jsonl'
    intake(package, output=output)
    row = json.loads(output.read_text().splitlines()[0])
    assert row['curriculum']['domain'] == 'other'
    assert row['curriculum']['external']['authored_domain'] == 'incident response'
