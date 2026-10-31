import pytest
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.rl_tasks.build_reviewed_luna_plan import select_balanced_train
from scripts.rl_tasks.intake_natlang_codebase import intake


class Protection:
    def __init__(self):
        self.checked = []

    def check(self, row):
        self.checked.append(row['id'])
        return {'checked': row['id']}


def row(identifier, quality_status='ready'):
    return {
        'id': identifier, 'split': 'train', 'source_groups': [f'tau2_bench:train:{identifier}'],
        'semantics': {'oracle': 'exact'},
        'curriculum': {'external': {
            'program_id': 'telecom-app', 'operation': 'resolveIncident.nl',
            'quality_status': quality_status,
            'quality_reason': 'unresolved upstream consent' if quality_status != 'ready' else None,
        }},
    }


def test_quality_pending_rows_are_protected_but_not_selected():
    pending, good = row('pending', 'quality_pending_paid_action_authority'), row('good')
    protection, exclusions = Protection(), []
    selected, receipts = select_balanced_train(
        [pending, good], max_cases=2, seed=3, protected=protection,
        quality_exclusions_out=exclusions,
    )
    assert protection.checked == ['pending', 'good']
    assert [item['id'] for item in selected] == ['good']
    assert len(receipts) == 1
    assert exclusions == [{
        'case_id': 'pending', 'quality_status': 'quality_pending_paid_action_authority',
        'quality_reason': 'unresolved upstream consent', 'source_groups': pending['source_groups'],
        'receipt': {'checked': 'pending'},
    }]


def test_explicit_selector_cannot_force_quality_pending_row():
    pending = row('pending', 'quality_pending_paid_action_authority')
    with pytest.raises(ValueError, match='quality-pending'):
        select_balanced_train([pending, row('good')], max_cases=1, seed=3,
                              protected=Protection(), case_ids={'pending'})


def test_intake_preserves_world_quality_disposition(tmp_path):
    from test_natlang_codebase_intake import _package

    worlds = [{
        'id': 'pending-source-row', 'group_id': 'world-a', 'task_family': 'consent',
        'operation': 'reconcile.nl', 'task': 'Respect missing authority.', 'inputs': {'state': {}},
        'expected': {'status': 'blocked'}, 'oracle': 'exact',
        'source_groups': ['incident-dataset:world-a'],
        'quality_status': 'quality_pending_paid_action_authority',
        'quality_reason': 'upstream gold expects an unauthorized charge',
    }, {
        'id': 'dev-control-row', 'group_id': 'world-dev', 'task_family': 'control',
        'operation': 'triage/inspectEvidence.nl', 'task': 'Inspect the dev control.',
        'inputs': {'state': {}}, 'expected': {'status': 'control'},
        'oracle': {'level': 'judged', 'rubric': 'Check the control.'},
        'source_groups': ['incident-dataset:world-dev'],
    }]
    package, _ = _package(tmp_path, worlds=worlds)
    output = tmp_path / 'cases.jsonl'
    intake(package, output=output)
    case = json.loads(output.read_text().splitlines()[0])
    external = case['curriculum']['external']
    assert external['quality_status'] == 'quality_pending_paid_action_authority'
    assert external['quality_reason'] == 'upstream gold expects an unauthorized charge'
