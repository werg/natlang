import json
import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.rl_tasks.audit_natlang_packages import audit
from scripts.rl_tasks.intake_natlang_codebase import intake
from scripts.rl_tasks.build_reviewed_luna_plan import build_plan, select_balanced_train
import scripts.rl_tasks.build_reviewed_luna_plan as plan_builder
import run_reviewed_luna_dispatcher as dispatcher


def _package(tmp_path: Path, *, program='case-app', worlds=None):
    root = tmp_path / program
    package = root / 'package'
    (package / 'triage').mkdir(parents=True)
    (package / 'natlang.json').write_text('{}\n')
    (package / 'reconcile.nl').write_text('''---
args:
  state: State
returns: Result
---
Use inspectEvidence and chooseAction with the supplied state.\n''')
    (package / 'types.ts').write_text('export type State = { events: string[] };\nexport type Result = { action: string };\n')
    (package / 'LICENSE.notice').write_text('Use the fixture source under CC0.\n')
    (package / 'triage' / 'inspectEvidence.nl').write_text('''---
args:
  state: State
returns: Finding
---
Find supported events and unresolved uncertainty.\n''')
    (package / 'triage' / 'chooseAction.nl').write_text('''---
args:
  finding: Finding
returns: Result
---
Choose an authorized action.\n''')
    metadata = {
        'program_id': program, 'source_revision': 'sha256:fixture', 'domain': 'incident response',
        'user_goal': 'make an evidence-based response decision', 'author': {'actor': 'test'},
        'dataset': {},
        'entity_model': {'entities': [{'name': 'Event', 'key': 'id'}], 'relations': [{'from': 'Event', 'to': 'Incident', 'via': 'incidentId'}]},
        'workflows': [{'id': 'reconcile', 'operations': ['reconcile.nl', 'triage.inspectEvidence.nl', 'triage.chooseAction.nl']}],
        'split': {'train_groups': ['world-a'], 'dev_groups': ['world-dev'], 'confirmation_groups': []},
        'split_policy': 'world',
        'package_assets': ['LICENSE.notice'],
    }
    artifact = root / 'dataset.json'
    provenance = root / 'PROVENANCE.json'
    license_evidence = root / 'LICENSE'
    artifact.write_text('{"worlds":2}\n')
    provenance.write_text('{"source":"fixture-incidents","revision":"r1"}\n')
    license_evidence.write_text('CC0-1.0\n')
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    metadata['dataset'] = {'identity': 'fixture-incidents', 'revision': 'r1', 'license': 'CC0-1.0',
                           'source_uri': 'https://example.invalid/fixture', 'artifact_path': str(artifact),
                           'artifact_sha256': digest(artifact), 'provenance_path': str(provenance),
                           'provenance_sha256': digest(provenance), 'license_evidence_path': str(license_evidence),
                           'license_evidence_sha256': digest(license_evidence), 'license_status': 'cleared'}
    (root / 'metadata.json').write_text(json.dumps(metadata))
    worlds_dir = root / 'worlds'
    worlds_dir.mkdir()
    rows = worlds or [
        {'id': 'case-1', 'group_id': 'world-a', 'task_family': 'corroborated', 'operation': 'reconcile.nl',
         'task': 'Select a safe response from corroborated current evidence.', 'inputs': {'state': {'events': ['a', 'b']}},
         'expected': {'action': 'rollback'}, 'oracle': 'exact', 'source_groups': ['incident-dataset:world-a']},
        {'id': 'case-2', 'group_id': 'world-dev', 'task_family': 'retraction', 'operation': 'triage/inspectEvidence.nl',
         'task': 'Identify which event was retracted and preserve uncertainty.', 'inputs': {'state': {'events': ['c', 'd']}},
         'expected': {'supported': [], 'unresolved': ['impact']}, 'oracle': {'level': 'judged', 'rubric': 'Judge evidence support and uncertainty.'},
         'source_groups': ['incident-dataset:world-dev']},
    ]
    (worlds_dir / 'train.jsonl').write_text(json.dumps(rows[0]) + '\n')
    (worlds_dir / 'dev.jsonl').write_text(json.dumps(rows[1]) + '\n')
    return package, root


def test_intake_binds_multiple_operations_and_preserves_exact_vs_judged(tmp_path):
    package, root = _package(tmp_path)
    output = tmp_path / 'tasks.jsonl'
    receipt = intake(package, output=output)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    judged_rows = [json.loads(line) for line in (tmp_path / 'tasks.judged.jsonl').read_text().splitlines()]
    assert receipt['case_count'] == 2
    assert receipt['exact_case_count'] == 1 and receipt['judged_case_count'] == 1
    assert receipt['distinct_operations'] == 2
    assert {row['split'] for row in rows} == {'train'}
    assert {row['split'] for row in judged_rows} == {'dev'}
    assert 'incident-dataset:world-a' in rows[0]['source_groups']
    assert not any(group.startswith('codebase:') for group in rows[0]['source_groups'])
    assert 'For this invocation: Select a safe response' in rows[0]['semantics']['files']['reconcile.nl']
    assert rows[0]['semantics']['oracle'] == 'exact'
    assert judged_rows[0]['semantics']['oracle']['level'] == 'judged'
    assert judged_rows[0]['curriculum']['reference']['root'] == []
    assert 'reconcile.nl' in rows[0]['semantics']['files']
    assert 'CC0' in rows[0]['semantics']['files']['LICENSE.notice']
    assert receipt['semantic_status'].startswith('fixture-backed, not yet runtime-verified')


def test_related_worlds_cannot_cross_splits(tmp_path):
    row_a = {'id': 'a', 'group_id': 'shared', 'task_family': 'state', 'operation': 'reconcile.nl',
             'task': 'Continue the previous decision.', 'inputs': {'state': {}}, 'expected': {}, 'oracle': 'exact',
             'source_groups': ['incident-dataset:shared']}
    row_b = {**row_a, 'id': 'b'}
    package, root = _package(tmp_path, worlds=[row_a, row_b])
    (root / 'metadata.json').write_text(json.dumps({**json.loads((root / 'metadata.json').read_text()),
        'split': {'train_groups': ['shared'], 'dev_groups': ['world-dev'], 'confirmation_groups': []}}))
    (root / 'worlds' / 'train.jsonl').write_text(json.dumps(row_a) + '\n')
    (root / 'worlds' / 'dev.jsonl').write_text(json.dumps(row_b) + '\n')
    with pytest.raises(ValueError, match='file disagrees|crosses splits'):
        intake(package, output=tmp_path / 'bad.jsonl')


def test_audit_reports_cross_package_near_duplicates_and_unknown_semantics(tmp_path):
    first, _ = _package(tmp_path, program='first')
    second, _ = _package(tmp_path, program='second')
    report = audit([first, second], near_threshold=.8, max_near_partners=0)
    assert report['summary']['near_duplicate_pairs'] == 1
    assert report['near_duplicate_cap_violations'] == ['first', 'second']
    assert report['summary']['packages_with_unknown_semantic_coverage'] == 0


def test_audit_uses_real_callable_folder_degree_multiset(tmp_path):
    first, _ = _package(tmp_path, program='one-child-set')
    second, _ = _package(tmp_path, program='larger-child-set')
    (second / 'triage' / 'extra.nl').write_text('''---
args: {}
returns: string
---
Inspect a separate signal.\n''')
    left = audit([first])['packages'][0]['topology']['callable_folder_degree_multiset']
    right = audit([second])['packages'][0]['topology']['callable_folder_degree_multiset']
    assert left == [2, 2]
    assert right == [2, 3]


def test_prompt_similarity_retains_lexical_instruction_differences(tmp_path):
    first, _ = _package(tmp_path, program='wording-one')
    second, _ = _package(tmp_path, program='wording-two')
    root = second / 'reconcile.nl'
    source = root.read_text()
    root.write_text(source.replace('Use inspectEvidence and chooseAction with the supplied state.',
                                    'Compare competing observations, retain uncertainty, then select a permitted response.'))
    report = audit([first, second], near_threshold=.8)
    pair = report['near_duplicates'][0]
    assert pair['normalized_syntax_jaccard'] > 0.75
    assert pair['prompt_scaffold_jaccard'] < 1.0


def test_balanced_plan_selection_checks_all_train_candidates_and_preserves_groups():
    rows = []
    for operation in ('audit.nl', 'reconcile.nl'):
        for system in ('loghub-system:Android', 'loghub-system:Linux'):
            rows.append({'id': f'{operation}:{system}', 'split': 'train',
                         'source_groups': [system], 'semantics': {'oracle': 'exact'},
                         'curriculum': {'external': {'operation': operation}}})

    class Protected:
        def __init__(self): self.checked = []
        def check(self, row):
            self.checked.append(row['id'])
            return {'receipt_for': row['id']}

    protected = Protected()
    selected, receipts = select_balanced_train(rows, max_cases=2, seed=1, protected=protected)
    assert len(protected.checked) == len(rows)
    assert len(receipts) == len(selected) == 2
    assert {row['curriculum']['external']['operation'] for row in selected} == {'audit.nl', 'reconcile.nl'}
    assert all(row['source_groups'][0].startswith('loghub-system:') for row in selected)


def test_plan_builder_emits_unapproved_full_prompt_plan_and_pins_inventory(tmp_path, monkeypatch):
    rows = []
    for operation in ('audit.nl', 'reconcile.nl'):
        for system in ('loghub-system:Android', 'loghub-system:Linux'):
            identifier = f'{operation}:{system}'
            rows.append({
                'id': identifier, 'kind': 'lambda_source', 'family': 'codebase-fixture', 'source': 'test',
                'split': 'train', 'source_groups': [system], 'source_ids': [identifier],
                'curriculum': {'reference': {'root': [['return_result', {'status': 'success', 'value': {'ok': True}}]]},
                               'external': {'program_id': 'test-codebase', 'operation': operation}},
                'semantics': {'root': operation, 'files': {operation: '---\nargs: {}\nreturns: object\n---\nDo the operation.\n'},
                              'inputs': {}, 'expected': {'ok': True}, 'oracle': 'exact'},
            })
    source = tmp_path / 'verified' / 'cases.jsonl'
    source.parent.mkdir()
    source.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    import hashlib
    verification = source.parent / 'verification.json'
    verification.write_text(json.dumps({
        'schema': 'natlang.broad-rl-case-verification/1',
        'cases_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'results': [{'id': row['id'], 'ok': True} for row in rows],
    }))
    runtime = tmp_path / 'runtime'
    for folder in ('dist', 'scripts', 'src'):
        (runtime / folder).mkdir(parents=True, exist_ok=True)
        (runtime / folder / 'runtime.js').write_text('// runtime fixture\n')
    (runtime / 'frozen-runtime.json').write_text(json.dumps({'files': plan_builder.tree_identity(runtime)}))
    authority = tmp_path / 'authority.json'
    authority.write_text('{}\n')
    intake_receipt = tmp_path / 'intake.json'
    intake_receipt.write_text(json.dumps({'admission_eligible': True}))
    calls = []

    class Protected:
        digest = 'current-protected-worlds-digest'
        def check(self, row):
            calls.append(row['id'])
            return {'case': row['id'], 'status': 'passed'}

    output = tmp_path / 'plan.json'
    report = build_plan(
        source=source, verification=verification, runtime=runtime, authority=authority,
        campaign_id='test-codebase-plan-v1', campaign_root=Path(dispatcher.ROOT / 'runs' / '.plan-test-never-created'),
        output=output, inventory_output=tmp_path / 'inventory.json',
        selected_source=tmp_path / 'selected.jsonl', protection_receipt=tmp_path / 'protection.json',
        max_cases=2, seed=7, intake_receipt=intake_receipt, protected=Protected(),
    )
    plan = json.loads(output.read_text())
    assert report['status'] == 'prepared_unapproved'
    assert plan['root_approved'] is False
    assert plan['slots'] == 6 and plan['prompt_rendering'] == 'full'
    assert len(plan['cases']) == 2 and len(calls) == len(rows)
    assert plan['source_inventory']['sha256']
    assert report['operation_counts'] == {'audit.nl': 1, 'reconcile.nl': 1}
    reviewed = {**plan, 'root_approved': True}
    reviewed_path = tmp_path / 'reviewed-plan.json'
    reviewed_path.write_text(json.dumps(reviewed, sort_keys=True))
    validated, identity = dispatcher.validate_plan(reviewed_path, dispatcher.digest(reviewed_path))
    assert validated['prompt_rendering'] == 'full'
    assert identity['slots'] == 6 and identity['source_inventory']['entries'] == 2


def test_intake_folder_files_and_verifier_targets(tmp_path):
    package, root = _package(tmp_path)
    path = root / 'worlds/train.jsonl'
    row = json.loads(path.read_text())
    row['folder_files'] = {'src/app.ts': 'before'}
    row['expected_files'] = {'src/app.ts': 'after'}
    path.write_text(json.dumps(row) + '\n')
    out = tmp_path / 'out.jsonl'
    intake(package, output=out)
    case = json.loads(out.read_text())
    assert case['semantics']['folder_files'] == {'src/app.ts': 'before'}
    assert case['semantics']['expected_files'] == {'src/app.ts': 'after'}
    assert 'after' not in json.dumps(case['semantics']['files'])
    for bad in ('../escape', '/absolute', 'a/./b', 'a//b', 'a\\b'):
        row['folder_files'] = {bad: 'content'}
        path.write_text(json.dumps(row) + '\n')
        with pytest.raises(ValueError, match='safe relative paths'):
            intake(package, output=out)
    del row['folder_files']
    path.write_text(json.dumps(row) + '\n')
    with pytest.raises(ValueError, match='expected_files requires folder_files'):
        intake(package, output=out)


def test_intake_binds_hidden_world_services_and_checks_fixture_hash(tmp_path):
    package, root = _package(tmp_path)
    source = root / 'service.ts'
    source.write_text('const data = /*__NATLANG_WORLD_FIXTURE__*/; export function read(): string { return data.value; }')
    fixture = root / 'service-worlds.jsonl'
    fixture.write_text(''.join(json.dumps({'id': key, 'fixture': {'value': val}}) + '\n' for key, val in [('case-1', 'private-a'), ('case-2', 'private-b')]))
    metadata_path = root / 'metadata.json'
    metadata = json.loads(metadata_path.read_text())
    metadata['external_services'] = {'store': {'source_path': source.name, 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'fixtures_path': fixture.name, 'fixtures_sha256': hashlib.sha256(fixture.read_bytes()).hexdigest()}}
    metadata_path.write_text(json.dumps(metadata))
    out = tmp_path / 'out.jsonl'
    intake(package, output=out)
    a = json.loads(out.read_text())
    b = json.loads((tmp_path / 'out.judged.jsonl').read_text())
    assert 'private-a' in a['semantics']['services']['store']
    assert 'private-b' not in a['semantics']['services']['store']
    assert 'private-b' in b['semantics']['services']['store']
    assert 'private-a' not in json.dumps(a['semantics']['files'])
    fixture.write_text(fixture.read_text() + '\n')
    with pytest.raises(ValueError, match='fixture hash mismatch'):
        intake(package, output=out)
