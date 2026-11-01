import json
import hashlib
import sys
import subprocess
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.rl_tasks.audit_natlang_packages import audit
from scripts.rl_tasks.intake_natlang_codebase import intake, _validate_natlang_child_call_visibility
from scripts.rl_tasks.execution_adapter import verify_runtime_trees, verify_runtime_tree
from scripts.rl_tasks.build_reviewed_luna_plan import (build_plan, select_balanced_train,
                                                       _intake_external_service_files)
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
    (package / 'types.ts').write_text('export type State = { events: string[] };\nexport type Finding = { supported: string[]; unresolved: string[] };\nexport type Result = { action: string };\n')
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


def test_intake_rejects_service_fixture_metadata_as_an_undeclared_root_argument(tmp_path):
    worlds = [
        {'id': 'case-1', 'group_id': 'world-a', 'task_family': 'corroborated', 'operation': 'reconcile.nl',
         'task': 'Select a safe response from corroborated current evidence.',
         'inputs': {'state': {'events': ['a', 'b']}, 'serviceFixtureId': 'case-1'},
         'expected': {'action': 'rollback'}, 'oracle': 'exact',
         'source_groups': ['incident-dataset:world-a']},
        {'id': 'case-2', 'group_id': 'world-dev', 'task_family': 'retraction',
         'operation': 'triage/inspectEvidence.nl',
         'task': 'Identify which event was retracted and preserve uncertainty.',
         'inputs': {'state': {'events': ['c', 'd']}},
         'expected': {'supported': [], 'unresolved': ['impact']},
         'oracle': {'level': 'judged', 'rubric': 'Judge evidence support and uncertainty.'},
         'source_groups': ['incident-dataset:world-dev']},
    ]
    package, _ = _package(tmp_path, worlds=worlds)
    with pytest.raises(ValueError, match=r'serviceFixtureId is not a parameter of reconcile'):
        intake(package, output=tmp_path / 'must-not-be-intaken.jsonl')


def test_child_call_visibility_rejects_sibling_and_accepts_companion_or_explicit_use():
    sibling = {
        'compare.nl': '---\nargs: {}\nreturns: Result\n---\nCall reviewHypothesis for both agreements.\n',
        'review/reviewHypothesis.nl': '---\nargs: {}\nreturns: Finding\n---\nReview one hypothesis.\n',
    }
    with pytest.raises(ValueError, match=r'compare\.nl calls reviewHypothesis.*only defined at'):
        _validate_natlang_child_call_visibility(sibling)

    companion = {
        'review.nl': '---\nargs: {}\nreturns: Finding\n---\nCall reviewHypothesis for each requested item.\n',
        'review/reviewHypothesis.nl': '---\nargs: {}\nreturns: Finding\n---\nAssess one hypothesis.\n',
    }
    receipt = _validate_natlang_child_call_visibility(companion)
    assert receipt['schema'] == 'natlang.child-call-visibility/1'
    assert receipt['call_reference_count'] == 1

    explicitly_used = {
        'compare.nl': '---\nargs: {}\nreturns: Result\nuses: [helpers/reviewHypothesis]\n---\nCall reviewHypothesis for both agreements.\n',
        'helpers/reviewHypothesis.nl': '---\nargs: {}\nreturns: Finding\n---\nReview one hypothesis.\n',
    }
    assert _validate_natlang_child_call_visibility(explicitly_used)['call_reference_count'] == 1


def test_intake_binds_collector_executed_application_controller_to_actual_package(tmp_path):
    package, root = _package(tmp_path)
    worlds_path = root / 'worlds' / 'dev.jsonl'
    dev = json.loads(worlds_path.read_text())
    dev['operation'] = 'reconcile.nl'
    worlds_path.write_text(json.dumps(dev) + '\n')
    root_source = package / 'reconcile.nl'
    root_source.write_text(root_source.read_text().replace('returns: Result\n', 'returns: Result\nuses: [workflow]\n'))
    (package / 'workflow.ts').write_text('export async function run(state: State, operations: { step(): Promise<Result> }): Promise<Result> { return await operations.step(); }\n')
    (package / 'reconcile').mkdir()
    (package / 'reconcile' / 'step.nl').write_text('---\nargs: {}\nreturns: Result\n---\nChoose the response.\n')
    metadata_path = root / 'metadata.json'
    metadata = json.loads(metadata_path.read_text())
    metadata['collector_controller'] = {'schema': 'natlang.application-controller/1', 'module': 'workflow',
        'export': 'run', 'operations': 'step', 'input_arguments': ['state']}
    metadata_path.write_text(json.dumps(metadata))
    out = tmp_path / 'controller.jsonl'
    receipt = intake(package, output=out)
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert receipt['case_count'] == 2
    assert all(row['semantics']['application_controller'] == metadata['collector_controller'] for row in rows)
    assert 'CC0' in rows[0]['semantics']['files']['LICENSE.notice']
    assert receipt['semantic_status'].startswith('fixture-backed, not yet runtime-verified')
    assert 'application_controller' in dispatcher._required_runtime_semantics_features(rows, [{'index': 0}])


def test_intake_rejects_collector_controller_not_attached_to_public_root(tmp_path):
    package, root = _package(tmp_path)
    worlds_path = root / 'worlds' / 'dev.jsonl'
    dev = json.loads(worlds_path.read_text())
    dev['operation'] = 'reconcile.nl'
    worlds_path.write_text(json.dumps(dev) + '\n')
    root_source = package / 'reconcile.nl'
    root_source.write_text(root_source.read_text().replace('returns: Result\n', 'returns: Result\nuses: [different]\n'))
    (package / 'reconcile').mkdir(exist_ok=True)
    (package / 'reconcile' / 'chooseAction.nl').write_text('---\nargs: {}\nreturns: Result\n---\nChoose an action.\n')
    (package / 'workflow.ts').write_text('export async function run(state: State, operations: object): Promise<Result> { return {} as Result; }\n')
    metadata_path = root / 'metadata.json'
    metadata = json.loads(metadata_path.read_text())
    metadata['collector_controller'] = {'schema': 'natlang.application-controller/1', 'module': 'workflow',
        'export': 'run', 'operations': 'chooseAction', 'input_arguments': ['state']}
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='must expose controller module'):
        intake(package, output=tmp_path / 'detached.jsonl')


def test_uninvoked_execution_adapter_is_rejected_at_intake(tmp_path):
    package, root = _package(tmp_path)
    host = root / 'host' / 'runner.py'
    host.parent.mkdir()
    host.write_text('def run(): return 1\n')
    digest = hashlib.sha256(host.read_bytes()).hexdigest()
    metadata_path = root / 'metadata.json'
    metadata = json.loads(metadata_path.read_text())
    metadata['execution_adapter'] = {'paths': {'host/runner.py': digest}}
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match='not invoked by'):
        intake(package, output=tmp_path / 'bound.jsonl')


def test_intake_rejects_tampered_execution_adapter_path(tmp_path):
    package, root = _package(tmp_path)
    host = root / 'host' / 'runner.py'
    host.parent.mkdir()
    host.write_text('def run(): return 1\n')
    metadata_path = root / 'metadata.json'
    metadata = json.loads(metadata_path.read_text())
    metadata['execution_adapter'] = {'paths': {'host/runner.py': '0' * 64}}
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='not invoked by'):
        intake(package, output=tmp_path / 'tampered.jsonl')


def test_dispatcher_entrypoint_imports_shared_tree_checker_from_other_cwd(tmp_path):
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'run_reviewed_luna_dispatcher.py'
    result = subprocess.run([sys.executable, str(script), '--help'], cwd=tmp_path,
                            text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    assert 'usage: run_reviewed_luna_dispatcher.py' in result.stdout


def test_fixture_binding_resolves_codebase_root_marker_for_runtime_service(tmp_path):
    package, root = _package(tmp_path)
    host = root / 'host'
    host.mkdir()
    service = host / 'service.ts'
    service.write_text('const fixture = /*__NATLANG_WORLD_FIXTURE__*/;\n'
                       'const root = /*__NATLANG_CODEBASE_ROOT__*/;\n')
    fixtures = root / 'source' / 'fixtures.jsonl'
    fixtures.parent.mkdir()
    fixtures.write_text(''.join(json.dumps({'id': identifier, 'fixture': {'x': index}}) + '\n'
                               for index, identifier in enumerate(('case-1', 'case-2'))))
    metadata_path = root / 'metadata.json'
    metadata = json.loads(metadata_path.read_text())
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    metadata['external_services'] = {'service': {
        'source_path': 'host/service.ts', 'source_sha256': digest(service),
        'fixtures_path': 'source/fixtures.jsonl', 'fixtures_sha256': digest(fixtures)}}
    metadata_path.write_text(json.dumps(metadata))
    receipt = intake(package, output=tmp_path / 'service.jsonl')
    row = json.loads((tmp_path / 'service.jsonl').read_text().splitlines()[0])
    bound_source = row['semantics']['services']['service']
    assert json.dumps(root.resolve().as_posix()) in bound_source
    assert '/*__NATLANG_CODEBASE_ROOT__*/' not in bound_source


def test_runtime_dependency_tree_is_verified_and_bound_into_service(tmp_path):
    package, root = _package(tmp_path)
    service = root / 'host' / 'service.ts'
    service.parent.mkdir()
    service.write_text('const fixture = /*__NATLANG_WORLD_FIXTURE__*/;\n'
                       'const deps = /*__NATLANG_RUNTIME_TREE_tau2_python_deps__*/;\n')
    fixtures = root / 'source' / 'fixtures.jsonl'
    fixtures.parent.mkdir()
    fixtures.write_text(''.join(json.dumps({'id': identifier, 'fixture': {'x': index}}) + '\n'
                               for index, identifier in enumerate(('case-1', 'case-2'))))
    deps = tmp_path / 'runtime-deps'
    module = deps / 'package' / 'mod.py'
    module.parent.mkdir(parents=True)
    module.write_text('value = 7\n')
    manifest_path = root / 'host' / 'runtime-manifest.json'
    manifest = {'schema': 'natlang.telecom-python-dependency-tree/1', 'root': str(deps.resolve()),
                'files': {'package/mod.py': hashlib.sha256(module.read_bytes()).hexdigest()}}
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + '\n')
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    metadata_path = root / 'metadata.json'
    metadata = json.loads(metadata_path.read_text())
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    metadata['external_services'] = {'service': {
        'source_path': 'host/service.ts', 'source_sha256': digest(service),
        'fixtures_path': 'source/fixtures.jsonl', 'fixtures_sha256': digest(fixtures)}}
    metadata['collector_runtime'] = {'trees': [{
        'name': 'tau2_python_deps', 'root': str(deps), 'manifest_path': 'host/runtime-manifest.json',
        'manifest_sha256': manifest_sha}]}
    metadata_path.write_text(json.dumps(metadata))
    out = tmp_path / 'runtime-tree.jsonl'
    receipt = intake(package, output=out)
    assert receipt['runtime_dependency_trees'][0]['file_count'] == 1
    row = json.loads(out.read_text().splitlines()[0])
    bound_source = row['semantics']['services']['service']
    assert json.dumps(str(deps.resolve())) in bound_source
    assert '/*__NATLANG_RUNTIME_TREE_tau2_python_deps__*/' not in bound_source
    module.write_text('value = 8\n')
    with pytest.raises(ValueError, match='runtime dependency file is missing or changed'):
        verify_runtime_trees(metadata, root)


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
    teacher = runtime / 'dist' / 'teacher'
    teacher.mkdir(parents=True, exist_ok=True)
    cli = teacher / 'cli.js'
    collector = teacher / 'collector.js'
    cli.write_text("console.log('--text-neuralese-emulation');\n")
    collector.write_text('// frozen collector fixture\n')
    build = {'mode': 'explicit_compiled_dist', 'compiled_dist_files': {
        'teacher/cli.js': hashlib.sha256(cli.read_bytes()).hexdigest(),
        'teacher/collector.js': hashlib.sha256(collector.read_bytes()).hexdigest(),
    }}
    (runtime / 'frozen-runtime.json').write_text(json.dumps({
        'build': build, 'files': plan_builder.tree_identity(runtime)}))
    authority = tmp_path / 'authority.json'
    authority.write_text('{}\n')
    codebase = tmp_path / 'service-codebase'
    package = codebase / 'package'
    package.mkdir(parents=True)
    service = codebase / 'host' / 'collector-service.ts'
    service.parent.mkdir()
    service.write_text('export function read(): number { return 1; }\n')
    fixtures = codebase / 'source' / 'fixtures.jsonl'
    fixtures.parent.mkdir()
    fixtures.write_text('{"id":"fixture","fixture":{}}\n')
    intake_receipt = tmp_path / 'intake.json'
    service_sha = hashlib.sha256(service.read_bytes()).hexdigest()
    fixtures_sha = hashlib.sha256(fixtures.read_bytes()).hexdigest()
    runtime_tree_root = tmp_path / 'shared-runtime'
    runtime_module = runtime_tree_root / 'pkg' / 'module.py'
    runtime_module.parent.mkdir(parents=True)
    runtime_module.write_text('value = 1\n')
    tree_manifest = tmp_path / 'runtime-tree.json'
    tree_manifest.write_text(json.dumps({'schema': 'natlang.telecom-python-dependency-tree/1',
        'root': str(runtime_tree_root.resolve()),
        'files': {'pkg/module.py': hashlib.sha256(runtime_module.read_bytes()).hexdigest()}},
        sort_keys=True) + '\n')
    tree_sha = hashlib.sha256(tree_manifest.read_bytes()).hexdigest()
    tree_binding = verify_runtime_tree(root=runtime_tree_root, manifest_path=tree_manifest,
                                       manifest_sha256=tree_sha)
    tree_binding['name'] = 'fixture-runtime'
    root_input_preflight = {'schema': 'natlang.root-input-binding-preflight/1',
        'loaded_by': 'teacher.prepareProgramNode', 'checked': [
            {'id': row['id'], 'root': row['semantics']['root'], 'input_names': [],
             'binding': 'teacher.prepareProgramNode/collector-root-argument-binding'} for row in rows]}
    intake_receipt.write_text(json.dumps({'admission_eligible': True, 'package_path': str(package.resolve()),
        'root_input_binding_preflight': root_input_preflight,
        'external_services': {'telecom': {'source_path': 'host/collector-service.ts', 'source_sha256': service_sha,
            'fixtures_path': 'source/fixtures.jsonl', 'fixtures_sha256': fixtures_sha}},
        'runtime_dependency_trees': [tree_binding]}))
    from scripts.rl_tasks import root_input_binding_frontend
    monkeypatch.setattr(root_input_binding_frontend, 'validate_root_input_bindings',
        lambda candidate_rows, runtime_dist=None: root_input_preflight)
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
        max_cases=2, slots=1, max_turns=12, max_model_requests=24, case_seconds=300,
        seed=7, intake_receipt=intake_receipt, protected=Protected(), reasoning_effort='medium',
    )
    plan = json.loads(output.read_text())
    assert report['status'] == 'prepared_unapproved'
    assert plan['root_approved'] is False
    assert plan['slots'] == 1 and plan['prompt_rendering'] == 'full'
    assert plan['reasoning_effort'] == 'medium'
    assert (plan['max_turns'], plan['max_model_requests'], plan['case_seconds']) == (12, 24, 300)
    assert plan['resource_policy']['resumable'] is False
    assert plan['resource_policy']['backfill'] is False
    assert '--no-resumable' in report['dispatch_command_after_review']
    assert len(plan['cases']) == 2 and len(calls) == len(rows)
    assert plan['source_inventory']['sha256']
    assert plan['artifact_hashes'][str(service.resolve())] == service_sha
    assert plan['artifact_hashes'][str(fixtures.resolve())] == fixtures_sha
    assert plan['runtime_dependency_trees'][0]['tree_sha256'] == tree_binding['tree_sha256']
    assert report['operation_counts'] == {'audit.nl': 1, 'reconcile.nl': 1}
    reviewed = {**plan, 'root_approved': True}
    reviewed_path = tmp_path / 'reviewed-plan.json'
    reviewed_path.write_text(json.dumps(reviewed, sort_keys=True))
    validated, identity = dispatcher.validate_plan(reviewed_path, dispatcher.digest(reviewed_path))
    assert validated['prompt_rendering'] == 'full'
    assert identity['slots'] == 1 and identity['source_inventory']['entries'] == 2
    service.write_text('export function read(): number { return 2; }\n')
    with pytest.raises(ValueError, match='Reviewed artifact changed'):
        dispatcher.validate_plan(reviewed_path, dispatcher.digest(reviewed_path))
    service.write_text('export function read(): number { return 1; }\n')
    runtime_module.write_text('value = 2\n')
    with pytest.raises(ValueError, match='runtime dependency file is missing or changed'):
        dispatcher.validate_plan(reviewed_path, dispatcher.digest(reviewed_path))
    runtime_module.write_text('value = 1\n')
    reviewed.pop('external_service_bindings')
    reviewed_path.write_text(json.dumps(reviewed, sort_keys=True))
    with pytest.raises(ValueError, match='service source pins differ from authenticated intake'):
        dispatcher.validate_plan(reviewed_path, dispatcher.digest(reviewed_path))


def test_intake_folder_files_and_verifier_targets(tmp_path):
    package, root = _package(tmp_path)
    path = root / 'worlds/train.jsonl'
    row = json.loads(path.read_text())
    row['folder_files'] = {'src/app.ts': 'before'}
    row['expected_files'] = {'src/app.ts': 'after'}
    row['files_oracle'] = {'compare': 'rewrite', 'threshold': 1.0, 'rubric': 'Preserve behavior.'}
    path.write_text(json.dumps(row) + '\n')
    out = tmp_path / 'out.jsonl'
    intake(package, output=out)
    case = json.loads(out.read_text())
    assert case['semantics']['folder_files'] == {'src/app.ts': 'before'}
    assert case['semantics']['expected_files'] == {'src/app.ts': 'after'}
    assert case['semantics']['files_oracle'] == row['files_oracle']
    assert 'after' not in json.dumps(case['semantics']['files'])
    for bad in ('../escape', '/absolute', 'a/./b', 'a//b', 'a\\b', 'C:/absolute', 'a\x00b'):
        row['folder_files'] = {bad: 'content'}
        path.write_text(json.dumps(row) + '\n')
        with pytest.raises(ValueError, match='safe relative paths'):
            intake(package, output=out)
    del row['folder_files']
    path.write_text(json.dumps(row) + '\n')
    with pytest.raises(ValueError, match='expected_files requires folder_files'):
        intake(package, output=out)


def test_intake_validates_semantic_json_file_oracle_sources_and_paths(tmp_path):
    package, root = _package(tmp_path)
    path = root / 'worlds/train.jsonl'
    row = json.loads(path.read_text())
    row['folder_files'] = {'source/clauses.json': '{"6":"The agreement includes all information."}\n'}
    row['expected_files'] = {'reports/ledger.json': '{"schema":"review/2","findings":[]}\n'}
    row['files_oracle'] = {
        'compare': 'json-semantic-structured',
        'json_report_paths': ['reports/ledger.json'],
        'rationale_context_paths': ['source/clauses.json'],
        'comparison_spec': {
            'schema_version': 'semantic-structured-comparison/1', 'unordered_paths': [], 'ordered_paths': [],
            'prose_paths': ['/findings/*/explanation'],
            'prose_policy': 'Check each rationale against the exact source clause; accept faithful wording variations.',
        },
    }
    semantic_comparison = {
        'schema_version': 'semantic-structured-comparison/1', 'unordered_paths': [], 'ordered_paths': [],
        'prose_paths': ['/findings/*/explanation'],
        'prose_policy': 'Check each rationale against the exact source clause; accept faithful wording variations.',
    }
    row['oracle'] = {'level': 'normalized', 'normalization': 'semantic-structured',
        'comparison_spec': semantic_comparison, 'context': {'source_files': row['folder_files']}}
    path.write_text(json.dumps(row) + '\n')
    out = tmp_path / 'semantic-files.jsonl'
    receipt = intake(package, output=out)
    case = json.loads(out.with_name('semantic-files.judged.jsonl').read_text().splitlines()[0])
    assert case['semantics']['files_oracle'] == row['files_oracle']
    assert case['semantics']['oracle'] == row['oracle']
    assert case['curriculum']['external']['oracle_kind'] == 'semantic-structured'
    # The fixture includes a separate judged dev world; semantic-structured is
    # counted as its own oracle kind rather than being folded into judged.
    assert receipt['oracle_counts'] == {'exact': 0, 'judged': 1, 'semantic_structured': 1}

    invalid = [
        {**row['files_oracle'], 'json_report_paths': ['../ledger.json']},
        {**row['files_oracle'], 'rationale_context_paths': ['missing/source.json']},
        {**row['files_oracle'], 'comparison_spec': {**row['files_oracle']['comparison_spec'],
            'prose_paths': ['/findings/*/bad~2field']}},
        {**row['files_oracle'], 'json_report_paths': ['source/clauses.json']},
    ]
    for spec in invalid:
        row['files_oracle'] = spec
        path.write_text(json.dumps(row) + '\n')
        with pytest.raises(ValueError, match='json-semantic-structured|JSON comparison paths|json_report_paths|rationale_context_paths|expected new report'):
            intake(package, output=out)
    row['files_oracle'] = {
        'compare': 'json-semantic-structured', 'json_report_paths': ['reports/ledger.json'],
        'rationale_context_paths': ['source/clauses.json'],
        'comparison_spec': {'schema_version': 'semantic-structured-comparison/1', 'unordered_paths': [],
            'ordered_paths': [], 'prose_paths': ['/findings/*/explanation'],
            'prose_policy': 'Check explanations against source.'},
    }
    row['oracle']['context']['source_files'] = {'forged/source.json': 'not bound to folder_files'}
    path.write_text(json.dumps(row) + '\n')
    with pytest.raises(ValueError, match='semantic-structured oracle context'):
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
