import json
import subprocess
from pathlib import Path
import pytest
from natlang_neuralese.rl import task_oracle_custody as pipeline


def put(path, value):
    path.write_text(json.dumps(value))
    return pipeline._reference(path)


def test_group_oracle_requires_complete_collector_runtime_inventory(tmp_path, monkeypatch):
    runtime = tmp_path / 'runtime' / 'dist' / 'teacher'; runtime.mkdir(parents=True)
    collector = put(runtime / 'collector.js', 1)
    actions = put(runtime / 'required-actions.js', 2)
    lock = put(tmp_path / 'runtime' / 'package-lock.json', 3)
    group = put(tmp_path / 'groups.jsonl', {})
    derivation = put(tmp_path / 'derivation.json', {})
    dependencies = put(tmp_path / 'runtime-inventory.json', {'schema': 'natlang.rl-oracle-runtime-inventory/1',
        'inventory': {'@dependency-files': {'module': {'sha256': 'abc'}}, '@dependency-source-links': {}}})
    preflight = put(tmp_path / 'runtime-preflight.json', {'matched': True})
    postflight = put(tmp_path / 'runtime-postflight.json', {'matched': True, 'inventory_sha256': dependencies['sha256']})
    from natlang_neuralese.common import runtime_inventory
    monkeypatch.setattr(runtime_inventory, 'task_runtime_inventory', lambda _: {
        '@dependency-files': {'module': {'sha256': 'abc'}}, '@dependency-source-links': {}})
    result = {'id': 'task', 'reference': {'ok': True}, 'wrong_answer_accepted': False,
              'missing_required_read_accepted': False}
    source_prefix = str(tmp_path / 'recorded-source' / 'ts-host') + '/'
    identity = put(tmp_path / 'verification-identity.json', {'runtime_root': source_prefix.rstrip('/'), 'node': subprocess.check_output(['node', '--version'], text=True).strip(), 'inputs': [
        {'path': source_prefix + 'dist/teacher/collector.js', 'sha256': collector['sha256']},
        {'path': source_prefix + 'dist/teacher/required-actions.js', 'sha256': actions['sha256']},
        {'path': source_prefix + 'package-lock.json', 'sha256': lock['sha256']},
        {'path': '/evidence/groups.jsonl', 'sha256': group['sha256']},
        {'path': '/evidence/derivation.json', 'sha256': derivation['sha256']},
        {'path': '/evidence/runtime-inventory.json', 'sha256': dependencies['sha256']}]})
    journal = put(tmp_path / 'oracle-groups.jsonl', {'passed': True, 'failure': None, 'results': [result]})
    cases = put(tmp_path / 'cases.jsonl', {'id': 'task'})
    proof = {'schema': 'natlang.lfm3-derived-skill-oracle-validation/1', 'source': {'sha256': cases['sha256']}, 'group_identity_sha256': identity['sha256'],
             'group_journal_sha256': journal['sha256'],
             'runtime': {'collector': collector['sha256'], 'required_actions': actions['sha256']},
             'runtime_dependency_inventory_sha256': dependencies['sha256'],
             'runtime_preflight_sha256': preflight['sha256'], 'runtime_postflight_sha256': postflight['sha256'],
             'results': [result]}
    validation = put(tmp_path / 'oracle.json', proof)
    manifest = put(tmp_path / 'source.json', {'cases': 'cases.jsonl',
        'oracle_validation': {'path': 'oracle.json', 'sha256': validation['sha256']}, 'runtime_requirement': proof['runtime']})
    pipeline.verify_task_oracle_runtime(manifest['path'], tmp_path / 'runtime', cases['sha256'])
    proof['schema'] = 'natlang.lfm3-effect-oracle-validation/1'
    proof['results'][0].update(effect_check='files', missing_required_effect_accepted=False,
                               missing_required_effect_check=False)
    journal = put(tmp_path / 'oracle-groups.jsonl', {'passed': True, 'failure': None, 'results': proof['results']})
    proof['group_journal_sha256'] = journal['sha256']
    validation = put(tmp_path / 'oracle.json', proof)
    manifest = put(tmp_path / 'source.json', {'cases': 'cases.jsonl',
        'oracle_validation': {'path': 'oracle.json', 'sha256': validation['sha256']}, 'runtime_requirement': proof['runtime']})
    pipeline.verify_task_oracle_runtime(manifest['path'], tmp_path / 'runtime', cases['sha256'])
    proof['results'][0]['missing_required_effect_check'] = True
    journal = put(tmp_path / 'oracle-groups.jsonl', {'passed': True, 'failure': None, 'results': proof['results']})
    proof['group_journal_sha256'] = journal['sha256']
    validation = put(tmp_path / 'oracle.json', proof)
    manifest = put(tmp_path / 'source.json', {'cases': 'cases.jsonl',
        'oracle_validation': {'path': 'oracle.json', 'sha256': validation['sha256']}, 'runtime_requirement': proof['runtime']})
    with pytest.raises(ValueError, match='adversarial evidence'):
        pipeline.verify_task_oracle_runtime(manifest['path'], tmp_path / 'runtime', cases['sha256'])
    proof['results'][0]['missing_required_effect_check'] = False
    service = put(runtime / 'service-receipt.js', 4)
    identity_doc = json.loads(Path(identity['path']).read_text())
    identity_doc['inputs'].append({'path': source_prefix + 'dist/teacher/service-receipt.js', 'sha256': service['sha256']})
    identity = put(tmp_path / 'verification-identity.json', identity_doc)
    proof.update(schema='natlang.lfm3-world-oracle-validation/1', group_identity_sha256=identity['sha256'])
    proof['runtime']['service_receipt'] = service['sha256']
    proof['results'][0].update(control_kind='delivery', unsafe_route_accepted=False, unsafe_world_check=False)
    journal = put(tmp_path / 'oracle-groups.jsonl', {'passed': True, 'failure': None, 'results': proof['results']})
    proof['group_journal_sha256'] = journal['sha256']
    validation = put(tmp_path / 'oracle.json', proof)
    manifest = put(tmp_path / 'source.json', {'cases': 'cases.jsonl',
        'oracle_validation': {'path': 'oracle.json', 'sha256': validation['sha256']}, 'runtime_requirement': proof['runtime']})
    pipeline.verify_task_oracle_runtime(manifest['path'], tmp_path / 'runtime', cases['sha256'])
    monkeypatch.setattr(runtime_inventory, 'task_runtime_inventory', lambda _: {
        '@dependency-files': {'module': {'sha256': 'changed'}}, '@dependency-source-links': {}})
    with pytest.raises(ValueError, match='installed runtime dependencies differ'):
        pipeline.verify_task_oracle_runtime(manifest['path'], tmp_path / 'runtime', cases['sha256'])
    monkeypatch.setattr(runtime_inventory, 'task_runtime_inventory', lambda _: {
        '@dependency-files': {'module': {'sha256': 'abc'}}, '@dependency-source-links': {}})
    proof['results'][0]['unsafe_route_accepted'] = True
    journal = put(tmp_path / 'oracle-groups.jsonl', {'passed': True, 'failure': None, 'results': proof['results']})
    proof['group_journal_sha256'] = journal['sha256']
    validation = put(tmp_path / 'oracle.json', proof)
    manifest = put(tmp_path / 'source.json', {'cases': 'cases.jsonl',
        'oracle_validation': {'path': 'oracle.json', 'sha256': validation['sha256']}, 'runtime_requirement': proof['runtime']})
    with pytest.raises(ValueError, match='adversarial evidence'):
        pipeline.verify_task_oracle_runtime(manifest['path'], tmp_path / 'runtime', cases['sha256'])
    proof['results'][0]['unsafe_route_accepted'] = False
    journal = put(tmp_path / 'oracle-groups.jsonl', {'passed': True, 'failure': None, 'results': proof['results']})
    proof['group_journal_sha256'] = journal['sha256']
    validation = put(tmp_path / 'oracle.json', proof)
    manifest = put(tmp_path / 'source.json', {'cases': 'cases.jsonl',
        'oracle_validation': {'path': 'oracle.json', 'sha256': validation['sha256']}, 'runtime_requirement': proof['runtime']})
    (runtime / 'added.js').write_text('changed')
    with pytest.raises(ValueError, match='runtime inventory differs'):
        pipeline.verify_task_oracle_runtime(manifest['path'], tmp_path / 'runtime', cases['sha256'])


