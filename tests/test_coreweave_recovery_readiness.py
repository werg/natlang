import hashlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'coreweave'))
import recovery_readiness as readiness


def test_checkpoint_source_tree_and_serialized_recipe_retain_exact_bindings():
    meta = {'model_config': {'base': '/old/model', 'base_inventory': {'config.json': 'c' * 64}},
            'training': {'source_receipts': [
                {'path': '/old/sources.json', 'sha256': 'a' * 64, 'sources': [
                    {'path': '/old/graphs.zst', 'sha256': 'b' * 64}]}],
                'options': {'recipe_binding': '{"path":"/old/recipe.json","sha256":"' + 'd' * 64 + '"}'}}}
    bindings = readiness.required_bindings(meta)
    assert {(b['path'], b['sha256']) for b in bindings} == {
        ('/old/sources.json', 'a' * 64), ('/old/graphs.zst', 'b' * 64),
        ('/old/model/config.json', 'c' * 64), ('/old/recipe.json', 'd' * 64)}


def test_local_bytes_archive_availability_and_missing_data_are_distinct(tmp_path):
    local = tmp_path / 'local'
    local.write_bytes(b'actual bytes')
    sha = hashlib.sha256(local.read_bytes()).hexdigest()
    needed = [{'path': '/old/actual', 'sha256': sha},
              {'path': '/old/archived', 'sha256': 'b' * 64},
              {'path': '/old/unresolved', 'sha256': 'c' * 64}]
    result = readiness.reconcile(needed, {sha: [str(local)], 'c' * 64: [str(local)]},
                                  {'b' * 64: [{'member': 'archived'}]})
    assert [b['status'] for b in result['bindings']] == [
        'local-exact', 'verified-archive-member', 'unresolved']
    assert result['local_exact'] == result['archive_members'] == result['unresolved'] == 1
    assert not result['launch_ready'] and not result['new_training_exposure']


def test_even_complete_declared_input_coverage_is_not_runtime_qualification(tmp_path):
    local = tmp_path / 'local'
    local.write_bytes(b'bytes')
    sha = hashlib.sha256(local.read_bytes()).hexdigest()
    result = readiness.reconcile([{'path': str(local), 'sha256': sha}], {}, {})
    assert result['unresolved'] == 0
    assert result['runtime_restored'] is False and result['launch_ready'] is False
