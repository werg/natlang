"""A union publication must preserve lineage inputs and outlive its caller's cgroup."""
import importlib.util
import json
from pathlib import Path
import subprocess
import pytest


def watcher(monkeypatch, tmp_path):
    source = Path(__file__).resolve().parents[1] / 'scripts/neuralese_data/target_set_watcher.py'
    spec = importlib.util.spec_from_file_location('target_set_watcher_under_test', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    directory = tmp_path / 'target-set'
    directory.mkdir()
    monkeypatch.setattr(module, 'SET_DIR', directory)
    monkeypatch.setattr(module, 'STATE', directory / 'state.json')
    monkeypatch.setattr(module, 'STAGING', tmp_path / 'staging')
    return module


def test_union_build_confirms_an_independent_ledger_waiter(monkeypatch, tmp_path):
    module = watcher(monkeypatch, tmp_path)
    monkeypatch.setattr(module, 'ledger_knows', lambda unit: False)
    calls = []
    def run(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 3 if args[0] == 'systemctl' else 0)
    monkeypatch.setattr(module.subprocess, 'run', run)
    assert module.build_union([], 4) == 'launched'
    args, kwargs = calls[-1]
    assert '--detach' in args and '--wait' in args
    assert args[args.index('--container') + 1] == 'union-target-r4'
    assert args[args.index('--priority') + 1] == 'low'
    assert kwargs['check'] is True


def test_publishing_r4_does_not_delete_a_required_r2_input(monkeypatch, tmp_path):
    module = watcher(monkeypatch, tmp_path)
    parent = tmp_path / 'runs' / (module.UNION_BASE + '-r2') / 'program-header/mellum21/text-data.jsonl'
    parent.parent.mkdir(parents=True)
    parent.write_text('required checkpoint input\n')
    (tmp_path / 'runs' / (module.UNION_BASE + '-r3')).mkdir()
    union = tmp_path / 'runs' / (module.UNION_BASE + '-r4')
    union.mkdir()
    (union / 'receipt.json').write_text(json.dumps({'renderings': {'full': {
        'twin_identity_equal': True, 'by_tokenizer': {'lfm25': {
            'documents': 1, 'text_data_sha256': 'a' * 64,
            'by_split': {'train': {'documents': 1}}}}}}}))
    monkeypatch.setattr(module.rntr, 'parse', lambda args: args)
    monkeypatch.setattr(module.rntr, 'make_entry', lambda args: {'id': 'union-r4'})
    monkeypatch.setattr(module, 'register', lambda *args: None)
    monkeypatch.setattr(module.subprocess, 'run', lambda args, **kwargs: subprocess.CompletedProcess(args, 0))
    state = {'revision': 3, 'union_revisions': [2, 3]}
    build = {'rev': 4, 'members': [], 'skipped': [], 'spec': {}, 'signature': 'r4'}
    module.finish_revision(state, build, union)
    assert parent.read_text() == 'required checkpoint input\n'
    assert state['revision'] == 4


@pytest.mark.parametrize('documents,train,twin_equal', [(0, 0, True), (1, 0, True), (1, 1, False)])
def test_empty_or_inconsistent_union_cannot_be_published(monkeypatch, tmp_path, documents, train, twin_equal):
    module = watcher(monkeypatch, tmp_path)
    union = tmp_path / 'union'
    union.mkdir()
    (union / 'receipt.json').write_text(json.dumps({'renderings': {'full': {
        'twin_identity_equal': twin_equal, 'by_tokenizer': {'lfm25': {
            'documents': documents, 'by_split': {'train': {'documents': train}}}}}}}))
    monkeypatch.setattr(module, 'register', lambda *args: pytest.fail('invalid union reached publication'))
    build = {'rev': 4, 'members': [], 'skipped': [], 'spec': {}, 'signature': 'bad'}
    with pytest.raises(ValueError, match='registration refused'):
        module.finish_revision({'revision': 3}, build, union)
