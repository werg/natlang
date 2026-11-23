"""Actual Git worktree ownership, pin guards and deliberately pinned registry reads."""
import json
import subprocess
from pathlib import Path

import pytest
from natlang_neuralese.common import corpus_registry as cr
from natlang_neuralese.common import data_registry as dr


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


def primary(tmp_path):
    owner = tmp_path / 'owner'
    owner.mkdir()
    git(owner, 'init', '-b', 'main')
    git(owner, 'config', 'user.name', 'Registry ownership test')
    git(owner, 'config', 'user.email', 'registry@example.invalid')
    (owner / 'README').write_text('exact code\n')
    (owner / 'training/corpora').mkdir(parents=True)
    (owner / 'training/corpora/frozen.json').write_text(json.dumps({'id': 'frozen', 'path': 'runs/frozen'}))
    git(owner, 'add', '.')
    git(owner, 'commit', '-m', 'initial exact snapshot')
    return owner


def pin(owner, destination, source=None):
    git(owner, 'worktree', 'add', '--detach', str(destination), 'HEAD')
    (destination / '.natlang-pin.json').write_text(json.dumps({'commit': git(destination, 'rev-parse', 'HEAD'),
                                                            'source': str(source or owner)}))
    return destination


def test_pin_writes_resolve_primary_without_rebinding_code_or_read_snapshot(tmp_path, monkeypatch):
    owner = primary(tmp_path)
    code = pin(owner, tmp_path / 'code')
    original = git(code, 'rev-parse', 'HEAD')
    monkeypatch.setattr(dr, 'code_repo', lambda: code)
    monkeypatch.setenv('NATLANG_REPO', str(code))
    monkeypatch.delenv('NATLANG_REGISTRY_REPO', raising=False)
    monkeypatch.setenv('NATLANG_DATA_REGISTRY', str(tmp_path / 'journal'))
    assert dr.registry_owner_repo() == owner
    cr.write_entries(owner, [{'id': 'live', 'path': 'runs/live'}])
    assert set(dr.load(fresh=True).git) == {'frozen'}
    assert set(dr.load(repo=dr.registry_owner_repo(), fresh=True).git) == {'frozen', 'live'}
    assert set(e['id'] for e in cr.load(code)['corpora']) == {'frozen'}
    assert dr.code_repo() == code and git(code, 'rev-parse', 'HEAD') == original
    assert dr.canonical_repo() == owner


def test_registered_pin_of_pin_survives_retirement_of_earlier_source(tmp_path):
    owner = primary(tmp_path)
    first = pin(owner, tmp_path / 'first')
    second = pin(owner, tmp_path / 'second', source=first)
    git(owner, 'worktree', 'remove', '--force', str(first))
    assert dr.canonical_checkout_repo(second) == owner


def test_marker_only_chains_are_followed_and_cycles_missing_or_relative_sources_rejected(tmp_path):
    owner = tmp_path / 'owner';owner.mkdir()
    first = tmp_path / 'first';first.mkdir()
    second = tmp_path / 'second';second.mkdir()
    (first / '.natlang-pin.json').write_text(json.dumps({'source': str(owner)}))
    (second / '.natlang-pin.json').write_text(json.dumps({'source': str(first)}))
    assert dr.canonical_checkout_repo(second) == owner
    (first / '.natlang-pin.json').write_text(json.dumps({'source': str(second)}))
    with pytest.raises(ValueError, match='cyclic'):dr.canonical_checkout_repo(second)
    for source in ['relative', str(tmp_path / 'missing')]:
        (first / '.natlang-pin.json').write_text(json.dumps({'source': source}))
        with pytest.raises(ValueError):dr.canonical_checkout_repo(first)


def test_pin_guards_cover_descendants_and_symlink_destinations_before_mutation(tmp_path):
    code = tmp_path / 'pin';code.mkdir()
    (code / '.natlang-pin.json').write_text('malformed-but-still-protected')
    alias = tmp_path / 'alias';alias.symlink_to(code, target_is_directory=True)
    for destination in [code, code / 'training/corpora', alias / 'new/subdirectory']:
        with pytest.raises(ValueError, match='immutable code pin'):dr.assert_registry_writable(destination)
    with pytest.raises(ValueError):cr.write_entries(code, [{'id': 'bad', 'path': 'runs/bad'}])
    with pytest.raises(ValueError):cr.save(code, {'corpora': []})
    assert sorted(p.name for p in code.iterdir()) == ['.natlang-pin.json']


def test_explicit_registry_owner_is_independent_of_data_anchors_and_cannot_be_a_pin(tmp_path, monkeypatch):
    owner = primary(tmp_path)
    code = pin(owner, tmp_path / 'code')
    alternate = tmp_path / 'alternate';alternate.mkdir()
    monkeypatch.setattr(dr, 'code_repo', lambda: code)
    monkeypatch.setenv('NATLANG_REPO', str(code))
    monkeypatch.setenv('NATLANG_REGISTRY_REPO', str(alternate))
    assert dr.registry_owner_repo() == alternate
    assert dr.canonical_repo() == owner
    monkeypatch.setenv('NATLANG_REGISTRY_REPO', str(code / 'nested'))
    with pytest.raises(ValueError, match='immutable code pin'):dr.registry_owner_repo()


def test_dangling_pin_marker_is_never_treated_as_mutable(tmp_path):
    marker = tmp_path / '.natlang-pin.json';marker.symlink_to(tmp_path / 'absent-marker')
    with pytest.raises(ValueError, match='immutable code pin'):dr.assert_registry_writable(tmp_path)


def test_registry_manifest_reads_follow_explicit_registry_not_executing_code(tmp_path, monkeypatch):
    code = tmp_path / 'code';owner = tmp_path / 'owner'
    for directory, value in [(code, 'pinned'), (owner, 'live')]:
        (directory / 'training/corpus-manifests').mkdir(parents=True)
        (directory / 'training/corpus-manifests/data.json').write_text(json.dumps({'files': [{'path': value}]}))
    monkeypatch.setattr(dr, 'code_repo', lambda: code)
    monkeypatch.setattr(dr, 'stored_manifest', lambda sha: None)
    entry = {'id': 'data', 'path': 'runs/data'}
    assert dr.Registry([entry], [], repo=owner).files('data') == [{'path': 'live'}]
    assert dr.Registry([entry], []).files('data') == [{'path': 'pinned'}]


def test_live_admission_and_producer_inputs_use_explicit_owner_without_unpinning_defaults(tmp_path, monkeypatch):
    owner = primary(tmp_path)
    code = pin(owner, tmp_path / 'code')
    data = tmp_path / 'data';source = data / 'live';source.mkdir(parents=True)
    source_file = source / 'cases.jsonl';source_file.write_text('{"id":"new-live"}\n')
    monkeypatch.setattr(dr, 'code_repo', lambda: code)
    monkeypatch.setenv('NATLANG_REPO', str(code))
    monkeypatch.setenv('NATLANG_DATA_NVME', str(data))
    monkeypatch.setenv('NATLANG_DATA_REGISTRY', str(tmp_path / 'journal'))
    monkeypatch.delenv(dr.ALLOW_ENV, raising=False)
    cr.write_entries(owner, [{'id':'live', 'path':'data_nvme:live','kind':'trajectories','format':'JSONL'}])
    dr._LOADED = None;dr._GIT_CACHE.clear()
    with pytest.raises(dr.UnregisteredData):dr.consume([source_file],run='frozen',how='snapshot',record=False)
    with pytest.raises(dr.UnregisteredData):dr.require_registered([source_file])
    assert dr.consume([source_file],run='live',how='current source',record=False,repo=owner)[0]['id']=='live'
    dr.require_registered([source_file],repo=owner)
    output=data/'output';output.mkdir();(output/'cases.jsonl').write_text('{"id":"derived"}\n')
    record=dr.record_dataset('derived',output,kind='trajectories',format='JSONL',inputs=[source_file],registry_repo=owner)
    assert record['inputs'][0]['id']=='live'
    assert set(dr.load(fresh=True).git)=={'frozen'}
