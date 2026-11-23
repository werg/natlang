"""Live consumers see post-pin publication without changing static experiment snapshots."""
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / 'scripts/neuralese_data'), str(REPO / 'training/neuralese'), str(REPO)]
import filled_reasoning as fr
import sft_sources as sft
from natlang_neuralese.common import data_registry as dr


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()


def test_fill_discovers_published_work_while_owner_and_code_stay_behind(tmp_path, monkeypatch):
    owner = tmp_path / 'owner'; owner.mkdir()
    git(owner, 'init', '-b', 'main')
    git(owner, 'config', 'user.name', 'Live registry test')
    git(owner, 'config', 'user.email', 'test@example.invalid')
    (owner / 'README').write_text('pinned code')
    git(owner, 'add', '.'); git(owner, 'commit', '-m', 'code snapshot')
    frozen = git(owner, 'rev-parse', 'HEAD')
    pin = tmp_path / 'pin'
    git(owner, 'worktree', 'add', '--detach', str(pin), frozen)
    (pin / '.natlang-pin.json').write_text(json.dumps({'commit': frozen, 'source': str(owner)}))
    work = owner / 'runs/live'; work.mkdir(parents=True)
    entries = owner / 'training/corpora'; entries.mkdir(parents=True)
    (entries / 'live.json').write_text(json.dumps({'id': 'live', 'owner': 'dgx', 'path': str(work),
        'roles': {'grpo-sd-teacher': {'order': 1, 'file': ''}}}))
    git(owner, 'add', 'training'); git(owner, 'commit', '-m', 'publish new teacher work')
    published = git(owner, 'rev-parse', 'HEAD')
    git(owner, 'remote', 'add', 'origin', str(owner))
    git(owner, 'update-ref', 'refs/remotes/origin/main', published)
    git(owner, 'checkout', '--detach', frozen)
    (owner / 'README').write_text('foreign scheduler work in progress')
    monkeypatch.setattr(dr, 'code_repo', lambda: pin)
    monkeypatch.setenv('NATLANG_REPO', str(pin))
    monkeypatch.setenv('NATLANG_DATA_REGISTRY', str(tmp_path / 'journal'))
    monkeypatch.delenv('NATLANG_REGISTRY_REPO', raising=False)
    assert fr.default_works() == []
    owner_ref = dr.registry_owner_ref(dr.registry_owner_repo())
    assert owner_ref == published
    assert fr.default_works(owner, owner_ref) == [str(work)]
    assert git(pin, 'rev-parse', 'HEAD') == frozen == git(owner, 'rev-parse', 'HEAD')
    assert (owner / 'README').read_text() == 'foreign scheduler work in progress'


@pytest.mark.parametrize('prefer,rescan,explicit,live', [([], 0, None, False),
    (['work'], 0, None, False), (['work'], 30, None, True), ([], 0, 'explicit', True)])
def test_only_explicit_or_standing_live_fills_select_mutable_registry(tmp_path, monkeypatch,
                                                                     prefer, rescan, explicit, live):
    owner = tmp_path / 'owner'; owner.mkdir()
    monkeypatch.setattr(dr, 'registry_owner_repo', lambda: owner)
    args = SimpleNamespace(prefer_work=prefer, rescan_minutes=rescan,
                           registry_repo=str(owner) if explicit else None)
    assert fr.generation_registry_repo(args) == (owner if live else None)


def test_work_record_keeps_live_input_admission_on_same_owner_and_ref(tmp_path, monkeypatch):
    captured = []
    monkeypatch.setattr(dr, 'record_dataset', lambda *a, **kw: captured.append(kw))
    work = tmp_path / 'work'
    sft.record_work(work, ['newly-published-input'], registry_repo=tmp_path, registry_ref='published')
    assert captured[0]['registry_repo'] == tmp_path
    assert captured[0]['registry_ref'] == 'published'
    assert captured[0]['inputs'] == ['newly-published-input']
    assert sft.registry_options(None) == sft.registry_options(None, writer=True) == {}
