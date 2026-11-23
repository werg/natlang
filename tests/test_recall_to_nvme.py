import importlib.util
import os
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'recall_to_nvme.py'


def load():
    spec = importlib.util.spec_from_file_location('recall_to_nvme', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_recall_copies_hot_files_and_links_cold_paths(tmp_path):
    m = load()
    archive = tmp_path / 'hdd' / 'teacher-gen'
    group = archive / 'pool' / 'episodes' / 't000001'
    (group / 'row-000' / 'sample-00').mkdir(parents=True)
    (archive / 'pool' / 'claims').mkdir()
    (archive / 'pool' / 'manifest.jsonl').write_text('{"index": 1}\n')
    (group / 'cases.jsonl').write_text('{"id": "a"}\n')
    (group / 'rollouts.jsonl').write_text('big\n')
    (group / 'row-000' / 'sample-00' / 'results.jsonl').write_text('trace\n')
    (archive / 'pool' / 'claims' / 'c1').write_text('x')
    link = tmp_path / 'nvme' / 'teacher-gen'
    link.parent.mkdir()
    link.symlink_to(archive)

    cold = ['*/claims', '*/episodes/*/row-*', '*/episodes/*/rollouts.jsonl']
    dirs, hot, cold_paths = m.plan(archive, cold)
    assert sorted(hot) == ['pool/episodes/t000001/cases.jsonl', 'pool/manifest.jsonl']
    assert sorted(cold_paths) == ['pool/claims', 'pool/episodes/t000001/rollouts.jsonl',
                                  'pool/episodes/t000001/row-000']
    staging = link.with_name('teacher-gen.recall-staging')
    digests = m.build(archive, staging, dirs, hot, cold_paths)
    assert set(digests) == set(hot)
    assert m.verify(archive, staging, hot, cold_paths, digests) == []
    (staging / 'pool' / 'manifest.jsonl').write_text('changed\n')
    assert m.verify(archive, staging, hot, cold_paths) == ['pool/manifest.jsonl']
    m.build(archive, staging, dirs, hot, cold_paths)   # a size change is copied again
    assert m.verify(archive, staging, hot, cold_paths) == []

    m.swap(link, staging, quiet_checks=1, pause=0)
    assert link.is_dir() and not link.is_symlink() and (link / '.keep-on-nvme').exists()
    assert (link / 'pool' / 'episodes' / 't000001' / 'cases.jsonl').read_text() == '{"id": "a"}\n'
    assert not (link / 'pool' / 'manifest.jsonl').is_symlink()
    rollouts = link / 'pool' / 'episodes' / 't000001' / 'rollouts.jsonl'
    assert rollouts.is_symlink() and rollouts.read_text() == 'big\n'
    assert os.path.realpath(link / 'pool' / 'claims') == str((archive / 'pool' / 'claims').resolve())
    assert (archive / 'pool' / 'manifest.jsonl').exists()   # the archive stays complete
