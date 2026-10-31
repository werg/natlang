from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.rl_tasks.codebase_versioning import version_codebase


def test_version_codebase_publishes_independent_bytes_and_never_mutates_source(tmp_path):
    source = tmp_path / 'app-v1'
    (source / 'package').mkdir(parents=True)
    (source / 'package' / 'app.nl').write_text('old source\n')
    (source / 'metadata.json').write_text('{}\n')
    target = tmp_path / 'app-v2'
    version_codebase(source, target, ('metadata.json', 'package'))
    original = source / 'package' / 'app.nl'
    copied = target / 'package' / 'app.nl'
    assert original.read_text() == copied.read_text() == 'old source\n'
    assert (original.stat().st_dev, original.stat().st_ino) != (copied.stat().st_dev, copied.stat().st_ino)
    copied.write_text('new source\n')
    assert original.read_text() == 'old source\n'
    with pytest.raises(FileExistsError):
        version_codebase(source, target, ('metadata.json', 'package'))


def test_version_codebase_rejects_symlinked_source_files(tmp_path):
    source, outside = tmp_path / 'app-v1', tmp_path / 'outside'
    source.mkdir(); outside.write_text('private')
    (source / 'linked').symlink_to(outside)
    with pytest.raises(ValueError, match='cannot be a symlink'):
        version_codebase(source, tmp_path / 'app-v2', ('linked',))


def test_version_codebase_independence_check_matches_ignored_runtime_cache_files(tmp_path):
    source = tmp_path / 'app-v1'
    cache = source / 'host' / 'python-deps' / '__pycache__'
    cache.mkdir(parents=True)
    (cache / 'module.pyc').write_bytes(b'ignored cache')
    journal = source / 'host' / 'journals' / 'old'
    journal.mkdir(parents=True)
    (journal / 'events.jsonl').write_text('ignored journal\n')
    (source / 'host' / 'worker.py').write_text('worker\n')
    target = tmp_path / 'app-v2'
    version_codebase(source, target, (part for part in ('host',)))
    assert (target / 'host' / 'worker.py').read_text() == 'worker\n'
    assert not (target / 'host' / 'python-deps' / '__pycache__').exists()
    assert not (target / 'host' / 'journals').exists()
