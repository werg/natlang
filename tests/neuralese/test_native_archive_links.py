"""Continuations share verified immutable history bytes, retaining normal cross-device writes."""
from types import SimpleNamespace
import errno
import os

import pytest

from test_native_state_batch_import import _history
from natlang_neuralese.serve.store import TensorStore
from natlang_neuralese.tune.build import import_artifacts, import_states, persist_artifacts


def _archive(tmp_path):
    engine, records = _history(tmp_path, 'lfm', 'adam')
    source = tmp_path / 'source'
    persist_artifacts(engine, [records[-1].id], source / 'artifacts', source / 'states')
    target = SimpleNamespace(store=TensorStore())
    import_artifacts(target, source / 'artifacts')
    import_states(target, source / 'states')
    return engine, records, source, target


def test_all_history_files_link_without_reencoding(tmp_path, monkeypatch):
    engine, records, source, target = _archive(tmp_path)
    try:
        import natlang_neuralese.model.native_artifact as artifacts
        import natlang_neuralese.serve.store as store

        def unexpected(*args, **kwargs):
            raise AssertionError('verified immutable archive was serialized again')

        monkeypatch.setattr(artifacts, 'encode_native_tensor_artifact', unexpected)
        monkeypatch.setattr(store, 'encode_native_state', unexpected)
        out = tmp_path / 'continued'
        persist_artifacts(target, [records[-1].id], out / 'artifacts', out / 'states')
        originals = [path for path in source.rglob('*') if path.is_file()]
        assert originals
        for path in originals:
            linked = out / path.relative_to(source)
            assert linked.stat().st_ino == path.stat().st_ino
            assert linked.read_bytes() == path.read_bytes()
    finally:
        target.store.close()
        engine.store.close()


@pytest.mark.parametrize('change', ['removed', 'replaced', 'edited', 'symlink', 'directory', 'cross-device'])
def test_unavailable_or_changed_archive_uses_original_admitted_values(tmp_path, monkeypatch, change):
    engine, records, source, target = _archive(tmp_path)
    try:
        expected = {path.relative_to(source): path.read_bytes() for path in source.rglob('*') if path.is_file()}
        if change == 'cross-device':
            import natlang_neuralese.tune.archive_links as links

            def different_device(*args, **kwargs):
                raise OSError(errno.EXDEV, 'different device')

            monkeypatch.setattr(links.os, 'link', different_device)
        else:
            for path in list(source.rglob('*')):
                if not path.is_file():
                    continue
                if change == 'removed':
                    path.unlink()
                elif change in ('symlink', 'directory'):
                    alternate = path.with_suffix('.preserved')
                    path.rename(alternate)
                    if change == 'symlink':
                        path.symlink_to(alternate)
                    else:
                        path.mkdir()
                elif change == 'replaced':
                    temporary = path.with_suffix('.changed')
                    temporary.write_bytes(b'changed source bytes')
                    os.replace(temporary, path)
                else:
                    path.write_bytes(b'changed source bytes')
        out = tmp_path / 'continued'
        persist_artifacts(target, [records[-1].id], out / 'artifacts', out / 'states')
        assert {path.relative_to(out): path.read_bytes() for path in out.rglob('*') if path.is_file()} == expected
    finally:
        target.store.close()
        engine.store.close()


def test_own_link_ctime_does_not_force_next_round_serialization(tmp_path, monkeypatch):
    from natlang_neuralese.tune.archive_links import read_archive, remember, link_verified
    source = tmp_path / 'original.nnt'
    source.write_bytes(b'already validated by caller')
    body, reference = read_archive(source)
    engine = SimpleNamespace()
    remember(engine, 'fixture', reference)
    assert link_verified(engine, 'fixture', tmp_path / 'round-one.nnt')
    assert link_verified(engine, 'fixture', tmp_path / 'round-two.nnt')
    assert (tmp_path / 'round-two.nnt').read_bytes() == body
