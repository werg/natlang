import hashlib

import pytest

from natlang_neuralese.common import hdd_guard


@pytest.fixture
def hdd(tmp_path, monkeypatch):
    prefix = tmp_path / 'external'
    prefix.mkdir()
    monkeypatch.setenv('NATLANG_HDD_PREFIX', str(prefix))
    monkeypatch.setenv('NATLANG_STAGED_INPUTS', str(tmp_path / 'staged'))
    hdd_guard.RECORDS.clear()
    return prefix


def test_nvme_input_untouched_and_hdd_input_staged_with_hash(hdd, tmp_path):
    nvme = tmp_path / 'a.jsonl'
    nvme.write_text('x')
    assert hdd_guard.stage_input(nvme, 'bad') == str(nvme)
    source = hdd / 'd.jsonl'
    source.write_text('data')
    link = tmp_path / 'link.jsonl'
    link.symlink_to(source)
    digest = hashlib.sha256(b'data').hexdigest()
    staged = hdd_guard.stage_input(link, digest)
    assert staged == str(tmp_path / 'staged' / digest / 'd.jsonl')
    assert open(staged).read() == 'data' and hdd_guard.RECORDS[0]['declared_hash']
    assert hdd_guard.stage_input(link) == staged  # undeclared hash is computed


def test_hash_mismatch_refused_and_out_dir_must_be_nvme(hdd, tmp_path):
    source = hdd / 'd.jsonl'
    source.write_text('data')
    with pytest.raises(ValueError, match='SHA-256'):
        hdd_guard.stage_input(source, '0' * 64)
    with pytest.raises(ValueError, match='external HDD'):
        hdd_guard.require_nvme(hdd / 'run')
    hdd_guard.require_nvme(tmp_path / 'run')


def test_noop_without_hdd(tmp_path, monkeypatch):
    monkeypatch.setenv('NATLANG_HDD_PREFIX', str(tmp_path / 'absent'))
    assert not hdd_guard.on_hdd(tmp_path / 'x')
    hdd_guard.require_nvme(tmp_path / 'absent' / 'run')


def test_auto_hdd_root_on_nonrotating_device_is_not_staged(tmp_path, monkeypatch):
    monkeypatch.delenv('NATLANG_HDD_PREFIX', raising=False)
    monkeypatch.setattr(hdd_guard, 'resolve', lambda name: tmp_path)
    monkeypatch.setattr(hdd_guard, '_rotational', lambda path: False)
    assert hdd_guard.hdd_prefix() is None
    assert not hdd_guard.on_hdd(tmp_path / 'run')
    # Explicit selection remains authoritative for development and real HDDs.
    monkeypatch.setenv('NATLANG_HDD_PREFIX', str(tmp_path))
    assert hdd_guard.on_hdd(tmp_path / 'run')


def test_stage_input_args_pass_staged_path(hdd, tmp_path):
    from natlang_neuralese.train import recipe
    source = hdd / 'graphs.jsonl'
    source.write_text('t')
    digest = hashlib.sha256(b't').hexdigest()
    kind = 'native_state_sft'
    assert 'native_graphs' in hdd_guard.STAGED_ROLES
    assert 'text_data' not in hdd_guard.STAGED_ROLES
    args = recipe.stage_input_args({'native_graphs': {'path': str(source), 'sha256': digest}}, kind)
    assert args == ['--native-graphs', str(tmp_path / 'staged' / digest / 'graphs.jsonl')]


def test_native_checkpoint_is_read_in_place_while_graph_corpus_is_staged(hdd, tmp_path):
    """The native checkpoint is loaded once in place; graph corpora are staged for repeated reads."""
    from natlang_neuralese.train import recipe
    state = hdd / 'checkpoint.pt'
    state.write_bytes(b'state')
    graphs = hdd / 'graphs.jsonl'
    graphs.write_text('g')
    assert hdd_guard.input_path('continue_from', state) == str(state)
    assert not (tmp_path / 'staged').exists()
    assert all(r['read_from_hdd'] for r in hdd_guard.RECORDS) and len(hdd_guard.RECORDS) == 1
    staged = recipe._staged_inputs({'continue_from': {'path': str(state)}, 'native_graphs': {'path': str(graphs)}})
    assert staged['continue_from']['path'] == str(state)
    assert staged['native_graphs']['path'].startswith(str(tmp_path / 'staged'))


def test_the_staged_cache_is_bounded_least_recently_used_first(hdd, tmp_path, monkeypatch):
    import os
    import time
    monkeypatch.setenv('NATLANG_STAGED_INPUTS_MAX_GB', str(3.5 * 1024 / 2 ** 30))  # room for three 1 KiB files
    paths = []
    for name in 'abc':
        source = hdd / f'{name}.jsonl'
        source.write_bytes(name.encode() * 1024)
        paths.append(hdd_guard.stage_input(source))
        time.sleep(0.01)
    os.utime(paths[0], (time.time() + 5, time.time() + 5))  # a used recently
    source = hdd / 'd.jsonl'
    source.write_bytes(b'd' * 1024)
    hdd_guard.stage_input(source)
    assert os.path.exists(paths[0]) and not os.path.exists(paths[1])
