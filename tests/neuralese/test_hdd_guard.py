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


def test_stage_input_args_pass_staged_path(hdd, tmp_path):
    from natlang_neuralese.train import recipe
    source = hdd / 'text.jsonl'
    source.write_text('t')
    digest = hashlib.sha256(b't').hexdigest()
    kind = next(k for k, h in recipe.HANDLERS.items() if 'text_data' in h['required_inputs'] | h['optional_inputs'])
    args = recipe.stage_input_args({'text_data': {'path': str(source), 'sha256': digest}}, kind)
    assert args == ['--text-data', str(tmp_path / 'staged' / digest / 'text.jsonl')]


def test_load_once_state_is_read_from_the_hdd_and_only_data_is_staged(hdd, tmp_path):
    """A continue_from / heads file is read once at startup: read in place (recorded read_from_hdd), never copied to
    the NVMe; data roles are staged."""
    from natlang_neuralese.train import recipe
    state = hdd / 'checkpoint.pt'
    state.write_bytes(b'state')
    text = hdd / 'text.jsonl'
    text.write_text('t')
    for role in ('continue_from', 'continue_identity', 'continue_monitor_reference', 'heads', 'warmup_checkpoint'):
        assert hdd_guard.input_path(role, state) == str(state)
    assert not (tmp_path / 'staged').exists()
    assert all(r['read_from_hdd'] for r in hdd_guard.RECORDS) and len(hdd_guard.RECORDS) == 5
    staged = recipe._staged_inputs({'continue_from': {'path': str(state)}, 'text_data': {'path': str(text)}})
    assert staged['continue_from']['path'] == str(state)
    assert staged['text_data']['path'].startswith(str(tmp_path / 'staged'))


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
