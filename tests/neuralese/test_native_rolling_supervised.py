import hashlib

import pytest

from natlang_neuralese.rl.native_supervised import rolling_resume_parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_rolling_control_binds_slot_and_preserves_old_parent(tmp_path):
    old = tmp_path / 'parent.pt'
    old.write_bytes(b'previous')
    output = tmp_path / 'learner'
    output.mkdir()
    slot = output / 'checkpoint.pt'
    slot.write_bytes(b'earned update')
    state = {'checkpoint':str(old)}
    control = {'schema':'natlang.rolling-sft-resume/1', 'checkpoint_path':str(slot),
        'checkpoint_sha256':sha(slot), 'previous_checkpoint_path':str(old),
        'previous_checkpoint_sha256':sha(old)}
    config = {'rolling_sft_resume':control}
    parent, record = rolling_resume_parent(config, state, output)
    assert parent == str(slot.resolve())
    assert record['previous_checkpoint_sha256'] == sha(old)
    assert state['checkpoint'] == str(old)
    slot.write_bytes(b'newer update')
    with pytest.raises(ValueError, match='checkpoint changed'):
        rolling_resume_parent(config, state, output)
    control['checkpoint_sha256'] = sha(slot)
    old.write_bytes(b'replaced')
    with pytest.raises(ValueError, match='parent changed'):
        rolling_resume_parent(config, state, output)
