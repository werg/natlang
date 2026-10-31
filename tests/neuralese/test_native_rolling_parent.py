import os

import pytest

from natlang_neuralese.train.native_state_sft import _validate_rolling_checkpoint_parent


def test_rolling_resume_binds_actual_parent(tmp_path):
    slot = tmp_path / 'checkpoint.pt'
    other = tmp_path / 'other.pt'
    other.write_text('old parent')
    _validate_rolling_checkpoint_parent(slot, other)
    slot.write_text('rolling parent')
    _validate_rolling_checkpoint_parent(slot, None)
    _validate_rolling_checkpoint_parent(slot, slot)
    same_inode = tmp_path / 'same.pt'
    os.link(slot, same_inode)
    _validate_rolling_checkpoint_parent(slot, same_inode)
    with pytest.raises(ValueError, match='differs from existing rolling'):
        _validate_rolling_checkpoint_parent(slot, other)
    with pytest.raises(ValueError, match='cannot verify'):
        _validate_rolling_checkpoint_parent(slot, tmp_path / 'missing.pt')
