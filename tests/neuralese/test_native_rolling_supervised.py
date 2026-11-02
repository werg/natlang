import hashlib
from pathlib import Path

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
    import torch
    from natlang_neuralese.model.native_model import CHECKPOINT_SCHEMA
    from natlang_neuralese.train.native_continuation import TRAINING_SCHEMA
    torch.save({'schema':CHECKPOINT_SCHEMA,'model_config':{'architecture':'fixture'},
        'training':{'schema':TRAINING_SCHEMA,'optimizer':{'moments':torch.tensor([1.])},
                    'rng':{'seed':torch.tensor([3])},'global_step':1,'step':1,
                    'identity':{'objective':None},'options':{'stage_id':'sft','epochs':1.},
                    'source_row_order':['row'],'source_row_bindings':{'row':'binding'},
                    'required_visits':{'row':1},'exposure_counts':{'row':1},
                    'source_receipts':[{'sha256':'train'},{'sha256':'held'}]}},slot)
    state = {'checkpoint':str(old)}
    control = {'schema':'natlang.rolling-sft-resume/1', 'checkpoint_path':str(slot),
        'checkpoint_sha256':sha(slot), 'previous_checkpoint_path':str(old),
        'previous_checkpoint_sha256':sha(old)}
    config = {'rolling_sft_resume':control}
    parent, record = rolling_resume_parent(config, state, output)
    assert parent == str(slot.resolve())
    assert record['previous_checkpoint_sha256'] == sha(old)
    import json
    custody=json.loads(Path(record['custody_snapshot']['path']).read_text())
    assert custody['checkpoint']['sha256']==sha(slot) and custody['sft_frames']['sft']['exposure_counts']=={'row':1}
    assert state['checkpoint'] == str(old)
    slot.write_bytes(b'newer update')
    with pytest.raises(ValueError, match='checkpoint changed'):
        rolling_resume_parent(config, state, output)
    control['checkpoint_sha256'] = sha(slot)
    old.write_bytes(b'replaced')
    with pytest.raises(ValueError, match='parent changed'):
        rolling_resume_parent(config, state, output)
