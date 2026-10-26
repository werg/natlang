import json
from pathlib import Path

import pytest

from test_native_foundation_recovery import case
from natlang_neuralese.common.hashing import sha256_file_hex
from natlang_neuralese.train.native_foundation_recovery import restore_declaration


def revise(proof, role, update):
    path=Path(proof[role]['path'])
    value=json.loads(path.read_text())
    update(value)
    path.write_text(json.dumps(value))
    proof[role]['sha256']=sha256_file_hex(path)


def test_mellum_checkpoint_identity_uses_the_same_recovery_mechanism(case):
    path,proof,kwargs=case
    kwargs['model_config']['architecture']='mellum'
    revise(proof,'historical_training_metadata',lambda value:value['model_config'].update(architecture='mellum'))
    revise(proof,'historical_checkpoint_verification',lambda value:value.update(architecture='mellum'))
    path.write_text(json.dumps(proof))
    assert restore_declaration(path,**kwargs)['qualification_granted'] is False


@pytest.mark.parametrize('same_path',[True,False])
def test_lfm_metadata_export_uses_its_exact_recorded_checkpoint_path(case,same_path):
    path,proof,kwargs=case
    def metadata(value):
        value.pop('checkpoint_identity')
        value['checkpoint_path']='/salvaged/checkpoint.pt'
    revise(proof,'historical_training_metadata',metadata)
    revise(proof,'historical_checkpoint_verification',lambda value:value.update(
        path='/salvaged/checkpoint.pt'if same_path else'/other/checkpoint.pt'))
    path.write_text(json.dumps(proof))
    if same_path:
        assert restore_declaration(path,**kwargs)['new_exposure']==0
    else:
        with pytest.raises(ValueError,match='metadata and checkpoint verification paths differ'):
            restore_declaration(path,**kwargs)
