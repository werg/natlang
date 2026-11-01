"""Resume bytes advance without changing a homogeneous coordinator's seed."""
from types import SimpleNamespace
import pytest
from natlang_neuralese.common.hashing import canonical_json_sha256_hex,sha256_file_hex_cached
from natlang_neuralese.train.native_diloco import IDENTITY_SCHEMA,rolling_resume_identity


def controls(tmp_path):
    slot=tmp_path/'checkpoint.pt';slot.write_bytes(b'earned full state')
    original={'schema':IDENTITY_SCHEMA,'parent_checkpoint_sha256':'a'*64,
        'trainer_argv':['--continue-from','original-seed.pt','--learning-rate','0.001'],'diloco':{'workers':1}}
    current={**original,'parent_checkpoint_sha256':sha256_file_hex_cached(slot),
        'trainer_argv':['--continue-from',str(slot),'--learning-rate','0.002']}
    manifest={'identity':original};manifest['fingerprint']=canonical_json_sha256_hex(manifest)
    return slot,original,current,manifest


def test_rolling_identity_retains_seed_without_erasing_other_argument_changes(tmp_path):
    slot,original,current,manifest=controls(tmp_path)
    result=rolling_resume_identity(current,original,args=SimpleNamespace(out=tmp_path,continue_from=slot),
        saved_state={'fingerprint':manifest['fingerprint']},manifest=manifest)
    assert result['parent_checkpoint_sha256']==original['parent_checkpoint_sha256']
    assert result['trainer_argv']==['--continue-from','original-seed.pt','--learning-rate','0.002']
    assert current['trainer_argv'][1]==str(slot)


@pytest.mark.parametrize('bad',['checkpoint','fingerprint','manifest'])
def test_rolling_identity_rejects_wrong_checkpoint_or_coordinator(tmp_path,bad):
    slot,original,current,manifest=controls(tmp_path)
    parent=slot;fingerprint=manifest['fingerprint']
    if bad=='checkpoint':parent=tmp_path/'other.pt';parent.write_bytes(b'wrong state')
    elif bad=='fingerprint':fingerprint='b'*64
    else:manifest={**manifest,'tampered':True}
    with pytest.raises(ValueError,match='rolling checkpoint|another coordinator'):
        rolling_resume_identity(current,original,args=SimpleNamespace(out=tmp_path,continue_from=parent),
            saved_state={'fingerprint':fingerprint},manifest=manifest)
