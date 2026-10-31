import copy
import pytest
from natlang_neuralese.common.hashing import canonical_json_sha256_hex
from natlang_neuralese.model.native_precision import base_restore_transition,same_source_precision_identity
from natlang_neuralese.train.native_resident import same_geometry_identity


def test_restore_receipt_migrates_out_of_identity_only_when_exact():
    inventory={'config.json':'a'*64,'tokenizer.json':'b'*64,'model.safetensors':'c'*64}
    digest=canonical_json_sha256_hex(inventory)
    receipt={'schema':'natlang.native-config-only-checkpoint-restore/1','mode':'checkpoint-backbone-state+verified-config-tokenizer',
             'pinned_base_inventory_sha256':digest,'verified_local_files':{'config.json':'a'*64,'tokenizer.json':'b'*64}}
    before={'base_inventory':inventory,'base_inventory_sha256':digest,'base_restore':receipt,'dtype':'torch.bfloat16'}
    after={'base_inventory':inventory,'base_inventory_sha256':digest,'dtype':'torch.bfloat16'}
    transition=base_restore_transition(before,after)
    assert transition['receipt_location']=='operational-metadata'
    assert same_source_precision_identity(before,after)
    assert same_geometry_identity({'model_config':before},{'model_config':after})
    invalid=copy.deepcopy(before);invalid['base_restore']['verified_local_files']['config.json']='f'*64
    with pytest.raises(ValueError,match='pinned base inputs'):
        base_restore_transition(invalid,after)
    assert not same_geometry_identity({'model_config':invalid},{'model_config':after})
    altered=copy.deepcopy(after);altered['dtype']='torch.float32'
    assert not same_geometry_identity({'model_config':before},{'model_config':altered})
