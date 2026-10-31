import json
import pytest
from natlang_neuralese.common.hashing import canonical_json_sha256_hex,sha256_file_hex
from natlang_neuralese.train.native_packed_capacity import policy_capacity_baseline


def test_policy_capacity_baseline_requires_actual_execution_contract(tmp_path):
    source={'path':'/source','sha256':'a'*64};objective={'sha256':'b'*64}
    capacity={'passed':True,'objective':objective,'source':source,'cotangent_mode':'plain',
              'production_packing':True,'retained_accumulation_gradients':True,'qat_point':None}
    proof={'schema':'natlang.native-policy-capacity-proof/1','passed':True,'parent_checkpoint_sha256':'c'*64,
           'packing_limits':{'forward_tokens':100,'kv_slots':120},'capacity':capacity}
    p=tmp_path/'proof.json';p.write_text(json.dumps(proof));ref={'path':str(p),'sha256':sha256_file_hex(p),'proof':proof,'proof_sha256':canonical_json_sha256_hex(proof)}
    options={'checkpoint_sha256':'c'*64,'objective_binding':objective,'source_binding':source,'cotangent_mode':'plain','retained_gradients':True}
    result=policy_capacity_baseline(ref,**options);assert result['forward_tokens']==100 and result['kv_slots']==120
    for key,value in [('checkpoint_sha256','d'*64),('cotangent_mode','vjp'),('retained_gradients',False),('objective_binding',{'sha256':'e'*64})]:
        with pytest.raises(ValueError,match='actual first-update execution'):
            policy_capacity_baseline(ref,**{**options,key:value})
