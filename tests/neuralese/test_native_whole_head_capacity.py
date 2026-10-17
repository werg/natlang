"""Exact reviewed shared HEAD can retain ordinary capacity, never policy/profile capacity."""
import copy
import json
from pathlib import Path
import pytest
from test_native_capacity_continuation import mechanism,fixture,invoke
from natlang_neuralese.train.native_resident import binding
from natlang_neuralese.common.hashing import canonical_json_sha256_hex as digest

PROOF=Path(__file__).resolve().parents[2]/'plans/neuralese/shared-state/WHOLE_HEAD_SOURCE_PROFILE_PROOF_20261017.json'


def review():
    witness=json.loads(PROOF.read_text())
    return {'optimizer_policy':mechanism.WHOLE_HEAD_SOURCE_POLICY,
        'old_code':witness['inventories']['source'],'new_code':witness['inventories']['target'],
        'identity':{'objective':None,'model_config':{'architecture':'lfm2','native_linear_precision':'bf16x6'}},
        'fresh_policy_capacity_required':True,'whole_head_arithmetic_proof':binding(PROOF)}


def test_pinned_complete_witness_matches_both_full_inventories_and_actual_reference_scope():
    value=review();mechanism._whole_head_source_contract(value)
    witness=json.loads(PROOF.read_text())
    assert len(value['new_code'])==296
    changes={key for key in value['old_code'].keys()|value['new_code'].keys() if value['old_code'].get(key)!=value['new_code'].get(key)}
    assert mechanism.allowed_changes(mechanism.WHOLE_HEAD_SOURCE_POLICY)-changes=={'train/native_continuation.py'}
    assert changes<=mechanism.allowed_changes(mechanism.WHOLE_HEAD_SOURCE_POLICY)
    assert witness['checks']['maple_module_exact_after_declared_norm_dispatch'] is True
    assert witness['checks']['shared_state_module_exact_after_explicit_profiles_and_identity_metadata'] is True
    assert witness['source_index_proof']['qualification']=='source-metadata-revalidation-20261016'
    assert set(witness['actual_references'])=={'cpu','cuda'}


def test_every_old_or_new_module_is_pinned_including_unchanged_kernels():
    value=review()
    for role in ('old_code','new_code'):
        for key in value[role]:
            mutated=copy.deepcopy(value);mutated[role][key]='0'*64
            with pytest.raises(ValueError,match='full inventories'):
                mechanism._whole_head_source_contract(mutated)


@pytest.mark.parametrize('change',['objective','norm','attention','storage','policy','witness','extra','missing'])
def test_profile_objective_and_unknown_inventory_changes_cannot_inherit_capacity(change):
    value=review()
    if change=='objective':value['identity']['objective']={'kind':'native-policy'}
    elif change=='norm':value['identity']['model_config']['native_norm_execution']='stable-v1'
    elif change=='attention':value['identity']['model_config']['native_attention_precision']='tf32x3'
    elif change=='storage':value['identity']['model_config']['native_ordinary_precision']='fp32-norms-v1'
    elif change=='policy':value['fresh_policy_capacity_required']=False
    elif change=='witness':value['whole_head_arithmetic_proof']['sha256']='0'*64
    elif change=='extra':value['new_code']['model/unknown.py']='0'*64
    else:value['new_code'].pop('rl/native_runtime_handoff.py')
    with pytest.raises(ValueError):mechanism._whole_head_source_contract(value)


def chain(tmp_path,monkeypatch):
    old,new,result,proof=fixture(tmp_path)
    pinned=review();old_code=pinned['old_code'];new_code=pinned['new_code']
    # Actual file-inventory checking is already covered by the shared verifier;
    # use the pinned complete inventories to isolate recursive certificate flow.
    monkeypatch.setattr(mechanism,'code_inventory',lambda root:old_code if root==old else new_code)
    measured=json.loads(proof.read_text());measured.update(identity=pinned['identity'],runtime_code_sha256=digest(old_code))
    measured['identity']['runtime']={'total_memory_bytes':20<<30}
    proof.write_text(json.dumps(measured))
    certpath=Path(result['source_certificate']['path']);cert=json.loads(certpath.read_text())
    cert.update(identity=measured['identity'],runtime_code_sha256=digest(old_code),evidence=binding(proof));certpath.write_text(json.dumps(cert))
    result.update(pinned);result.update(identity=measured['identity'],source_certificate=binding(certpath),
        changes={name:{'old':old_code.get(name),'new':new_code.get(name)}
                 for name in old_code.keys()|new_code.keys() if old_code.get(name)!=new_code.get(name)},
        optimizer_state_unchanged=True,metadata_extra_bytes=64<<20)
    return old,new,result,proof


@pytest.mark.parametrize('tamper',[False,True])
def test_recursive_whole_head_evidence_preserves_actual_origin_and_rechecks_policy_scope(tmp_path,monkeypatch,tamper):
    old,new,value,_=chain(tmp_path,monkeypatch)
    output=invoke(tmp_path,old,new,value);cert=json.loads(output.read_text());evidence=json.loads(Path(cert['evidence']['path']).read_text())
    assert evidence['new_parent_measured'] is False and evidence['optimizer_updates']==0
    assert evidence['measured_parent_checkpoint_sha256']=='a'*64 and evidence['continued_parent_checkpoint_sha256']=='b'*64
    assert 'fresh policy measurement required' in cert['scope']
    assert invoke(tmp_path,old,new,value)==output
    assert mechanism._proof_cases(cert['evidence'],identity=value['identity'],code_sha=digest(value['new_code']),parent='b'*64)
    if tamper:
        value['fresh_policy_capacity_required']=False
        path=Path(evidence['review']['path']);path.write_text(json.dumps(value));evidence['review']=binding(path)
        path=Path(cert['evidence']['path']);path.write_text(json.dumps(evidence))
        with pytest.raises(ValueError,match='ordinary source-v1'):
            mechanism._proof_cases(binding(path),identity=value['identity'],code_sha=digest(value['new_code']),parent='b'*64)


@pytest.mark.parametrize('change',['device','host','reserve','layout','identity','allowance','optimizer'])
def test_whole_head_retains_measured_limits_identity_and_all_memory_guards(tmp_path,monkeypatch,change):
    old,new,value,_=chain(tmp_path,monkeypatch)
    if change=='device':value['device_memory_bytes']=8<<30
    elif change=='host':value['host_memory_limit_bytes']=4<<30
    elif change=='reserve':value['reserve_bytes']=(1<<30)-1
    elif change=='layout':value['parameter_layout_unchanged']=False
    elif change=='identity':value['identity']={**value['identity'],'checkpoint_input_offload':True}
    elif change=='optimizer':value['optimizer_state_unchanged']=False
    else:value['metadata_extra_bytes']=(64<<20)-1
    with pytest.raises(ValueError):invoke(tmp_path,old,new,value)


def test_prior_reviewed_inventory_remains_verifiable_but_cannot_mix_witnesses():
    value=review();old=PROOF.with_name('WHOLE_HEAD_SOURCE_PROFILE_PROOF_293E2_20261017.json')
    value.update(new_code=json.loads(old.read_text())['inventories']['target'],whole_head_arithmetic_proof=binding(old))
    mechanism._whole_head_source_contract(value)
    value['whole_head_arithmetic_proof']=binding(PROOF)
    with pytest.raises(ValueError,match='exact arithmetic'):
        mechanism._whole_head_source_contract(value)


def test_378_inventory_requires_its_own_witness_and_exact_continuation_module():
    proof=PROOF.with_name('WHOLE_HEAD_SOURCE_PROFILE_PROOF_378FF_20261017.json')
    witness=json.loads(proof.read_text());value=review()
    value.update(new_code=witness['inventories']['target'],whole_head_arithmetic_proof=binding(proof))
    mechanism._whole_head_source_contract(value)
    changed={name for name in value['old_code'].keys()|value['new_code'].keys()
             if value['old_code'].get(name)!=value['new_code'].get(name)}
    assert changed==mechanism.allowed_changes(mechanism.WHOLE_HEAD_SOURCE_POLICY)
    value['whole_head_arithmetic_proof']=binding(PROOF)
    with pytest.raises(ValueError,match='exact arithmetic'):
        mechanism._whole_head_source_contract(value)
    value['whole_head_arithmetic_proof']=binding(proof)
    value['new_code']['train/native_continuation.py']='0'*64
    with pytest.raises(ValueError,match='full inventories'):
        mechanism._whole_head_source_contract(value)
