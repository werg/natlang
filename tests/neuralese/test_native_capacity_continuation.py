"""Inherited memory evidence must not become a fictitious new qualification."""
import copy
import json
from pathlib import Path

import pytest

from natlang_neuralese.train.native_resident import binding, code_inventory, geometry_capacity
from natlang_neuralese.common.hashing import canonical_json_sha256_hex as digest

from natlang_neuralese.train import native_capacity_continuation as mechanism


def fixture(tmp_path):
    old, new = tmp_path/'old', tmp_path/'new'
    for root in (old, new):
        (root/'train').mkdir(parents=True)
        (root/'model').mkdir()
        (root/'model/forward.py').write_text('unchanged forward\n')
        (root/'train/optim.py').write_text('old routing\n')
    (new/'train/optim.py').write_text('reviewed corrected routing\n')
    codes = [code_inventory(root) for root in (old, new)]
    identity = {'checkpoint_input_offload': False, 'runtime': {'total_memory_bytes': 20 << 30}}
    cases = [{'id': 'actual maximum', 'status': 'passed', 'forward_tokens': 1234, 'kv_slots': 2345,
        'cuda_peak_allocated_bytes': 8 << 30, 'host_peak': {'cgroup_current_bytes': 4 << 30}}]
    proof = tmp_path/'proof.json'
    proof.write_text(json.dumps({'schema': 'natlang.native-memory-qualification/1',
        'passed': True, 'identity': identity, 'runtime_code_sha256': digest(codes[0]),
        'parent_checkpoint_sha256': 'a'*64, 'optimizer_updates': 0,
        'numerical_reference': {'passed': True}, 'capacity': {'passed': True, 'cases': cases}}))
    cert = tmp_path/'source-capacity.json'
    cert.write_text(json.dumps({'schema': 'natlang.native-resident-geometry-qualification/1',
        'passed': True, 'identity': identity, 'runtime_code_sha256': digest(codes[0]),
        'limits': {'forward_tokens': 1234, 'kv_slots': 2345}, 'evidence': binding(proof)}))
    review = {'schema': 'natlang.native-capacity-continuation-review/1', 'approved': True,
        'old_code': codes[0], 'new_code': codes[1], 'changes': {
            'train/optim.py': {'old': codes[0]['train/optim.py'], 'new': codes[1]['train/optim.py']}},
        'identity': identity, 'source_certificate': binding(cert),
        'old_parent_checkpoint_sha256': 'a'*64, 'new_parent_checkpoint_sha256': 'b'*64,
        'parameter_layout_unchanged': True, 'optimizer_policy': mechanism.POLICY,
        'ordinary_parameters': [{'name': 'learned role rows', 'shape': [4, 2304], 'dtype': 'torch.float32'}],
        'device_memory_bytes': 20 << 30, 'host_memory_limit_bytes': 16 << 30, 'reserve_bytes': 1 << 30}
    return old, new, review, proof


def invoke(tmp_path, old, new, review):
    path = tmp_path/'review.json'
    path.write_text(json.dumps(review))
    return mechanism.continue_capacity(binding(path), old_package=old, new_package=new, out=tmp_path/'out')


def test_mechanical_continuation_retains_measured_parent_and_bounds_extra_memory(tmp_path):
    old, new, review, _ = fixture(tmp_path)
    result = invoke(tmp_path, old, new, review)
    cert = json.loads(result.read_text())
    evidence = json.loads(Path(cert['evidence']['path']).read_text())
    assert evidence['new_parent_measured'] is False
    assert evidence['measured_parent_checkpoint_sha256'] == 'a'*64
    assert evidence['continued_parent_checkpoint_sha256'] == 'b'*64
    assert evidence['extra_bytes_per_memory_domain'] == 64*4*2304
    assert geometry_capacity(binding(result), identity=review['identity'], code=code_inventory(new)) == cert['limits']
    assert 'new parent not measured' in cert['scope']
    assert invoke(tmp_path, old, new, review) == result


@pytest.mark.parametrize('mutation', [None, 'old-source', 'new-source', 'unbound-policy'])
def test_saved_recipe_metadata_policy_accepts_only_exact_reviewed_trainer(mutation):
    old, new = mechanism.LINEAGE_METADATA_COMPOSITION
    name = mechanism.LINEAGE_METADATA_FILE
    review = {'optimizer_policy': mechanism.LINEAGE_METADATA_POLICY,
              'old_code': {name: old}, 'new_code': {name: new},
              'optimizer_state_unchanged': True, 'metadata_extra_bytes': 64 << 20,
              'ordinary_parameters': [{'name': 'learned rows', 'shape': [4, 2304], 'dtype': 'torch.float32'}]}
    if mutation == 'old-source': review['old_code'][name] = 'a' * 64
    if mutation == 'new-source': review['new_code'][name] = 'b' * 64
    if mutation == 'unbound-policy': review['optimizer_state_unchanged'] = False
    if mutation:
        with pytest.raises(ValueError):
            mechanism._lineage_metadata_contract(review)
            mechanism._memory_allowance(review)
    else:
        mechanism._lineage_metadata_contract(review)
        assert mechanism._memory_allowance(review) == 64 << 20
        assert mechanism.allowed_changes(mechanism.LINEAGE_METADATA_POLICY) == {name}


@pytest.mark.parametrize('mutation', ['unapproved', 'offload', 'unreviewed', 'forward',
    'device', 'host', 'reserve', 'layout', 'parameters', 'source-parent', 'source-mutated', 'numerics', 'nonzero-update'])
def test_changed_execution_unbound_proof_or_memory_shortfall_cannot_inherit_capacity(tmp_path, mutation):
    old, new, review, proof = fixture(tmp_path)
    if mutation == 'unapproved': review['approved'] = False
    elif mutation == 'offload': review['identity'] = {**review['identity'], 'checkpoint_input_offload': True}
    elif mutation == 'unreviewed': (new/'train/optim.py').write_text('unreviewed edit\n')
    elif mutation == 'forward':
        (new/'model/forward.py').write_text('changed forward\n')
        review['new_code'] = code_inventory(new)
        review['changes']['model/forward.py'] = {'old': review['old_code']['model/forward.py'],
            'new': review['new_code']['model/forward.py']}
    elif mutation == 'device': review['device_memory_bytes'] = 8 << 30
    elif mutation == 'host': review['host_memory_limit_bytes'] = 4 << 30
    elif mutation == 'reserve': review['reserve_bytes'] = 0
    elif mutation == 'layout': review['parameter_layout_unchanged'] = False
    elif mutation == 'parameters': review['ordinary_parameters'] *= 2
    elif mutation == 'source-parent': review['old_parent_checkpoint_sha256'] = 'c'*64
    elif mutation == 'source-mutated': proof.write_text('{}')
    else:
        value = json.loads(proof.read_text())
        if mutation == 'numerics': value['numerical_reference']['passed'] = False
        else: value['optimizer_updates'] = 1
        proof.write_text(json.dumps(value))
        cert_path = Path(review['source_certificate']['path'])
        cert = json.loads(cert_path.read_text()); cert['evidence'] = binding(proof)
        cert_path.write_text(json.dumps(cert)); review['source_certificate'] = binding(cert_path)
    with pytest.raises(ValueError):
        invoke(tmp_path, old, new, review)
    assert not (tmp_path/'out').exists()


def test_combined_evidence_requires_exact_transitive_actual_case_union(tmp_path):
    old, new, review, proof = fixture(tmp_path)
    value = json.loads(proof.read_text())
    combined = tmp_path/'combined.json'
    combined.write_text(json.dumps({**value, 'schema': 'natlang.native-memory-combined-capacity/1',
        'proofs': [binding(proof)], 'capacity': {'cases': copy.deepcopy(value['capacity']['cases'])}}))
    cert_path = Path(review['source_certificate']['path'])
    cert = json.loads(cert_path.read_text()); cert['evidence'] = binding(combined)
    cert_path.write_text(json.dumps(cert)); review['source_certificate'] = binding(cert_path)
    assert invoke(tmp_path, old, new, review).is_file()
    value = json.loads(combined.read_text()); value['capacity']['cases'][0]['kv_slots'] += 1
    combined.write_text(json.dumps(value))
    cert['evidence'] = binding(combined); cert_path.write_text(json.dumps(cert))
    review['source_certificate'] = binding(cert_path)
    with pytest.raises(ValueError, match='actual bound proofs'):
        invoke(tmp_path, old, new, review)


def metadata_chain(tmp_path):
    import shutil
    old, adam, review, _ = fixture(tmp_path)
    first=tmp_path/'first';first.mkdir()
    certificate=invoke(first,old,adam,review)
    new=tmp_path/'scheduled';shutil.copytree(adam,new)
    (new/'train/native_pending_sources.py').write_text('reviewed pending row metadata\n')
    continued=copy.deepcopy(review)
    continued.update(old_code=code_inventory(adam),new_code=code_inventory(new),
        source_certificate=binding(certificate),old_parent_checkpoint_sha256='b'*64,
        new_parent_checkpoint_sha256='c'*64,optimizer_policy=mechanism.METADATA_POLICY,
        optimizer_state_unchanged=True,metadata_extra_bytes=1<<30)
    continued['changes']={'train/native_pending_sources.py':{
        'old':None,'new':continued['new_code']['train/native_pending_sources.py']}}
    second=tmp_path/'second';second.mkdir()
    return second,adam,new,continued,certificate


@pytest.mark.parametrize('resident_change',[False,True])
def test_metadata_continuation_keeps_actual_origin_and_cumulative_memory(tmp_path,resident_change):
    phase,old,new,review,original=metadata_chain(tmp_path)
    if resident_change:
        (new/'train/native_resident.py').write_text('reviewed resident pending source obligations\n')
        review['new_code']=code_inventory(new)
        review['changes']['train/native_resident.py']={
            'old':None,'new':review['new_code']['train/native_resident.py']}
    output=invoke(phase,old,new,review)
    certificate=json.loads(output.read_text())
    result=json.loads(Path(certificate['evidence']['path']).read_text())
    assert result['measured_parent_checkpoint_sha256']=='a'*64
    assert result['source_parent_checkpoint_sha256']=='b'*64
    assert result['continued_parent_checkpoint_sha256']=='c'*64
    assert result['new_parent_measured'] is False
    assert result['extra_bytes_per_memory_domain']==1<<30
    assert result['measured_peaks'][0]['inherited_extra_bytes']==64*4*2304
    assert result['measured_peaks'][0]['device_bytes']==8<<30
    assert geometry_capacity(binding(output),identity=review['identity'],code=code_inventory(new))==certificate['limits']


@pytest.mark.parametrize('mutation',['lost-optimizer','small-allowance','chain-policy','chain-extra','chain-origin','chain-review-schema','chain-summary-policy','cumulative-headroom','new-model-code'])
def test_metadata_chain_rejects_changed_optimizer_unbound_math_and_lost_reserves(tmp_path,mutation):
    phase,old,new,review,original=metadata_chain(tmp_path)
    if mutation=='lost-optimizer':review['optimizer_state_unchanged']=False
    elif mutation=='small-allowance':review['metadata_extra_bytes']=1
    elif mutation=='cumulative-headroom':review['device_memory_bytes']=review['identity']['runtime']['total_memory_bytes']=10<<30
    elif mutation=='new-model-code':
        (new/'model/forward.py').write_text('changed math\n')
        review['new_code']=code_inventory(new)
        review['changes']['model/forward.py']={'old':review['old_code']['model/forward.py'],'new':review['new_code']['model/forward.py']}
    else:
        cert=json.loads(original.read_text());path=Path(cert['evidence']['path']);value=json.loads(path.read_text())
        if mutation=='chain-extra':value['extra_bytes_per_memory_domain']=0
        elif mutation=='chain-origin':value['measured_parent_checkpoint_sha256']='e'*64
        elif mutation=='chain-summary-policy':value['continuation_policy']='incorrect-policy'
        elif mutation=='chain-review-schema':
            child_path=Path(value['review']['path']);child=json.loads(child_path.read_text())
            child['schema']='incorrect-review-schema';child_path.write_text(json.dumps(child))
            value['review']=binding(child_path)
        else:value['runtime_code_sha256']='d'*64
        path.write_text(json.dumps(value));cert['evidence']=binding(path);original.write_text(json.dumps(cert))
        review['source_certificate']=binding(original)
    with pytest.raises(ValueError):invoke(phase,old,new,review)


@pytest.mark.parametrize('name',['train/native_source_preparation.py','train/native_cohort_schedule.py','train/native_operation_report.py',
    'data/native_source_tree.py','data/native_sources.py','model/shared_state.py','train/optim.py','train/losses.py'])
def test_metadata_helpers_require_review_and_leaf_parser_still_cannot_inherit(tmp_path,name):
    phase,old,new,review,_=metadata_chain(tmp_path)
    path=new/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('# reviewed CPU metadata helper\n')
    review['new_code']=code_inventory(new)
    review['changes'][name]={'old':review['old_code'].get(name),'new':review['new_code'][name]}
    if name in {'data/native_sources.py','model/shared_state.py','train/optim.py','train/losses.py'}:
        with pytest.raises(ValueError,match='unreviewed forward/runtime'):
            invoke(phase,old,new,review)
    else:
        result=invoke(phase,old,new,review)
        certificate=json.loads(result.read_text())
        assert 'new parent not measured' in certificate['scope']
        assert geometry_capacity(binding(result),identity=review['identity'],code=code_inventory(new))==certificate['limits']


@pytest.mark.parametrize('injected',[-(16<<30),0,1<<30])
def test_raw_measurement_cannot_forge_an_inherited_memory_allowance(tmp_path,injected):
    old,new,review,proof=fixture(tmp_path)
    value=json.loads(proof.read_text());value['capacity']['cases'][0]['inherited_extra_bytes']=injected
    proof.write_text(json.dumps(value))
    cert_path=Path(review['source_certificate']['path']);cert=json.loads(cert_path.read_text())
    cert['evidence']=binding(proof);cert_path.write_text(json.dumps(cert));review['source_certificate']=binding(cert_path)
    with pytest.raises(ValueError,match='raw measured cases'):
        invoke(tmp_path,old,new,review)


@pytest.mark.parametrize('policy',['metadata','loader-io'])
def test_loader_io_requires_its_separate_explicit_review_policy(tmp_path,policy):
    phase,old,new,review,_=metadata_chain(tmp_path)
    for name in ('checkpoint_policy.py','checkpoint_readahead.py'):
        (new/'train'/name).write_text('reviewed exact-mapping sequential OS hint\n')
    review['new_code']=code_inventory(new)
    review['changes']={name:{'old':review['old_code'].get(name),'new':review['new_code'].get(name)}
        for name in review['old_code'].keys()|review['new_code'].keys()
        if review['old_code'].get(name)!=review['new_code'].get(name)}
    if policy=='metadata':
        with pytest.raises(ValueError,match='unreviewed forward/runtime'):
            invoke(phase,old,new,review)
    else:
        review['optimizer_policy']=mechanism.LOADER_IO_POLICY
        result=invoke(phase,old,new,review)
        certificate=json.loads(result.read_text())
        evidence=json.loads(Path(certificate['evidence']['path']).read_text())
        assert evidence['continuation_policy']==mechanism.LOADER_IO_POLICY
        assert evidence['new_parent_measured'] is False
        # A later continuation must reverify the distinct I/O policy chain.
        again=phase/'later';again.mkdir()
        review.update(source_certificate=binding(result),old_code=code_inventory(new),
            old_parent_checkpoint_sha256='c'*64,new_parent_checkpoint_sha256='d'*64)
        later=phase/'next-package'
        import shutil
        shutil.copytree(new,later)
        (later/'train/native_pending_sources.py').write_text('next reviewed scheduling metadata\n')
        review['new_code']=code_inventory(later)
        review['changes']={'train/native_pending_sources.py':{
            'old':review['old_code']['train/native_pending_sources.py'],
            'new':review['new_code']['train/native_pending_sources.py']}}
        assert invoke(again,new,later,review).is_file()


@pytest.mark.parametrize('mutation', [None, 'old', 'new', 'extra', 'missing', 'objective', 'policy-gate'])
def test_frozen_policy_gate_continuation_is_exact_and_ordinary_sft_only(mutation):
    pins = mechanism.POLICY_GATE_SFT_COMPOSITION
    review = {'optimizer_policy': mechanism.POLICY_GATE_SFT_POLICY,
        'old_code': {name: pair[0] for name, pair in pins.items()},
        'new_code': {name: pair[1] for name, pair in pins.items()},
        'identity': {}, 'fresh_policy_capacity_required': True,
        'optimizer_state_unchanged': True, 'metadata_extra_bytes': 64 << 20,
        'ordinary_parameters': [{'name': 'learned rows', 'shape': [4, 2304], 'dtype': 'torch.float32'}]}
    name = 'rl/native_stage.py'
    if mutation == 'old': review['old_code'][name] = 'a'*64
    elif mutation == 'new': review['new_code'][name] = 'b'*64
    elif mutation == 'extra': review['new_code']['model/forward.py'] = 'c'*64
    elif mutation == 'missing': review['new_code'][name] = review['old_code'][name]
    elif mutation == 'objective': review['identity']['objective'] = {'schema': 'native-policy'}
    elif mutation == 'policy-gate': review['fresh_policy_capacity_required'] = False
    if mutation:
        with pytest.raises(ValueError): mechanism._policy_gate_sft_contract(review)
    else:
        mechanism._policy_gate_sft_contract(review)
        assert mechanism.allowed_changes(mechanism.POLICY_GATE_SFT_POLICY) == set(pins)
        assert mechanism._memory_allowance(review) == 64 << 20


def test_policy_gate_sft_contract_is_rechecked_transitively(tmp_path, monkeypatch):
    old, new, review, proof = fixture(tmp_path)
    pins = mechanism.POLICY_GATE_SFT_COMPOSITION
    old_code = {name: pair[0] for name, pair in pins.items()}
    new_code = {name: pair[1] for name, pair in pins.items()}
    monkeypatch.setattr(mechanism, 'code_inventory', lambda root: old_code if root == old else new_code)
    saved = json.loads(proof.read_text()); saved['runtime_code_sha256'] = digest(old_code)
    proof.write_text(json.dumps(saved))
    cert_path = Path(review['source_certificate']['path'])
    certificate = json.loads(cert_path.read_text())
    certificate.update(runtime_code_sha256=digest(old_code), evidence=binding(proof))
    cert_path.write_text(json.dumps(certificate))
    review.update(old_code=old_code, new_code=new_code,
        changes={name: {'old': pair[0], 'new': pair[1]} for name, pair in pins.items()},
        optimizer_policy=mechanism.POLICY_GATE_SFT_POLICY,
        source_certificate=binding(cert_path), fresh_policy_capacity_required=True,
        optimizer_state_unchanged=True, metadata_extra_bytes=64 << 20)
    output = invoke(tmp_path, old, new, review)
    cert = json.loads(output.read_text())
    evidence = json.loads(Path(cert['evidence']['path']).read_text())
    assert evidence['measured_parent_checkpoint_sha256'] == 'a'*64
    assert evidence['new_parent_measured'] is False
    assert 'fresh policy measurement required' in cert['scope']
    assert mechanism._proof_cases(cert['evidence'], identity=review['identity'],
        code_sha=digest(new_code), parent='b'*64)
    review['fresh_policy_capacity_required'] = False
    review_path = Path(evidence['review']['path']); review_path.write_text(json.dumps(review))
    evidence['review'] = binding(review_path)
    evidence_path = Path(cert['evidence']['path']); evidence_path.write_text(json.dumps(evidence))
    with pytest.raises(ValueError, match='ordinary SFT only'):
        mechanism._proof_cases(binding(evidence_path), identity=review['identity'],
            code_sha=digest(new_code), parent='b'*64)


@pytest.mark.parametrize('mutation', [None, 'old-trainer', 'new-trainer', 'helper',
    'old-helper-present', 'missing-helper', 'extra-model', 'objective', 'policy-required',
    'optimizer', 'small-allowance'])
def test_ordinary_writer_policy_requires_exact_addition_and_ordinary_identity(mutation):
    pins=mechanism.ORDINARY_WRITER_COMPOSITION
    review={'optimizer_policy':mechanism.ORDINARY_WRITER_POLICY,
        'old_code':{name:pair[0] for name,pair in pins.items() if pair[0] is not None},
        'new_code':{name:pair[1] for name,pair in pins.items()},
        'identity':{'objective':None},'fresh_policy_capacity_required':True,
        'optimizer_state_unchanged':True,'metadata_extra_bytes':64<<20,
        'ordinary_parameters':[{'name':'learned rows','shape':[4,2304],'dtype':'torch.float32'}]}
    trainer='train/native_state_sft.py';helper='train/native_operation_report.py'
    if mutation=='old-trainer':review['old_code'][trainer]='a'*64
    elif mutation=='new-trainer':review['new_code'][trainer]='b'*64
    elif mutation=='helper':review['new_code'][helper]='c'*64
    elif mutation=='old-helper-present':review['old_code'][helper]='d'*64
    elif mutation=='missing-helper':review['new_code'].pop(helper)
    elif mutation=='extra-model':review['new_code']['model/forward.py']='e'*64
    elif mutation=='objective':review['identity']['objective']={'schema':'native-policy'}
    elif mutation=='policy-required':review['fresh_policy_capacity_required']=False
    elif mutation=='optimizer':review['optimizer_state_unchanged']=False
    elif mutation=='small-allowance':review['metadata_extra_bytes']=(64<<20)-1
    if mutation:
        with pytest.raises(ValueError):
            mechanism._ordinary_writer_contract(review)
            mechanism._memory_allowance(review)
    else:
        mechanism._ordinary_writer_contract(review)
        assert mechanism._memory_allowance(review)==64<<20
        assert mechanism.allowed_changes(mechanism.ORDINARY_WRITER_POLICY)==set(pins)


def writer_chain(tmp_path,monkeypatch):
    old,new,review,proof=fixture(tmp_path)
    pins=mechanism.ORDINARY_WRITER_COMPOSITION
    old_code={name:pair[0] for name,pair in pins.items() if pair[0] is not None}
    new_code={name:pair[1] for name,pair in pins.items()}
    # The inventory verifier is independently exercised by the base tests;
    # these exact pinned maps isolate full certificate/recursive policy flow.
    monkeypatch.setattr(mechanism,'code_inventory',lambda root:old_code if root==old else new_code)
    measured=json.loads(proof.read_text());measured['runtime_code_sha256']=digest(old_code)
    proof.write_text(json.dumps(measured))
    cert_path=Path(review['source_certificate']['path']);certificate=json.loads(cert_path.read_text())
    certificate.update(runtime_code_sha256=digest(old_code),evidence=binding(proof))
    cert_path.write_text(json.dumps(certificate))
    review.update(old_code=old_code,new_code=new_code,
        changes={name:{'old':pair[0],'new':pair[1]} for name,pair in pins.items()},
        optimizer_policy=mechanism.ORDINARY_WRITER_POLICY,source_certificate=binding(cert_path),
        fresh_policy_capacity_required=True,optimizer_state_unchanged=True,metadata_extra_bytes=64<<20)
    return old,new,review,proof


@pytest.mark.parametrize('tamper',[None,'trainer','helper','missing-helper','policy-required'])
def test_writer_inherited_evidence_rechecks_exact_pins_even_when_rehashed(tmp_path,monkeypatch,tamper):
    old,new,review,_=writer_chain(tmp_path,monkeypatch)
    output=invoke(tmp_path,old,new,review)
    cert=json.loads(output.read_text());evidence=json.loads(Path(cert['evidence']['path']).read_text())
    assert evidence['new_parent_measured'] is False
    assert evidence['measured_parent_checkpoint_sha256']=='a'*64
    assert evidence['continued_parent_checkpoint_sha256']=='b'*64
    assert evidence['extra_bytes_per_memory_domain']==64<<20
    assert evidence['measured_peaks']==[{'id':'actual maximum','device_bytes':8<<30,'host_bytes':4<<30}]
    assert 'report-writer-only' in cert['scope'] and 'fresh policy measurement required' in cert['scope']
    cases=mechanism._proof_cases(cert['evidence'],identity=review['identity'],
        code_sha=digest(review['new_code']),parent='b'*64)
    assert cases[0]['inherited_extra_bytes']==64<<20
    if tamper is None:
        assert invoke(tmp_path,old,new,review)==output
        return
    if tamper=='trainer':review['new_code']['train/native_state_sft.py']='f'*64
    elif tamper=='helper':review['new_code']['train/native_operation_report.py']='e'*64
    elif tamper=='missing-helper':review['new_code'].pop('train/native_operation_report.py')
    else:review['fresh_policy_capacity_required']=False
    review['changes']={name:{'old':review['old_code'].get(name),'new':review['new_code'].get(name)}
        for name in review['old_code'].keys()|review['new_code'].keys()
        if review['old_code'].get(name)!=review['new_code'].get(name)}
    review_path=Path(evidence['review']['path']);review_path.write_text(json.dumps(review))
    evidence['review']=binding(review_path);evidence['runtime_code_sha256']=digest(review['new_code'])
    evidence_path=Path(cert['evidence']['path']);evidence_path.write_text(json.dumps(evidence))
    with pytest.raises(ValueError,match='ordinary SFT only|exact two-file'):
        mechanism._proof_cases(binding(evidence_path),identity=review['identity'],
            code_sha=digest(review['new_code']),parent='b'*64)


@pytest.mark.parametrize('mutation',['device-headroom','host-headroom','reserve','layout','identity'])
def test_writer_continuation_keeps_existing_measured_memory_and_identity_guards(tmp_path,monkeypatch,mutation):
    old,new,review,_=writer_chain(tmp_path,monkeypatch)
    if mutation=='device-headroom':review['device_memory_bytes']=8<<30
    elif mutation=='host-headroom':review['host_memory_limit_bytes']=4<<30
    elif mutation=='reserve':review['reserve_bytes']=(1<<30)-1
    elif mutation=='layout':review['parameter_layout_unchanged']=False
    else:review['identity']={**review['identity'],'checkpoint_input_offload':True}
    with pytest.raises(ValueError):invoke(tmp_path,old,new,review)
    assert not(tmp_path/'out').exists()


def test_historical_cli_executes_the_shared_verifier(tmp_path):
    import subprocess
    import sys
    old, new, review, _ = fixture(tmp_path)
    review_path = tmp_path/'cli-review.json'
    review_path.write_text(json.dumps(review))
    script = Path(__file__).resolve().parents[2]/'scripts/neuralese_data/continue_native_capacity.py'
    result = subprocess.run([sys.executable, str(script), '--review', json.dumps(binding(review_path)),
        '--old-package', str(old), '--new-package', str(new), '--out', str(tmp_path/'cli-out')],
        capture_output=True, text=True, check=True)
    certificate = json.loads(result.stdout)
    value = json.loads(Path(certificate['path']).read_text())
    assert geometry_capacity(certificate, identity=review['identity'], code=code_inventory(new)) == value['limits']
    evidence = json.loads(Path(value['evidence']['path']).read_text())
    assert mechanism._proof_cases(value['evidence'], identity=review['identity'],
        code_sha=digest(code_inventory(new)), parent=review['new_parent_checkpoint_sha256'])
    assert evidence['new_parent_measured'] is False


@pytest.mark.parametrize('change',['unpassed-numerics','inflated-limit','wrong-parent'])
def test_public_certificate_verifier_checks_measured_evidence_not_just_root(tmp_path,change):
    old, _, review, proof = fixture(tmp_path)
    certificate_path=Path(review['source_certificate']['path'])
    certificate=json.loads(certificate_path.read_text())
    if change=='unpassed-numerics':
        value=json.loads(proof.read_text());value['numerical_reference']['passed']=False
        proof.write_text(json.dumps(value));certificate['evidence']=binding(proof)
    elif change=='inflated-limit':certificate['limits']['kv_slots']+=1
    certificate_path.write_text(json.dumps(certificate))
    with pytest.raises(ValueError):
        mechanism.verify_capacity_certificate(binding(certificate_path),identity=review['identity'],
            code=code_inventory(old),parent='0'*64 if change=='wrong-parent' else review['old_parent_checkpoint_sha256'])


@pytest.mark.parametrize('architecture,dtype',[('lfm2','torch.float32'),('mellum','torch.bfloat16')])
def test_metadata_profile_continuation_reverifies_original_capacity_chain(tmp_path,architecture,dtype):
    import shutil
    old,adam,review,proof=fixture(tmp_path)
    identity={**review['identity'],'model_config':{'architecture':architecture,'dtype':dtype,
        'native_attention_precision':'ieee','native_linear_precision':'bf16x6'}}
    value=json.loads(proof.read_text());value['identity']=copy.deepcopy(identity);proof.write_text(json.dumps(value))
    original_path=Path(review['source_certificate']['path']);value=json.loads(original_path.read_text())
    value.update(identity=copy.deepcopy(identity),evidence=binding(proof));original_path.write_text(json.dumps(value))
    review.update(identity=identity,source_certificate=binding(original_path));first=tmp_path/'first';first.mkdir()
    source=invoke(first,old,adam,review);source_bytes=source.read_bytes();proof_bytes=proof.read_bytes()
    candidate=tmp_path/'candidate';shutil.copytree(adam,candidate)
    (candidate/'train/native_resident.py').write_text('reviewed KEEP memory-identity normalizer\n')
    current=copy.deepcopy(review);current['identity']['model_config']['history_reasoning_policy']='keep'
    current.update(source_certificate=binding(source),old_code=code_inventory(adam),new_code=code_inventory(candidate),
        old_parent_checkpoint_sha256='b'*64,new_parent_checkpoint_sha256='c'*64,
        optimizer_policy=mechanism.METADATA_POLICY,optimizer_state_unchanged=True,metadata_extra_bytes=64<<20)
    current['changes']={'train/native_resident.py':{'old':None,'new':current['new_code']['train/native_resident.py']}}
    phase=tmp_path/'second';phase.mkdir();result=invoke(phase,adam,candidate,current)
    mechanism.verify_capacity_certificate(binding(result),identity=current['identity'],code=code_inventory(candidate),parent='c'*64)
    assert source.read_bytes()==source_bytes and proof.read_bytes()==proof_bytes
    continued=json.loads(Path(json.loads(result.read_text())['evidence']['path']).read_text())
    assert continued['new_parent_measured'] is False and continued['measured_parent_checkpoint_sha256']=='a'*64
    # Inheritance still reaches the exact original measured proof; KEEP cannot
    # conceal changed numerics, model shape, source bytes or parent ancestry.
    value=json.loads(proof.read_text());value['numerical_reference']['passed']=False;proof.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        mechanism.verify_capacity_certificate(binding(result),identity=current['identity'],code=code_inventory(candidate),parent='c'*64)


@pytest.mark.parametrize('role',['lfm','mellum'])
@pytest.mark.parametrize('mutation',[None,'old-module','new-module','missing-module','numerical-neighbor',
    'source-parser','objective','missing-profile','missing-policy-gate'])
def test_keep_profile_policy_binds_exact_reviewed_overlay_and_parser(role,mutation):
    composition=mechanism.HISTORY_PROFILE_COMPOSITIONS[role]
    old={name:pair[0]for name,pair in composition.items()};new={name:pair[1]for name,pair in composition.items()}
    for name in mechanism.HISTORY_PROFILE_SOURCE_POLICY_FILES:old[name]=new[name]='f'*64
    review={'optimizer_policy':mechanism.HISTORY_PROFILE_POLICY,'old_code':old,'new_code':new,
        'identity':{'model_config':{'history_reasoning_policy':'keep'}},'fresh_policy_capacity_required':True,
        'optimizer_state_unchanged':True,'metadata_extra_bytes':64<<20,
        'ordinary_parameters':[{'name':'rows','shape':[4,1024],'dtype':'torch.float32'}]}
    if mutation=='old-module':old['model/native_model.py']='0'*64
    elif mutation=='new-module':new['model/native_model.py']='0'*64
    elif mutation=='missing-module':new.pop('serve/engine.py')
    elif mutation=='numerical-neighbor':new['model/shared_state.py']='0'*64
    elif mutation=='source-parser':new[mechanism.HISTORY_PROFILE_SOURCE_POLICY_FILES[0]]='0'*64
    elif mutation=='objective':review['identity']['objective']={'kind':'policy'}
    elif mutation=='missing-profile':review['identity']['model_config'].clear()
    elif mutation=='missing-policy-gate':review['fresh_policy_capacity_required']=False
    if mutation:
        with pytest.raises(ValueError):mechanism._history_profile_contract(review)
    else:
        mechanism._history_profile_contract(review)
        assert mechanism._memory_allowance(review)==64<<20
        assert mechanism.allowed_changes(mechanism.HISTORY_PROFILE_POLICY)==set(composition)
        # Already adopted one-file progress-copy optimization is not reverted.
        old['train/native_continuation.py']=new['train/native_continuation.py']
        mechanism._history_profile_contract(review)
