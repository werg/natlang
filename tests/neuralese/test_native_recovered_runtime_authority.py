"""Recovered execution is verified independently of its inherited plan's reader."""
import copy
import json
from pathlib import Path

import pytest

from test_native_plan_recovery import fixture
from natlang_neuralese.rl import native_raw_handoff as handoff, native_runtime_handoff as runtime
from natlang_neuralese.train import native_resident as resident
from natlang_neuralese.common.jsonio import write_json_durable as write


def evidence(tmp_path, monkeypatch):
    _, _, _, _, saved = fixture(tmp_path)
    package = tmp_path/'executed'; package.mkdir(); (package/'math.py').write_text('value=1\n')
    code = resident.digest(resident.code_inventory(package))
    parent = 'a'*64
    request = tmp_path/'capacity-request.json'
    write(request, {'schema':'natlang.native-memory-qualification-request/1',
                    'runtime_code_sha256':code,'identity':{'runtime':'actual'}})
    ref = resident.binding(request)
    proof = tmp_path/'proof.json'
    write(proof, {'schema':'natlang.native-memory-qualification/1','passed':True,
                  'runtime_code_sha256':code,'identity':{'runtime':'actual'},
                  'parent_checkpoint_sha256':parent,'request':ref,'optimizer_updates':0,
                  'numerical_reference':{'passed':True},'capacity':{'passed':True,
                  'cases':[{'status':'passed','forward_tokens':11,'kv_slots':13}]}})
    certificate = tmp_path/'capacity.json'
    write(certificate, {'schema':'natlang.native-resident-geometry-qualification/1','passed':True,
                        'runtime_code_sha256':code,'identity':{'runtime':'actual'},'evidence':resident.binding(proof),
                        'limits':{'forward_tokens':11,'kv_slots':13}})
    options = saved['training']['options']
    options.update(continuation={'checkpoint_sha256':parent}, memory_qualification={
        'passed':True,'request':ref,'parent_checkpoint_sha256':parent},
        resident_geometry_qualification=str(certificate))
    argv = ['python','--memory-qualification',json.dumps([ref])]
    monkeypatch.setattr(runtime, '_owned_launch_chain', lambda *a,**k: (None,argv,{}))
    return package, saved, request, proof, certificate, argv


def test_recovered_runtime_uses_exact_saved_launch_capacity_and_optional_binding_bytes(tmp_path, monkeypatch):
    package, saved, request, proof, certificate, argv = evidence(tmp_path, monkeypatch)
    saved['training']['options']['memory_qualification']['request']['bytes'] = request.stat().st_size
    actual = handoff._verify_recovered_execution({}, package, saved)
    assert actual['package'] == str(package.resolve())
    assert actual['capacity_certificate'] == resident.binding(certificate)
    assert actual['code'] == resident.digest(resident.code_inventory(package))


@pytest.mark.parametrize('change', ['code','parent','launch','passed','proof-parent','proof-code',
                                  'proof-request','proof-passed','certificate-code','certificate-passed',
                                  'identity','request-bytes','missing-saved'])
def test_recovered_runtime_rejects_borrowed_or_changed_evidence(tmp_path, monkeypatch, change):
    package, saved, request, proof, certificate, argv = evidence(tmp_path, monkeypatch)
    options = saved['training']['options']
    if change=='code':(package/'math.py').write_text('value=2\n')
    elif change=='parent':options['continuation']['checkpoint_sha256']='b'*64
    elif change=='launch':argv[1]='--unrelated'
    elif change=='passed':options['memory_qualification']['passed']=False
    elif change=='request-bytes':options['memory_qualification']['request']['bytes']=0
    elif change=='missing-saved':saved=None
    else:
        p = certificate if change.startswith('certificate') else proof
        v=json.loads(p.read_text())
        if change.endswith('parent'):v['parent_checkpoint_sha256']='c'*64
        elif change.endswith('code'):v['runtime_code_sha256']='d'*64
        elif change.endswith('passed'):v['passed']=False
        elif change=='proof-request':v['request']['sha256']='e'*64
        elif change=='identity':v['identity']={'runtime':'borrowed'}
        write(p,v)
        if p==proof:
            v=json.loads(certificate.read_text());v['evidence']=resident.binding(proof);write(certificate,v)
    with pytest.raises((ValueError,FileNotFoundError)):
        handoff._verify_recovered_execution({}, package, saved)


def cohort_evidence(tmp_path, monkeypatch, count):
    from natlang_neuralese.eval.native_memory import combine_capacity_proofs
    package,saved,request,proof,certificate,argv=evidence(tmp_path,monkeypatch)
    options=saved['training']['options'];parent=options['continuation']['checkpoint_sha256']
    output=tmp_path/'learner';output.mkdir();requests=[];checks=[];proofs=[]
    original=json.loads(request.read_text())
    for index in range(count):
        p=tmp_path/f'request-{index}.json';write(p,{**original,'case':index});ref=resident.binding(p)
        value={'schema':'natlang.native-memory-qualification/1','passed':True,'optimizer_updates':0,
               'request':ref,'parent_checkpoint_sha256':parent,'runtime_code_sha256':original['runtime_code_sha256'],
               'identity':original['identity'],'numerical_reference':{'passed':True},
               'capacity':{'passed':True,'cases':[{'status':'passed','forward_tokens':11+index,'kv_slots':13+index}]}}
        p=output/'memory-qualifications'/ref['sha256']/parent/'proof.json';p.parent.mkdir(parents=True);write(p,value)
        requests.append(ref);checks.append(value);proofs.append(resident.binding(p))
    if count>1:
        certificate=combine_capacity_proofs(output,proofs,identity=original['identity'],
            code_sha=original['runtime_code_sha256'],checkpoint_sha256=parent)
    else:
        write(certificate,{'schema':'natlang.native-resident-geometry-qualification/1','passed':True,
            'runtime_code_sha256':original['runtime_code_sha256'],'identity':original['identity'],
            'limits':{'forward_tokens':11,'kv_slots':13},'evidence':proofs[0]})
    summary={'schema':'natlang.native-memory-cohort-qualification/1','passed':True,'optimizer_updates':0,
        'parent_checkpoint_sha256':parent,'request':requests,'checks':checks,
        'request_aliases':[{'requested_binding':r,'resolved_binding':r,'canonical_binding':r}for r in requests],
        'retained_capacity':str(certificate)}
    write(output/'memory-qualification.json',summary)
    options['memory_qualification']['request']=requests
    options['resident_geometry_qualification']=str(certificate)
    argv[2]=json.dumps(requests);argv+=['--out',str(output)]
    return package,saved,certificate,output,summary


@pytest.mark.parametrize('count',[1,2])
def test_recovered_runtime_verifies_the_actual_complete_cohort_and_combined_capacity(tmp_path,monkeypatch,count):
    package,saved,certificate,output,_=cohort_evidence(tmp_path,monkeypatch,count)
    result=handoff._verify_recovered_execution({},package,saved)
    assert result['cohort_summary']==resident.binding(output/'memory-qualification.json')
    assert result['capacity_certificate']==resident.binding(certificate)


@pytest.mark.parametrize('change',['incomplete','failed','parent','aliases','unmeasured-capacity','failed-reference'])
def test_recovered_cohort_cannot_borrow_or_skip_actual_capacity_evidence(tmp_path,monkeypatch,change):
    package,saved,certificate,output,summary=cohort_evidence(tmp_path,monkeypatch,2)
    if change=='incomplete':summary['checks'].pop()
    elif change=='failed':summary['passed']=False
    elif change=='parent':summary['parent_checkpoint_sha256']='f'*64
    elif change=='aliases':summary['request_aliases'].reverse()
    elif change=='unmeasured-capacity':
        v=json.loads(certificate.read_text());v['limits']['forward_tokens']=999;write(certificate,v)
    elif change=='failed-reference':summary['checks'][0]['numerical_reference']['passed']=False
    write(output/'memory-qualification.json',summary)
    with pytest.raises(ValueError):handoff._verify_recovered_execution({},package,saved)


@pytest.mark.parametrize('change', ['updates', 'updates-bool', 'reference', 'capacity', 'empty', 'failed-case', 'invalid-tokens', 'limits'])
def test_scalar_recovery_requires_actual_zero_update_numerical_whole_graph_capacity(tmp_path, monkeypatch, change):
    package, saved, request, proof, certificate, argv = evidence(tmp_path, monkeypatch)
    value = json.loads(proof.read_text())
    if change == 'updates': value['optimizer_updates'] = 1
    elif change == 'updates-bool': value['optimizer_updates'] = False
    elif change == 'reference': value['numerical_reference']['passed'] = False
    elif change == 'capacity': value['capacity']['passed'] = False
    elif change == 'empty': value['capacity']['cases'] = []
    elif change == 'failed-case': value['capacity']['cases'][0]['status'] = 'failed'
    elif change == 'invalid-tokens': value['capacity']['cases'][0]['forward_tokens'] = True
    write(proof, value)
    cert = json.loads(certificate.read_text()); cert['evidence'] = resident.binding(proof)
    if change == 'limits': cert['limits']['kv_slots'] = 999
    write(certificate, cert)
    with pytest.raises(ValueError): handoff._verify_recovered_execution({}, package, saved)
