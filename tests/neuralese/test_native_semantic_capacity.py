"""Whole paired objectives retain gradients and resume exact phase evidence."""
import copy
import json
from pathlib import Path
import pytest
import torch
from test_native_semantic import setup_pair
from natlang_neuralese.train.native_semantic_capacity import (capacity_frontier,qualify_capacity,
    ensure_phase_qualification)
from natlang_neuralese.train.native_semantic_stage import PairedSource
from natlang_neuralese.common.hashing import sha256_file_hex,canonical_json_sha256_hex
from natlang_neuralese.model.native_weight_carrier import active_cotangent_scope


def source(tmp_path,rows):
    path=tmp_path/'pairs.jsonl';path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    return PairedSource({'path':str(path),'sha256':sha256_file_hex(path)},split='train')


def test_frontier_covers_actual_larger_consumer_and_refuses_mutated_admission(tmp_path):
    participants,row,_=setup_pair()
    larger=copy.deepcopy(row);larger['id']='larger'
    for view in larger['models'].values():
        view['graph']['nodes'][1]['token_ids'] += [14,15]
        view['graph']['nodes'][1]['target_weights'] += [1.,1.]
    admitted=source(tmp_path,[row,larger])
    frontier,representatives,count=capacity_frontier(admitted,participants)
    assert count==2 and [entry['id']for _,entry in frontier]==['larger']
    assert len(representatives)==1
    modified=copy.deepcopy(row);modified['id']='larger'
    admitted.rows=lambda:iter([modified])
    with pytest.raises(ValueError,match='row changed'):
        capacity_frontier(admitted,participants)


@pytest.mark.parametrize('bf16',[False,True])
def test_actual_whole_pair_capacity_preserves_weights_moments_rng_and_clears_all_gradients(bf16):
    participants,row,vae=setup_pair()
    from natlang_neuralese.train.native_state_sft import make_native_optimizer
    for participant in participants.values():
        if bf16:participant.model.port.hf.to(torch.bfloat16)
        participant.model.port.hf.natlang_precision='bf16'
        participant.optimizer=make_native_optimizer(participant.model,lr=1e-4,in_backward=False)[0]
    optimizer=torch.optim.AdamW(vae.parameters(),lr=1e-4)
    for p in vae.parameters():p.grad=torch.ones_like(p)
    optimizer.step();optimizer.zero_grad(set_to_none=True)
    before=copy.deepcopy(optimizer.state_dict())
    parameters=[p for entry in participants.values()for p in entry.model.parameters()]+list(vae.parameters())
    weights=[p.detach().clone()for p in parameters];rng=torch.get_rng_state().clone()
    proof=qualify_capacity(vae,participants,row,phase='joint',options={},optimizer=optimizer,
        fetch_external=None,device='cpu')
    assert proof['passed'] and proof['retained_gradient_bytes']>0
    assert proof['optimizer_updates']==proof['exposure_updates']==0
    assert all(p.grad is None and torch.equal(p,saved)for p,saved in zip(parameters,weights))
    assert torch.equal(torch.get_rng_state(),rng) and active_cotangent_scope() is None
    for key,value in before['state'].items():
        for field,expected in value.items():
            torch.testing.assert_close(optimizer.state_dict()['state'][key][field],expected,rtol=0,atol=0)


def test_failed_capacity_restores_boundary_without_consuming_exposure(monkeypatch):
    from natlang_neuralese.train import native_semantic_capacity as capacity
    participants,row,vae=setup_pair();rng=torch.get_rng_state().clone()
    def fail(*args,**kwargs):
        torch.rand(17)
        raise RuntimeError('actual objective failed')
    monkeypatch.setattr(capacity,'joint_loss',fail)
    with pytest.raises(RuntimeError,match='actual objective failed'):
        qualify_capacity(vae,participants,row,phase='joint',options={},optimizer=None,
            fetch_external=None,device='cpu')
    assert torch.equal(torch.get_rng_state(),rng) and active_cotangent_scope() is None
    assert all(p.grad is None for v in participants.values()for p in v.model.parameters())


def test_phase_evidence_is_carried_on_resume_and_new_options_requalify(tmp_path):
    participants,row,vae=setup_pair();admitted=source(tmp_path,[row])
    optimizer=torch.optim.AdamW(vae.parameters(),lr=1e-4)
    training={'step':17};cache={}
    args=dict(phase='joint',optimizer=optimizer,fetch_external=None,device='cpu',
              training=training,out=tmp_path,cache=cache)
    rng=torch.get_rng_state().clone()
    proof=ensure_phase_qualification(vae,participants,admitted,options={},**args)
    assert proof['passed'] and proof['entered_at_step']==17 and proof['source_rows']==1
    assert torch.equal(torch.get_rng_state(),rng)
    assert training['step']==17 and 'exposure'not in training
    # Exact checkpoint-carried proof, without replay or a geometry scan.
    resumed=json.loads(json.dumps(training))
    admitted.rows=lambda:(_ for _ in ()).throw(AssertionError('unneeded source rescan'))
    reused=ensure_phase_qualification(vae,participants,admitted,options={},**{**args,'training':resumed,'cache':{}})
    assert canonical_json_sha256_hex(reused)==canonical_json_sha256_hex(proof)
    # Adapted objective cannot inherit the previous profile certificate.
    with pytest.raises(AssertionError,match='unneeded source rescan'):
        ensure_phase_qualification(vae,participants,admitted,options={'control_weight':2.},
            **{**args,'training':resumed,'cache':{}})


def test_new_runtime_math_cannot_reuse_checkpoint_carried_phase_proof(tmp_path,monkeypatch):
    from natlang_neuralese.train import native_resident
    participants,row,vae=setup_pair();admitted=source(tmp_path,[row]);training={'step':0}
    args=dict(phase='joint',optimizer=None,fetch_external=None,device='cpu',training=training,out=tmp_path)
    proof=ensure_phase_qualification(vae,participants,admitted,options={},cache={},**args)
    assert proof['passed']
    actual=native_resident.code_inventory
    def changed(package):
        inventory=actual(package)
        inventory['model/shared_state.py']='0'*64
        return inventory
    monkeypatch.setattr(native_resident,'code_inventory',changed)
    admitted.rows=lambda:(_ for _ in ()).throw(AssertionError('new runtime must qualify'))
    with pytest.raises(AssertionError,match='new runtime must qualify'):
        ensure_phase_qualification(vae,participants,admitted,options={},cache={},**args)
    monkeypatch.setattr(native_resident,'code_inventory',actual)
    monkeypatch.setattr(torch,'__version__','test-different-runtime')
    with pytest.raises(AssertionError,match='new runtime must qualify'):
        ensure_phase_qualification(vae,participants,admitted,options={},cache={},**args)


def test_failed_phase_keeps_exact_failed_subproof_without_readiness_or_exposure_credit(tmp_path,monkeypatch):
    from natlang_neuralese.train import native_semantic_capacity as capacity
    participants,row,vae=setup_pair();admitted=source(tmp_path,[row]);training={'step':23}
    actual=capacity.qualify_capacity
    def failed(*args,**kwargs):
        proof=actual(*args,**kwargs)
        proof.update(passed=False,diagnostic='test observed nonfinite capacity')
        return proof
    monkeypatch.setattr(capacity,'qualify_capacity',failed)
    rng=torch.get_rng_state().clone()
    with pytest.raises(RuntimeError,match='capacity qualification failed'):
        ensure_phase_qualification(vae,participants,admitted,phase='joint',options={},optimizer=None,
            fetch_external=None,device='cpu',training=training,out=tmp_path,cache={})
    paths=list((tmp_path/'phase-qualifications').glob('*.progress.json'))
    assert len(paths)==1
    failure=json.loads(paths[0].read_text())
    assert failure['status']=='failed' and failure['entered_at_step']==23
    assert failure['complete_gradients'][0]['passed'] is True
    assert failure['whole_pair_capacity'][0]['passed'] is False
    assert failure['whole_pair_capacity'][0]['diagnostic']=='test observed nonfinite capacity'
    assert failure['optimizer_updates']==failure['exposure_updates']==0
    assert training=={'step':23,'phase_qualifications':{}}
    assert torch.equal(torch.get_rng_state(),rng) and active_cotangent_scope() is None
