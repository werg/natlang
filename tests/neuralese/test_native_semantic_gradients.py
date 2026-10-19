"""Actual two-model semantic objectives, complete derivatives and restored state."""
import copy
import pytest
import torch
from natlang_neuralese.train.native_semantic_gradients import SemanticObjectiveScope,qualify_joint_gradients
from natlang_neuralese.model.native_weight_carrier import active_cotangent_scope
from natlang_neuralese.train.loop import capture_training_rng_state
from test_native_semantic import setup_pair


@pytest.mark.parametrize('phase',['vae','joint','distill'])
def test_real_complete_semantic_gradient_qualification_restores_state(phase):
    participants,row,vae=setup_pair()
    for participant in participants.values():
        for parameter in participant.model.parameters():
            parameter.requires_grad_(parameter.requires_grad and phase!='vae')
    vae.requires_grad_(phase!='distill')
    parameters=[p for v in participants.values() for p in v.model.parameters()]+list(vae.parameters())
    before=[p.detach().clone() for p in parameters]
    rng=capture_training_rng_state('cpu')
    proof=qualify_joint_gradients(vae,participants,row,phase=phase,atol=1e-5,rtol=1e-4,device='cpu')
    assert proof['passed'],proof
    assert proof['optimizer_updates']==proof['exposure_updates']==0
    assert proof['complete_gradients']['derivative_comparison_count']==sum(p.requires_grad for p in parameters)
    assert not any(name=='input_embeddings' for name in proof['complete_gradients']['comparisons'])
    assert all(torch.equal(p,saved) and p.grad is None for p,saved in zip(parameters,before))
    assert torch.equal(torch.get_rng_state(),rng['torch_rng'])
    assert capture_training_rng_state('cpu')['python_rng']==rng['python_rng']
    assert active_cotangent_scope() is None
    expected={name+'.'+key for name,v in participants.items() for key,p in v.model.named_parameters() if p.requires_grad}
    expected|={'vae.'+key for key,p in vae.named_parameters() if p.requires_grad}
    assert {entry['name']for entry in proof['complete_gradients']['inventory']}==expected


def test_scope_routes_low_precision_parameters_from_both_models_and_cleans_up():
    participants,_,vae=setup_pair()
    for participant in participants.values():
        participant.model.port.hf.to(torch.bfloat16)
        from natlang_neuralese.train.native_state_sft import make_native_optimizer
        participant.model.port.hf.natlang_precision='bf16'
        participant.optimizer=make_native_optimizer(participant.model,lr=1e-4,in_backward=False)[0]
    with pytest.raises(RuntimeError,match='test interrupted'):
        with SemanticObjectiveScope(participants,vae) as scope:
            entries=scope.cotangents.entries
            assert entries
            for participant in participants.values():
                expected={id(p)for p in participant.model.parameters() if p.requires_grad and p.dtype==torch.bfloat16}
                assert expected and expected<=set(entries)
            raise RuntimeError('test interrupted')
    assert active_cotangent_scope() is None


def test_failed_reference_restores_rng_and_leaves_all_weights_unmodified(monkeypatch):
    from natlang_neuralese.train import native_semantic as objective
    participants,row,vae=setup_pair();actual=objective.joint_loss
    parameters=[p for v in participants.values() for p in v.model.parameters()]+list(vae.parameters())
    before=[p.detach().clone()for p in parameters];rng=torch.get_rng_state().clone()
    def fail_reference(*args,**kwargs):
        if kwargs.get('reference'):
            torch.rand(11)
            raise RuntimeError('test reference failure')
        return actual(*args,**kwargs)
    monkeypatch.setattr(objective,'joint_loss',fail_reference)
    with pytest.raises(RuntimeError,match='test reference failure'):
        qualify_joint_gradients(vae,participants,row,phase='joint',atol=1e-5,rtol=1e-4,device='cpu')
    assert active_cotangent_scope() is None and torch.equal(torch.get_rng_state(),rng)
    assert all(torch.equal(p,saved)and p.grad is None for p,saved in zip(parameters,before))


def test_both_bf16_backbones_have_complete_joint_derivative_agreement():
    participants,row,vae=setup_pair()
    for participant in participants.values():
        participant.model.port.hf.to(torch.bfloat16)
        from natlang_neuralese.train.native_state_sft import make_native_optimizer
        participant.model.port.hf.natlang_precision='bf16'
        participant.optimizer=make_native_optimizer(participant.model,lr=1e-4,in_backward=False)[0]
    proof=qualify_joint_gradients(vae,participants,row,phase='joint',atol=1e-5,rtol=1e-4,device='cpu')
    assert proof['passed'],proof
    for name,participant in participants.items():
        expected={name+'.'+key for key,p in participant.model.named_parameters() if p.requires_grad}
        assert expected<={entry['name']for entry in proof['complete_gradients']['inventory']}


def test_layout_probe_uses_the_actual_selected_backend_dispatch(monkeypatch):
    import contextlib
    from natlang_neuralese.train import native_semantic_stage as stage
    from natlang_neuralese.train.native_semantic import parse_pair
    participants,row,_=setup_pair();parsed=parse_pair(row,participants)
    active=[]
    class Backend:
        @contextlib.contextmanager
        def autocast(self):
            active.append('backend')
            try:yield
            finally:active.pop()
    participants['mellum'].backend=Backend()
    monkeypatch.setattr(stage,'_first_view',lambda source,participants,name:parsed[name])
    producer=stage.producer
    def checked(*args,**kwargs):
        assert active==['backend'] and not torch.is_grad_enabled()
        return producer(*args,**kwargs)
    monkeypatch.setattr(stage,'producer',checked)
    layout=stage._layout(None,participants,'mellum',lambda *args:None)
    assert layout.embedding_width>0 and active==[]
