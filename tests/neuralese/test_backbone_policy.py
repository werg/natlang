"""Shared full-backbone policy selection and checkpoint restoration."""
from types import SimpleNamespace

import pytest
import torch

from natlang_neuralese.train.backbone_policy import (
    backbone_trainable_state,
    configure_backbone_training,
    full_backbone_parameter_names,
    resolve_backbone_policy,
    restore_backbone_trainables,
)
from natlang_neuralese.model.lfm2_port import ControlTokens, PortBackbone


def tiny_lfm(seed=31):
    pytest.importorskip('transformers')
    from transformers import Lfm2Config, Lfm2ForCausalLM
    torch.manual_seed(seed)
    config=Lfm2Config(vocab_size=64,hidden_size=32,intermediate_size=64,
        num_hidden_layers=3,num_attention_heads=4,num_key_value_heads=2,
        block_multiple_of=8,block_auto_adjust_ff_dim=False,
        layer_types=['conv','full_attention','conv'])
    return PortBackbone(Lfm2ForCausalLM(config).eval(),ControlTokens(62,63),fast=False)


def test_full_policy_selects_all_and_only_native_transformer_layer_parameters():
    backbone=tiny_lfm()
    named=configure_backbone_training(backbone,'full')
    selected={name for name,_ in named}
    expected=full_backbone_parameter_names(backbone)
    assert selected==expected
    assert selected
    assert all(parameter.requires_grad for _,parameter in named)
    assert all(not parameter.requires_grad for name,parameter in backbone.hf.named_parameters()
               if name not in expected)


def test_full_parameter_names_form_a_stable_complete_muon_schema():
    from natlang_neuralese.train.trajectory_state import trajectory_optimizer
    backbone=tiny_lfm(seed=36)
    named=configure_backbone_training(backbone,'full')
    names=[name for name,_ in named]
    parameters=[parameter for _,parameter in named]
    optimizer=trajectory_optimizer('muon',{},parameters,[],vocab_size=64,lr=1e-4,
        lora_lr=3e-5,heads_lr=1e-4,lora_names=names)
    schema=optimizer.state_dict()['schema']
    assert {row['name'] for row in schema}=={'backbone.'+name for name in names}
    # Full selected parameter identities are complete and unique in the optimizer schema.
    assert len(schema)==len(names)==len({row['name'] for row in schema})
    restored=trajectory_optimizer('muon',{},parameters,[],vocab_size=64,lr=1e-4,
        lora_lr=3e-5,heads_lr=1e-4,lora_names=names)
    restored.load_state_dict(optimizer.state_dict())


def test_auto_policy_resolves_native_to_full_and_ternary_to_qat():
    native=tiny_lfm()
    assert resolve_backbone_policy(native,'auto')=='full'
    native.ternary=True
    assert resolve_backbone_policy(native,'auto')=='qat'


def test_full_backbone_state_roundtrips_through_load_engine(tmp_path,monkeypatch):
    import natlang_neuralese.model.lfm2_port as lfm
    import natlang_neuralese.serve as serve

    source=tiny_lfm(seed=32)
    named=configure_backbone_training(source,'full')
    with torch.no_grad():
        for index,(_,parameter) in enumerate(named):
            parameter.fill_((index+1)*.001)
    from natlang_neuralese.model.heads import PortHeads
    source_heads=PortHeads(source,cutoff=2,max_length=8,profile='latent-sketch-v2')
    checkpoint=tmp_path/'full-heads.pt'
    torch.save({'heads':source_heads.state_dict(),'control_rows':source.control_rows.detach().cpu(),
        'backbone_trainables':backbone_trainable_state(named),'backbone_training':'full',
        'port_config':{'cutoff':2,'max_length':8,'profile':'latent-sketch-v2'},
        'lora':{}},checkpoint)

    fresh=tiny_lfm(seed=33)
    class TinyTokenizer:
        eos_token_id=2
        def convert_tokens_to_ids(self,token): return 3
    tokenizer=TinyTokenizer()
    monkeypatch.setattr(lfm,'load_backbone',lambda *args,**kwargs:(fresh.hf,tokenizer))
    monkeypatch.setattr(lfm.ControlTokens,'from_tokenizer',staticmethod(lambda _tokenizer:ControlTokens(62,63)))
    monkeypatch.setattr(lfm,'load_conv_kernel',lambda:None)
    loaded=serve.load_engine(heads_checkpoint=str(checkpoint),device='cpu')
    actual=dict(loaded.backbone.hf.named_parameters())
    for name,value in backbone_trainable_state(named).items():
        torch.testing.assert_close(actual[name],value)
    assert all(not parameter.requires_grad for parameter in loaded.backbone.hf.parameters())


def test_recurrence_checkpoint_overrides_with_full_backbone_state(tmp_path,monkeypatch):
    import natlang_neuralese.serve.recurrence_checkpoint as checkpoint_loader
    from natlang_neuralese.model.heads import PortHeads

    base=tiny_lfm(seed=34)
    heads=PortHeads(base,cutoff=2,max_length=8,profile='latent-sketch-v2')
    changed_named=configure_backbone_training(base,'full')
    with torch.no_grad():
        for _,parameter in changed_named:
            parameter.add_(.125)
    parent=tmp_path/'parent-heads.pt';torch.save({},parent)
    from natlang_neuralese.common.hashing import sha256_file_hex
    state={'schema':'natlang.neuralese_recurrence_checkpoint/1',
        'identity':{'options':{'heads':str(parent),'base':None},'files':{str(parent.resolve()):sha256_file_hex(parent)}},'step':7,
        'heads':heads.state_dict(),'port_config':{'max_length':8,'content_transport':'top-state'},
        'backbone_training':'full','backbone_trainables':backbone_trainable_state(changed_named),
        'lora':{},'control_rows':base.control_rows.detach().cpu()}
    path=tmp_path/'recurrence.pt';torch.save(state,path)

    fresh=tiny_lfm(seed=35)
    fresh_heads=PortHeads(fresh,cutoff=2,max_length=8,profile='latent-sketch-v2')
    engine=SimpleNamespace(backbone=fresh,heads=fresh_heads,max_block=8)
    monkeypatch.setattr(checkpoint_loader,'load_engine',lambda *args,**kwargs:engine)
    loaded,_=checkpoint_loader.load_recurrence_checkpoint(path)
    actual=dict(loaded.backbone.hf.named_parameters())
    for name,value in backbone_trainable_state(changed_named).items():
        torch.testing.assert_close(actual[name],value)
