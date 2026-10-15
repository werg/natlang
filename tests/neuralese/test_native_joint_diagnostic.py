"""Diagnostic trace plumbing; actual full-size CUDA evidence remains separate."""
import importlib.util
from pathlib import Path
import pytest
import torch
from natlang_neuralese.model.native_model import load_native_model,native_identity
from natlang_neuralese.train.native_semantic import SemanticParticipant
from natlang_neuralese.rl.native_replay import compile_replay,replay_logps
from test_native_full_updates import source


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_forced_writer_actual_sampled_reader_trace_replays_and_backpropagates(tmp_path,family):
    dtype=source(tmp_path/'base',family)
    model,tokenizer,roles,config=load_native_model(tmp_path/'base',dtype=dtype)
    model.port.hf.natlang_precision='bf16'
    identity=native_identity(model,config,checkpoint_sha256=config['base_inventory_sha256'])
    participant=SemanticParticipant(model,tokenizer,config,identity)
    path=Path(__file__).resolve().parents[2]/'scripts/neuralese_data/qualify_native_joint_cuda.py'
    spec=importlib.util.spec_from_file_location('joint_cuda_diagnostic',path)
    diagnostic=importlib.util.module_from_spec(spec);spec.loader.exec_module(diagnostic)
    payload=[4,5]
    block_id,block,writer,reader,token=diagnostic.sampled_native_graph(participant,roles,payload,901)
    assert not writer['samples'] and writer['generated_inputs']
    assert reader['samples'][0]['token_id']==token
    assert token not in (roles.produce_open,roles.produce_close,roles.consume_open,roles.consume_close)
    turns=[{'graph_turn':'writer','sampling':writer,'written_nodes':{block_id:'write-0'}},
           {'graph_turn':'reader','sampling':reader,'read_bindings':{block_id:'write-0'}}]
    replay=compile_replay(turns,identity=identity,roles=roles,fetch_block=lambda key:block)
    logps,old=replay_logps(model,replay)
    torch.testing.assert_close(logps.detach(),old,rtol=2e-5,atol=2e-6)
    (-logps.mean()).backward()
    assert model.adapter.up.weight.grad is not None and model.adapter.up.weight.grad.abs().sum()>0
    assert block.token_ids.tolist()==payload
