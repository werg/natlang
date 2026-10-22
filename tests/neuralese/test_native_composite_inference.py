import json
from pathlib import Path

import pytest
import torch

from natlang_neuralese.common.hashing import sha256_file_hex_cached
from natlang_neuralese.model.native_bundle import write_actor_view
from natlang_neuralese.model.native_composite import write_native_composite_checkpoint
from natlang_neuralese.model.native_model import load_native_checkpoint, native_identity
from natlang_neuralese.model.shared_state import Invocation, ImportSpan, compile_graph
from natlang_neuralese.serve import load_native_engine
from natlang_neuralese.serve.native_optim import _native_optimizer_step_impl
from natlang_neuralese.train.checkpoint_policy import CheckpointPolicy


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_public_readonly_composite_chain_preserves_exact_actor_and_rejects_training(tmp_path,family):
    from test_native_serving_adapters import engine_fixture
    initial=engine_fixture(tmp_path,family)
    try:
        state=CheckpointPolicy.load(initial.native_checkpoint)
        state.update(step=71,training={'optimizer':{'moment':torch.arange(17.)},
            'rng':{'cpu':torch.get_rng_state()},'source_receipts':{'train':{'sha256':'a'*64}}})
        full=tmp_path/'full.pt';torch.save(state,full)
        stat=full.stat();ref=write_actor_view(full,tmp_path/'views',validated_state=CheckpointPolicy.load(full),
            decoded_source_stat=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns))
        proof=Path(ref['path']);view=proof.parent/json.loads(proof.read_text())['view']['path']
        parent=view
        plan=compile_graph([Invocation('p',(4,5,6,7),producing=((1,3),)),
            Invocation('c',(8,9),imports=(ImportSpan(1,'p',1,3),))])
        for generation in range(2):
            model,tokenizer,roles,config,_=load_native_checkpoint(parent,allow_weights_only=True)
            identity=native_identity(model,config,checkpoint_sha256=sha256_file_hex_cached(parent))
            with torch.no_grad():model.input_rows.add_(.02*(generation+1))
            expected={name:value.detach().clone()for name,value in model.named_parameters()}
            with torch.no_grad():expected_logits=model.logits(model(plan)['h_final'])
            child=write_native_composite_checkpoint(parent,tmp_path/('child'+str(generation)),model,config,
                base_identity=identity,training_state={'optimizer_fragment':torch.ones(3),'claimed_step':900})
            # A child metadata fragment cannot restore the missing complete root.
            parent=Path(child['path'])
            with pytest.raises(Exception,match='optimizer|pruned|complete'):
                load_native_checkpoint(parent)
            restored=load_native_engine(str(parent))
            try:
                assert restored.native_identity['checkpoint_sha256']==child['sha256']
                assert restored.native_inference_only
                for name,value in restored.shared_model.named_parameters():
                    assert torch.equal(value,expected[name])
                with torch.no_grad():actual=restored.shared_model.logits(restored.shared_model(plan)['h_final'])
                torch.testing.assert_close(actual,expected_logits,rtol=0,atol=0)
                with pytest.raises(Exception,match='complete parent checkpoint'):
                    _native_optimizer_step_impl(restored,{},'test')
                assert not any(value.requires_grad for value in restored.shared_model.parameters())
            finally:restored.store.close()
            _,_,_,_,loaded=load_native_checkpoint(parent,allow_weights_only=True)
            assert loaded['inference_only'] and loaded['optimizer_pruned']
    finally:initial.store.close()
