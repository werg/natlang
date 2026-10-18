"""Real optimized-state observations keep exact ancestry through RL admission."""
import copy
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from test_native_serving_adapters import engine_fixture
from natlang_neuralese.model.native_artifact import NativeTensorArtifact,encode_native_tensor_artifact
from natlang_neuralese.model.native_state_derivation import state_values,optimize_state
from natlang_neuralese.serve.store import TensorStore,encode_native_state,native_state_id
from natlang_neuralese.serve.engine import GenerationRequest
from natlang_neuralese.rl.graph_replay import save_bundle,restore_bundle,NativeBundleStore
from natlang_neuralese.rl.native_replay import prepare_objective,NativePolicyObjective
from natlang_neuralese.train.native_state_sft import packed_graph_loss
from natlang_neuralese.data.native_graph import parse_graph


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_actual_rl_external_optimized_state_archive_restore_and_objective(tmp_path,monkeypatch,family):
    engine=engine_fixture(tmp_path,family,service_template=True)
    original=engine.native_write_text('t5 t6')['id'];ident=original;history=None
    for _ in range(2):
        block=engine.store.get_native_state(ident).block
        name=next(key for key in state_values(block) if key.endswith('.v'))
        gradient=engine.store.put_native_tensor_artifact(NativeTensorArtifact('state-gradient',engine.native_identity,
            {name:torch.full_like(state_values(block)[name],.03)},ident))
        result,_,_=optimize_state(engine,gradient,{'name':'adam','learning_rate':.01},history)
        ident=result['native_state_id'];history=result['optimizer_state_id']
    def fetch(key):
        return (encode_native_state(engine.store.get_native_state(key).block) if key.startswith('nzs1_')
            else encode_native_tensor_artifact(engine.store.get_native_tensor_artifact(key).artifact))
    # Only the latest state ID is supplied: its earlier history is discovered
    # from the shared state/tensor dependency inventories, not episode copying.
    manifest=save_bundle([{'native_input':ident}],tmp_path/'bundle',policy_id=engine.checkpoint_sha256,fetch=fetch)
    bundle=tmp_path/'bundle/manifest.json'
    assert original in manifest['blocks'] and result['update_artifact_id'] in manifest['blocks']
    prior=engine.store.get_native_state(ident).block.derivation['optimizer_state_parent_id']
    assert prior in manifest['blocks']
    reader=NativeBundleStore(bundle,policy_id=engine.checkpoint_sha256)
    assert torch.equal(reader.state(ident).tensors()[name],engine.store.get_native_state(ident).block.tensors()[name])
    assert reader._records.get() is None,'decoded episode KV must not accumulate across objective calls'
    # A new content ID authenticates bytes, not their claimed optimizer history.
    # Reject invalid derived state before replacing the prior usable manifest.
    previous_manifest=bundle.read_bytes()
    valid=engine.store.get_native_state(ident).block
    forged=replace(valid,derivation={**valid.derivation,'step':valid.derivation['step']+1})
    forged_id=native_state_id(forged);forged_bytes=encode_native_state(forged)
    with pytest.raises(ValueError):
        save_bundle([{'native_input':forged_id}],bundle.parent,policy_id=engine.checkpoint_sha256,
            fetch=lambda key:forged_bytes if key==forged_id else fetch(key))
    assert bundle.read_bytes()==previous_manifest
    assert torch.equal(NativeBundleStore(bundle,policy_id=engine.checkpoint_sha256).state(ident).tensors()[name],
                       valid.tensors()[name])
    messages=[{'role':'user','content':[{'type':'text','text':'t4 '},{'type':'native_state','id':ident},{'type':'text','text':' t7'}]}]
    episodes=[];engine.start()
    try:
        for sample,value in enumerate((7,8)):
            selected=iter([value,2])
            def token(seq):
                chosen=next(selected)
                return chosen,float(seq.logits[0].float().div(seq.request.temperature).log_softmax(-1)[chosen])
            monkeypatch.setattr(engine,'_native_token',token)
            reply=engine.generate(GenerationRequest(messages=messages,temperature=.8,seed=41+sample,max_tokens=4,rl_capture=True))
            capture=reply['x_natlang_rl_capture']
            assert next(part for part in capture['prompt'] if part['kind']=='native_state')['state_derivation']['step']==2
            episodes.append({'task_id':'explicit-train-derived-state','split':'train','sample':sample,'group':'group',
                'reward':{'quality':1-sample},'policy':{'checkpoint_sha256':engine.checkpoint_sha256},
                'turns':[{'messages':messages,'target':reply['choices'][0]['message'],'invocation_id':'reader',
                    'sampling':capture,'replay':{'raw_calls_retained':True}}]})
        from test_rl_episode_graph import attach_execution_graph
        attach_execution_graph(episodes)
        from natlang_neuralese.rl.native_stage import prepare
        engine.tokenizer.save_pretrained(tmp_path/'rl-tokenizer')
        rows,spec=prepare(episodes,checkpoint=engine.native_checkpoint,tokenizer=tmp_path/'rl-tokenizer',bundle=bundle)
        path=tmp_path/'objective.json';path.write_text(json.dumps(spec))
        objective=NativePolicyObjective(path);objective.bind_actor(engine)
        examples=[parse_graph(row,tokenizer_fingerprint=engine.native_identity['tokenizer_fingerprint']) for row in rows]
        before,_=packed_graph_loss(engine.shared_model,examples,objective=objective)
        assert torch.isfinite(before)
        from natlang_neuralese.serve.native_grad import native_graph_gradient
        from natlang_neuralese.common.hashing import sha256_file_hex_cached
        parameter=next(key for key in dict(engine.shared_model.named_parameters()) if key.endswith('v_proj.weight'))
        update=native_graph_gradient(engine,{'schema':'natlang.native-gradient-request/1','graph':rows[0],
            'target':{'kind':'model-parameters','names':[parameter]},
            'objective':{'kind':'native-policy','path':str(path),'sha256':sha256_file_hex_cached(path)}},None)
        assert engine.store.get_native_tensor_artifact(update['gradient']['id']).artifact.tensors[parameter].norm()>0
        # Restore order is adversarial and the learner store starts empty.
        manifest['blocks']=dict(reversed(list(manifest['blocks'].items())))
        bundle.write_text(json.dumps(manifest))
        engine.store.close();engine.store=TensorStore(tmp_path/'restored')
        restore_bundle(engine,bundle,policy_id=engine.checkpoint_sha256)
        assert engine.store.get_native_state(original) is not None
        assert engine.store.get_native_tensor_artifact(prior) is not None
        objective=NativePolicyObjective(path);objective.bind_actor(engine)
        after,_=packed_graph_loss(engine.shared_model,examples,objective=objective)
        assert torch.equal(before,after)
        missing=copy.deepcopy(manifest);del missing['blocks'][prior]
        # A previously populated engine cannot substitute missing bundle proof.
        with pytest.raises(ValueError,match='previous optimizer history'):
            NativeBundleStore(bundle,policy_id=engine.checkpoint_sha256,manifest=missing).state(ident)
        missing_store=TensorStore(tmp_path/'missing-store')
        try:
            with pytest.raises(ValueError,match='missing required'):
                missing_path=bundle.parent/'missing.json';missing_path.write_text(json.dumps(missing))
                restore_bundle(SimpleNamespace(native_state_enabled=True,store=missing_store),
                    missing_path,policy_id=engine.checkpoint_sha256)
        finally:missing_store.close()
        forged=copy.deepcopy(episodes)
        part=next(part for part in forged[0]['turns'][0]['sampling']['prompt'] if part['kind']=='native_state')
        part['state_derivation']['step']+=1
        with pytest.raises(ValueError,match='actual state derivation receipt'):
            prepare_objective(forged,identity=engine.native_identity,roles=engine.role_tokens,
                fetch_block=reader.state,block_bundle=bundle)
    finally:engine.stop();engine.store.close()
