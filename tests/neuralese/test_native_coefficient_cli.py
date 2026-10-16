"""Real native admission/CLI/TERM/restart, plus exact controller transactions."""
import copy
import json
import os
from pathlib import Path
import signal

import pytest
import torch
from natlang_neuralese.common.paths import root as logical_root

from test_native_coefficient_stage import adapters
from test_native_serving_adapters import engine_fixture


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_actual_coefficient_stage_cli_sigterm_resume_and_real_held_collector(tmp_path,monkeypatch,family):
    from test_native_adapted_rl import capture_actor
    from natlang_neuralese.common.hashing import sha256_file_hex_cached
    from natlang_neuralese.model.native_coefficient_policy import initialize_policy,restore_policy
    from natlang_neuralese.model.native_artifact import encode_native_tensor_artifact
    from natlang_neuralese.rl.graph_replay import save_bundle
    from natlang_neuralese.rl.native_stage import main
    from natlang_neuralese.rl import native_coefficient_stage
    from natlang_neuralese.serve.engine import Engine
    from natlang_neuralese.serve.store import encode_native_state
    engine=engine_fixture(tmp_path,family,service_template=True,context=8192)
    source,target=adapters(engine)
    name=next(name for name in source.artifact.tensors if name.endswith('v_proj.weight'))
    policy=initialize_policy(engine,{source.id:[name],target.id:[name]},defaults=[source.id],
        out=tmp_path/'initial',state_transport='same-base-adapter-state')
    _,turns=capture_actor(engine,[source.id],monkeypatch,reader_entries=[target.id],explicit_state_mode=True)
    rows=[{'messages':[],'target':{},'invocation_id':turn['graph_turn'],'sampling':turn['sampling'],
           'replay':{'raw_calls_retained':True}} for turn in turns]
    episodes=[{'task_id':'coefficient-cli-train','split':'train','group':'one',
        'sample':index,'reward':{'quality':quality},'policy':{'checkpoint_sha256':engine.native_identity['checkpoint_sha256']},
        'turns':copy.deepcopy(rows)} for index,quality in enumerate((1,0))]
    from test_rl_episode_graph import attach_execution_graph
    attach_execution_graph(episodes)
    def fetch(ident):
        return encode_native_tensor_artifact(engine.store.get_native_tensor_artifact(ident).artifact) if ident.startswith('nnt1_') else encode_native_state(engine.store.get_native_state(ident).block)
    save_bundle(episodes,tmp_path/'rollout-bundle',policy_id=engine.native_identity['checkpoint_sha256'],fetch=fetch)
    rollouts=tmp_path/'rollouts.jsonl';rollouts.write_text(''.join(json.dumps(row)+'\n' for row in episodes))
    engine.tokenizer.save_pretrained(tmp_path/'tokenizer')
    held=tmp_path/'held.jsonl';held.write_text(json.dumps({'schema':'natlang.neuralese-native-graph/1','id':'protected-cli-held',
        'split':'test','source_groups':['protected-coefficient-cli-held'],
        'tokenizer_fingerprint':engine.native_identity['tokenizer_fingerprint'],
        'nodes':[{'name':'held','token_ids':[4,5,6],'target_weights':[0,1,1]}]})+'\n')
    packet=tmp_path/'held-task.jsonl'
    original=logical_root('repo')/'runs/natlang-held-task-packet-v3-20261012/cases.jsonl'
    packet.write_text(original.open().readline())
    spec={'packet':{'path':str(packet),'sha256':sha256_file_hex_cached(packet)},'cases':[0],
        'workers':1,'noise_repeats':1,'neuralese_server':True,'runtime':str(Path(__file__).resolve().parents[2]/'ts-host'),
        'max_model_requests':1,'max_turns':4,'case_seconds':120,'min_successes':0}
    out=tmp_path/'stage';base=Path(engine.native_checkpoint);base_sha=sha256_file_hex_cached(base)
    argv=['--rollouts',str(rollouts),'--rl-tokenizer',str(tmp_path/'tokenizer'),
        '--rl-block-bundle',str(tmp_path/'rollout-bundle/manifest.json'),
        '--coefficient-policy',str(policy),'--continue-from',str(base),'--held-native-graphs',str(held),
        '--out',str(out),'--device',str(engine.device),'--epochs','1',
        '--coefficient-optimizer',json.dumps({'name':'adam','learning_rate':.01}),
        '--task-eval',json.dumps(spec),'--task-eval-every','100','--held-eval-graphs','1']
    # Termination arrives during an actual graph adjoint. The installed CLI
    # handler lets the complete whole-policy update commit, then exits cleanly.
    gradient=native_coefficient_stage.native_coefficient_gradients
    calls=[]
    def terminate_after_graph(*args,**kwargs):
        result=gradient(*args,**kwargs);calls.append(result)
        os.kill(os.getpid(),signal.SIGTERM)
        return result
    monkeypatch.setattr(native_coefficient_stage,'native_coefficient_gradients',terminate_after_graph)
    # The fixture bounds emitted text while retaining actual logits and native
    # prefill. It does not claim task success or a learned behavioral result.
    monkeypatch.setattr(Engine,'_native_token',lambda self,sequence:(2,float(sequence.logits[0].float().log_softmax(-1)[2])))
    from natlang_neuralese.rl.native_coefficient_stage import bind_policy
    from natlang_neuralese.eval.task_cadence import TaskEval,parse_task_eval
    bind_policy(engine,policy)
    monkeypatch.setattr(engine,'_native_token',lambda sequence:(2,float(sequence.logits[0].float().log_softmax(-1)[2])))
    baseline=TaskEval(parse_task_eval(spec),tmp_path/'initial-task-reference',engine,prompt_rendering='full').run(0)
    assert baseline['eligible'] is True and baseline['outcome_counts']['infrastructure_error']==0
    spec['reference_successes']=baseline['task_successes']
    argv[argv.index('--task-eval')+1]=json.dumps(spec)
    try:
        assert main(argv)==0
        partial=json.loads((out/'report.json').read_text())
        assert len(calls)==1 and partial['step']==1 and not partial['completed']
        assert partial['evaluation'] is None
        monkeypatch.setattr(native_coefficient_stage,'native_coefficient_gradients',gradient)
        assert main(argv)==0
        result=json.loads((out/'report.json').read_text())
        assert result['step']==2 and result['completed'] and result['qualification_complete']
        assert result['this_stage_updates']==2
        assert result['evaluation']['task']['eligible'] is True
        assert result['evaluation']['held']['loss']>0
        assert result['evaluation']['task']['coefficient_policy_sha256']==result['coefficient_policy_sha256']
        restored=restore_policy(engine,result['policy']['path'])
        assert all(engine.store.get_native_tensor_artifact(entry['optimizer_state_id']).artifact.provenance['step']==2
                   for entry in restored['handles'].values())
        assert main(argv)==0
        repeated=json.loads((out/'report.json').read_text())
        assert repeated['step']==2 and repeated['coefficient_policy_sha256']==result['coefficient_policy_sha256']
        assert sha256_file_hex_cached(base)==base_sha
        # Two real collect/admit/learn cycles through the existing controller.
        # Only task reward/selection and endpoint publication are fixture data;
        # sampling/cache receipts, admission, CLI, held/task execution, gradients,
        # optimizer, checkpoint and reload all use the actual implementations.
        from natlang_neuralese.rl import production
        from natlang_neuralese.rl.native_coefficient_stage import bind_policy
        controlled=tmp_path/'controller';cases=tmp_path/'train-cases.jsonl'
        cases.write_text(json.dumps({'id':'coefficient-cli-train','split':'train'})+'\n')
        export=tmp_path/'fixture-semantic-identity.json'
        export.write_text(json.dumps({'schema':'natlang.native-semantic-adapter/1',
            'checkpoint':{'path':str(base),'sha256':base_sha},
            'models':{family:{'native_identity':engine.native_identity,
                'checkpoint':{'path':str(base),'sha256':base_sha,'participant':family}}},
            'tokenizers':{family:{'path':'tokenizer'}}}))
        config=tmp_path/'controller.json'
        config.write_text(json.dumps({'out':str(controlled),'checkpoint':str(base),
            'native_state':True,'native_participant':family,'semantic_adapter':str(export),
            'coefficient_policy':result['policy']['path'],'cases':str(cases),
            'endpoints':['fixture-actual-engine'],'rows_per_round':1,
            'learner_command':['fixture-real-native-cli'],'after_update':['fixture-publication']}))
        collected=[];learned=[];publications=[]
        def collect(**kwargs):
            destination=Path(kwargs['out']);destination.mkdir(parents=True,exist_ok=True)
            saved=destination/'score.json'
            if saved.exists():
                receipt=json.loads(saved.read_text())
                assert receipt['coefficient_policy']['path']==kwargs['coefficient_policy'],'publication retry retains sealed original actor'
                return receipt
            current=bind_policy(engine,kwargs['coefficient_policy'])
            writer=current['handles'][source.id]['current_id'];reader=current['handles'][target.id]['current_id']
            _,actual_turns=capture_actor(engine,[writer],monkeypatch,reader_entries=[reader],explicit_state_mode=True)
            actual_rows=[{'messages':[],'target':{},'invocation_id':turn['graph_turn'],'sampling':turn['sampling'],
                          'replay':{'raw_calls_retained':True}} for turn in actual_turns]
            actual_episodes=[{**episode,'turns':copy.deepcopy(actual_rows)} for episode in episodes]
            # Updated actors produce different immutable block IDs. Their
            # whole-episode occurrence inventory belongs to these fresh captures.
            attach_execution_graph(actual_episodes)
            save_bundle(actual_episodes,destination/'blocks',policy_id=base_sha,fetch=fetch)
            (destination/'rollouts.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in actual_episodes))
            receipt={'scored':2,'successes':1,'groups_with_spread':1,'block_bundle':str(destination/'blocks/manifest.json'),
                'coefficient_policy':{'path':str(kwargs['coefficient_policy']),'sha256':sha256_file_hex_cached(kwargs['coefficient_policy'])},
                'groups':[{'task_id':'coefficient-cli-train','scored':2,'mean':.5}]}
            saved.write_text(json.dumps(receipt));collected.append(writer)
            return receipt
        def command(command,values):
            if command[0]=='fixture-publication':
                publications.append(values['coefficient_policy'])
                if len(publications)==1:raise RuntimeError('fixture publication interrupted')
                return
            assert command==['fixture-real-native-cli']
            learned.append(values['learner_out'])
            actual=list(argv)
            for option,value in (('--rollouts',values['rollouts']),('--rl-block-bundle',values['block_bundle']),
                ('--coefficient-policy',values['coefficient_policy']),('--out',values['learner_out'])):
                actual[actual.index(option)+1]=str(value)
            assert main(actual)==0
        monkeypatch.setattr(production,'collect_with_actor',lambda config,values,**kwargs:collect(**kwargs))
        monkeypatch.setattr(production,'run_command',command)
        with pytest.raises(RuntimeError,match='publication interrupted'):production.run(config,rounds=1)
        first=production.run(config,rounds=1)
        assert first['round']==1 and first['trained_rounds']==1 and len(learned)==1
        second=production.run(config,rounds=1)
        assert second['round']==2 and second['trained_rounds']==2 and len(learned)==2
        assert len(collected)==2 and collected[0]!=collected[1]
        assert second['checkpoint']==str(base) and 'pending_native_checkpoint' not in second
        assert len(publications)==3,'publication retry does not train a completed update twice'
        latest=restore_policy(engine,second['coefficient_policy'])
        assert all(engine.store.get_native_tensor_artifact(entry['optimizer_state_id']).artifact.provenance['step']==6
                   for entry in latest['handles'].values())
        reports=[json.loads((Path(path)/'report.json').read_text()) for path in learned]
        assert all(report['this_stage_updates']==2 for report in reports)
        assert second['trained_target_tokens']==sum(report['train_exposure']['target_tokens'] for report in reports)
        assert sha256_file_hex_cached(base)==base_sha
    finally:engine.store.close()


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_actual_managed_actor_policy_cli_and_durable_restart(tmp_path,family):
    import subprocess
    import sys
    import time
    import urllib.request
    from natlang_neuralese.common.hashing import sha256_file_hex_cached
    from natlang_neuralese.model.native_coefficient_policy import initialize_policy,runtime_binding,policy_core,publish_policy
    engine=engine_fixture(tmp_path,family,service_template=True)
    actor,_=adapters(engine)
    name=next(name for name in actor.artifact.tensors if name.endswith('v_proj.weight'))
    policy=initialize_policy(engine,{actor.id:[name]},defaults=[actor.id],out=tmp_path/'policy')
    from dataclasses import replace
    changed=engine.store.put_native_tensor_artifact(replace(actor.artifact,tensors={key:value+.01 for key,value in actor.artifact.tensors.items()}))
    core=policy_core(json.loads(policy.read_text()));core['handles'][actor.id]['current_id']=changed.id
    policy=publish_policy(engine,core,tmp_path/'current-policy')
    expected=runtime_binding(json.loads(policy.read_text()),policy)
    fixture=tmp_path/'runtime-fixture.json'
    fixture.write_text(json.dumps({'runtime':expected,'handle':actor.id,'current':changed.id,
                                  'defaults':[{'id':changed.id,'scale':1}]}))
    command=[sys.executable,str(Path(__file__).resolve().parents[2]/'scripts/neuralese_data/serve_direct_checkpoint.py'),
        '--native-state','--checkpoint',engine.native_checkpoint,
        '--checkpoint-sha256',sha256_file_hex_cached(engine.native_checkpoint),
        '--coefficient-policy',str(policy),'--coefficient-policy-sha256',sha256_file_hex_cached(policy),
        '--device',str(engine.device),'--port','0','--store-dir',str(tmp_path/'durable-store')]
    process=None
    def start(index):
        log=tmp_path/f'actor-{index}.log'
        with log.open('w') as stream:
            child=subprocess.Popen(command,stdout=stream,stderr=stream,env={**os.environ,'OMP_NUM_THREADS':'2'})
        deadline=time.monotonic()+60
        while time.monotonic()<deadline:
            for line in log.read_text().splitlines():
                if line.startswith('{'):
                    receipt=json.loads(line)
                    if receipt.get('event')=='direct_training_checkpoint_serving':return child,receipt
            if child.poll() is not None:raise AssertionError(log.read_text())
            time.sleep(.05)
        child.terminate();child.wait(30)
        raise AssertionError('managed fixture did not report readiness: '+log.read_text())
    def request(url,body=None):
        data=None if body is None else json.dumps(body).encode()
        with urllib.request.urlopen(urllib.request.Request(url,data=data,headers={'content-type':'application/json'}),timeout=30) as response:
            return response.read()
    try:
        process,receipt=start(1)
        info=json.loads(request(receipt['listening']+'/v1/neuralese/info'))
        assert info['rl_policy_binding']['coefficient_policy']==expected
        runtime=subprocess.run(['node',str(Path(__file__).resolve().parents[2]/'ts-host/test/fixtures/native-coefficient-runtime.mjs'),
            receipt['listening'],str(fixture)],capture_output=True,text=True)
        assert runtime.returncode==0,runtime.stdout+'\n'+runtime.stderr
        result=json.loads(request(receipt['listening']+'/v1/neuralese/native-states/write-text',{'text':'t5 t6','adapters':[changed.id]}))
        raw=request(receipt['listening']+'/v1/neuralese/native-states/'+result['id'])
        from natlang_neuralese.serve.store import decode_native_state
        block,ident=decode_native_state(raw)
        assert ident==result['id'] and block.adapter_binding['adapters'][0]['id']==changed.id
        process.terminate();process.wait(30);process=None
        process,receipt=start(2)
        assert request(receipt['listening']+'/v1/neuralese/native-states/'+ident)==raw
        assert json.loads(request(receipt['listening']+'/v1/neuralese/info'))['rl_policy_binding']['coefficient_policy']==expected
    finally:
        if process is not None and process.poll() is None:process.terminate();process.wait(30)
        engine.store.close()
