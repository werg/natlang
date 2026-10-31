import hashlib
import json
from pathlib import Path
import pytest
from natlang_neuralese.rl.amended_collection import resume_amended_collection


def test_interrupted_group_projection_is_authenticated(tmp_path):
    out=tmp_path/'round';out.mkdir();policy='a'*64
    def bind(path):return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}
    def write(name,value):
        p=out/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value)+'\n');return p
    cases=write('selected-cases.jsonl',{'id':'task'})
    binding={'schema':'natlang.rl-collection-binding/1','policy_id':policy,'cases_sha256':bind(cases)['sha256'],
        'rows':1,'offset':0,'group':2,'root_mode':'student','seed_base':700000,'case_seconds':600,
        'max_model_requests':128,'max_turns':160,'neuralese_server':False,'temperature':1.,'guided':False,'endpoints':['actor']}
    write('collection-binding.json',binding)
    write('score.json',{'collection_complete':False,'groups':[{'task_id':'task','scored':2,'spread':True}],
         'actor_receipts':{}})
    for name in ['packet.jsonl','journal.jsonl','preparation.json','sampling-config.json']:write(name,{})
    entries=[];rows=[]
    for sample in range(2):
        result=write(f'result-{sample}.jsonl',{})
        entries.append({'group':0,'sample':sample,'index':0,'task_id':'task','output':str(result)})
        rows.append({'group':0,'sample':sample,'task_id':'task','policy':{'checkpoint_sha256':policy},
                     'reward':{'quality':sample},'turns':[{}]})
    (out/'queue.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in entries))
    (out/'rollouts.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
    projected=tmp_path/'projection.jsonl';projected.write_bytes((out/'rollouts.jsonl').read_bytes())
    names=['collection-binding.json','score.json','rollouts.jsonl','packet.jsonl','queue.jsonl','journal.jsonl',
           'preparation.json','sampling-config.json','result-0.jsonl','result-1.jsonl']
    proof=tmp_path/'proof.json';proof.write_text(json.dumps({'schema':'natlang.amended-collection-continuation/1',
         'out':str(out),'groups':[0],'files':{n:bind(out/n) for n in names},'filtered_rollouts':bind(projected)}))
    request={'out':out,'cases':cases,'policy_id':policy,'rows':1,'group':2,'endpoints':['actor'],'neuralese_server':False}
    result=resume_amended_collection(bind(proof),request,validate_actors=lambda *a,**kw:None)
    assert result['scored']==2 and result['groups_with_spread']==1 and result['collection_complete'] is False
    rows[0]['reward']['quality']=1;projected.write_text(''.join(json.dumps(x)+'\n' for x in rows))
    p=json.loads(proof.read_text());p['filtered_rollouts']=bind(projected);proof.write_text(json.dumps(p))
    with pytest.raises(ValueError,match='projection'):
        resume_amended_collection(bind(proof),request,validate_actors=lambda *a,**kw:None)

def test_policy_arguments_do_not_apply_sft_communication_quota():
    from natlang_neuralese.rl.production import native_policy_trainer_args
    from natlang_neuralese.train.recipe import stage_parameter_args
    parameters={'communication_min_visits':7,'pending_focus':'focus.json','sft_packed_geometry':{'path':'sft'},'foundation_communication_batches':1,'learning_rate':1e-6}
    args=native_policy_trainer_args(parameters,{})
    assert all(flag not in args for flag in ('--communication-min-visits','--pending-focus','--sft-packed-geometry','--foundation-communication-batches'))
    assert '--communication-min-visits' in stage_parameter_args(parameters)
    assert parameters['communication_min_visits']==7

def test_policy_quota_frame_excludes_prior_sft_obligations():
    from natlang_neuralese.train.native_continuation import policy_pending_frame
    frame={'source_row_bindings':{'policy':'binding'},'source_row_order':['policy'],
           'exposure_counts':{},'options':{'epochs':1,'_objective_required_visits':{'old-sft':7},
           '_pending_native_source_declarations':[{'binding':{'sha256':'old'},'rows':{}}]}}
    projected=policy_pending_frame(frame)
    assert projected['required_visits']=={'policy':1}
    assert not projected.get('pending_source_declarations')
    assert frame['options']['_objective_required_visits']=={'old-sft':7}

def test_policy_cli_rejects_direct_sft_override_and_unknown_control():
    from natlang_neuralese.rl.production import validate_native_policy_trainer_cli
    validate_native_policy_trainer_cli(['--batch-tokens','32768','--async-checkpoint'])
    for flag in ('--pending-focus','--sft-packed-geometry','--communication-min-visits','--unknown-control'):
        with pytest.raises(ValueError,match='unclassified native RL trainer flag'):
            validate_native_policy_trainer_cli([flag,'x'])
