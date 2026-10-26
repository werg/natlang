"""A current policy owns moments/RNG; its carried stage owns the finite cohort."""
import copy
import json
from pathlib import Path

import pytest
import torch

from test_native_model import tiny_source
from test_native_state_sft import graph, authenticated_pair, source_manifest
from natlang_neuralese.model.native_model import load_native_model
from natlang_neuralese.train import native_state_sft as trainer
from natlang_neuralese.train.native_continuation import collect_sft_progress


@pytest.mark.parametrize('mode,stage', [
    ('policy', 'full'), ('ordinary', 'full'), ('policy', None), ('recovered_declaration', 'full'),
    ('wrong_stage', 'full'), ('missing_flag', 'full'), ('unrelated_top_flag', 'full')])
def test_real_resume_uses_only_the_requested_carried_foundation(tmp_path, mode, stage):
    base=tmp_path/'base';tiny_source(base)
    _,_,roles,config=load_native_model(base,dtype=torch.float32,rank=4)
    fp=config['tokenizer_fingerprint']
    original=tmp_path/'original';original.mkdir()
    ordinary=[dict(graph('ordinary'),tokenizer_fingerprint=fp)]
    old_pair=authenticated_pair('old',roles=roles.ids,fingerprint=fp)
    manifest=source_manifest(original,ordinary,[old_pair])
    foundation_directory=tmp_path/'foundation';foundation_directory.mkdir()
    foundation_manifest=source_manifest(foundation_directory,[],[old_pair])
    foundation_value=json.loads(foundation_manifest.read_text())
    foundation_value['sources']=[item for item in foundation_value['sources'] if item['kind']=='semantic-pairs']
    foundation_manifest.write_text(json.dumps(foundation_value))
    held=tmp_path/'held.jsonl'
    held.write_text(json.dumps(dict(graph('held'),split='test',tokenizer_fingerprint=fp))+'\n')
    args=['--base',str(base),'--dtype','float32','--rank','4','--device','cpu',
        '--native-graphs',str(manifest),'--foundation-native-graphs',str(foundation_manifest),
        '--communication-foundation','--held-native-graphs',str(held),'--epochs','1',
        '--eval-every','100','--out',str(tmp_path/'first'),
        '--forward-atol','1e-4','--gradient-atol','1e-5']
    if stage is not None:args+=['--stage-id',stage]
    assert trainer.main(args)==0
    first=torch.load(tmp_path/'first/checkpoint.pt',weights_only=False)
    source_frame=first['training']
    key=stage or 'native-graph-ce'
    assert source_frame['exposure_counts']=={'ordinary':0,'lfm:old':1}
    old_report=json.loads((tmp_path/'first/report.json').read_text())
    old_cohort=old_report['communication_foundation']['cohort_sha256']
    carried=collect_sft_progress(source_frame)
    carried[key]['completed_foundations']=[old_cohort]
    parent=copy.deepcopy(first)
    recovery_path=None
    if mode=='recovered_declaration':
        from natlang_neuralese.common.hashing import sha256_file_hex
        def bind(path):return {'path':str(path),'sha256':sha256_file_hex(path)}
        checkpoint={'sha256':sha256_file_hex(tmp_path/'first/checkpoint.pt'),
                    'bytes':(tmp_path/'first/checkpoint.pt').stat().st_size}
        historical_recipe=tmp_path/'historical-recipe.json'
        historical_recipe.write_text(json.dumps({'input_bindings':{'foundation':bind(foundation_manifest)},
            'stages':[{'id':key,'kind':'native_state_sft','inputs':{'foundation_native_graphs':'foundation'},
                       'parameters':{'communication_foundation':True,'epochs':1.}}]}))
        historical_metadata=tmp_path/'historical-metadata.json'
        historical_metadata.write_text(json.dumps({'checkpoint_identity':checkpoint,'model_config':first['model_config'],
            'training':{'global_step':source_frame['global_step'],'options':source_frame['options']}}))
        verification=tmp_path/'historical-checkpoint-verification.json'
        verification.write_text(json.dumps({'passed':True,'checkpoint':checkpoint,'stage_id':key,
            'architecture':config['architecture'],'global_step':source_frame['global_step'],
            'optimizer_present':True,'rng_present':True}))
        recovery_path=tmp_path/'foundation-declaration-recovery.json'
        recovery_path.write_text(json.dumps({'schema':'natlang.native-foundation-declaration-recovery/1',
            'stage_id':key,'historical_recipe':bind(historical_recipe),
            'historical_training_metadata':bind(historical_metadata),
            'historical_checkpoint_verification':bind(verification),
            'foundation_sources':bind(foundation_manifest),'reason':'Restore exact unavailable historical declaration.'}))
        parent['training']['options']['native_snapshot_ancestors']=[checkpoint['sha256']]
        carried[key]['options']['communication_foundation']=False
        carried[key]['options']['foundation_native_graphs']=None
    if mode!='ordinary':
        top=parent['training']
        top['identity']['objective']={'kind':'native-policy','sha256':'a'*64}
        top['options']['stage_id']='native_rl'
        top['options']['communication_foundation']=mode=='unrelated_top_flag'
        top['sft_progress']=carried
        if mode=='wrong_stage':top['sft_progress']={'another_stage':carried[key]}
        if mode in ('missing_flag','unrelated_top_flag'):
            top['sft_progress'][key]['options'].pop('communication_foundation')
    else:
        parent['training']['completed_foundations']=[old_cohort]
    parent_path=tmp_path/'parent.pt';torch.save(parent,parent_path)
    expanded=tmp_path/'expanded';expanded.mkdir()
    expanded_manifest=source_manifest(expanded,ordinary,[old_pair,
        authenticated_pair('new',roles=roles.ids,fingerprint=fp)])
    run=tmp_path/'resumed';continued=list(args)
    continued[continued.index('--native-graphs')+1]=str(expanded_manifest)
    continued[continued.index('--out')+1]=str(run)
    continued[continued.index('--eval-every')+1]='1'
    continued+=['--continue-from',str(parent_path),'--objective-continuation',
        '--continued-native-graphs',str(manifest),'--refresh-at-checkpoint','--checkpoint-every','1']
    if recovery_path:continued+=['--foundation-declaration-recovery',str(recovery_path)]
    if mode in ('wrong_stage','missing_flag','unrelated_top_flag'):
        with pytest.raises(ValueError,match='entire currently declared communication cohort'):
            trainer.main(continued)
        assert not (run/'checkpoint.pt').exists()
        return
    assert trainer.main(continued)==0
    saved=torch.load(run/'checkpoint.pt',weights_only=False)
    frame=saved['training'];report=json.loads((run/'report.json').read_text())
    assert frame['global_step']==source_frame['global_step']+1
    assert frame['exposure_counts']['lfm:old']==1
    assert sum(frame['exposure_counts'].values())==2
    assert frame['exposure_counts']['lfm:new']==1
    assert frame['exposure_counts']['ordinary']==0
    assert report['communication_foundation']['cohort_sha256']==old_cohort
    assert not report['foundation_ready']
    assert not report['completed']
    assert report['mandatory_sft']['remaining_graph_visits']==1
    assert any(not torch.equal(first['backbone_state'][name],value)
               for name,value in saved['backbone_state'].items())
    assert frame['optimizer']
    if mode=='recovered_declaration':
        history=frame['options']['foundation_declaration_recovery_history']
        assert len(history)==1 and history[0]['new_exposure']==0
        assert history[0]['qualification_granted'] is False
        assert history[0]['cohort_sha256']==old_cohort
    if mode=='policy' and stage=='full':
        # The same actual weights/moments/RNG/credits through a direct ordinary
        # parent must produce the identical next update. No fake policy update
        # is claimed by the metadata-only policy-top fixture above.
        control=copy.deepcopy(first)
        control['training']['completed_foundations']=[old_cohort]
        control_path=tmp_path/'ordinary-parent.pt';torch.save(control,control_path)
        control_args=list(continued)
        control_args[control_args.index('--continue-from')+1]=str(control_path)
        control_args[control_args.index('--out')+1]=str(tmp_path/'control')
        assert trainer.main(control_args)==0
        expected=torch.load(tmp_path/'control/checkpoint.pt',weights_only=False)
        def equal(left,right):
            if torch.is_tensor(left):torch.testing.assert_close(left,right,rtol=0,atol=0)
            elif isinstance(left,dict):
                assert left.keys()==right.keys()
                for name in left:equal(left[name],right[name])
            elif isinstance(left,(tuple,list)):
                assert len(left)==len(right)
                for a,b in zip(left,right):equal(a,b)
            else:assert left==right
        for section in ('backbone_state','interface_state'):equal(saved[section],expected[section])
        for section in ('optimizer','rng','exposure_counts'):
            equal(frame[section],expected['training'][section])


def test_fresh_incomplete_foundation_stays_strict(tmp_path):
    from types import SimpleNamespace
    from natlang_neuralese.train.native_continuation import verify_initial_foundation_cohort
    train=SimpleNamespace(entries=[{'id':'old','binding':'a','communication':True},
                                   {'id':'new','binding':'b','communication':True}])
    foundation=SimpleNamespace(entries=train.entries[:1])
    with pytest.raises(ValueError,match='entire currently declared communication cohort'):
        verify_initial_foundation_cohort(train,foundation)


@pytest.mark.parametrize("acknowledged", [True, False])
def test_inherited_foundation_requalifies_after_completed_new_source_boundary(tmp_path, monkeypatch, acknowledged):
    """A fixed earned cohort gets one fresh proof after new supervised work."""
    from natlang_neuralese.eval import native_backend, native_state, task_cadence
    from natlang_neuralese.serve import engine as engine_module, store as store_module
    from natlang_neuralese.train.native_state_sft import _foundation_readiness_boundary

    cohort={'complete':True,'cohort_sha256':'a'*64}
    prior={cohort['cohort_sha256']}
    assert not _foundation_readiness_boundary(cohort,prior,completed=False,
        new_optimizer_updates=1,policy_operation=False,signaled=False)
    assert not _foundation_readiness_boundary(cohort,prior,completed=True,
        new_optimizer_updates=0,policy_operation=False,signaled=False)
    assert not _foundation_readiness_boundary(cohort,prior,completed=True,
        new_optimizer_updates=1,policy_operation=True,signaled=False)
    assert _foundation_readiness_boundary(cohort,prior,completed=False,
        new_optimizer_updates=0,policy_operation=False,signaled=False,refresh_requested=True)
    assert _foundation_readiness_boundary(cohort,prior,completed=True,
        new_optimizer_updates=1,policy_operation=False,signaled=False)

    calls=[]
    def qualify(model,tokenizer,roles,model_config,**kwargs):
        checkpoint_sha=kwargs['checkpoint_sha256'];calls.append(checkpoint_sha)
        identity=trainer.native_identity(model,model_config,checkpoint_sha256=checkpoint_sha)
        return {'passed':True,'native_identity':identity,'backbone_optimizer_tested':True,
                'backbone_optimizer_updates':kwargs['backbone_optimizer_updates']}
    monkeypatch.setattr(native_backend,'qualify_complete_runtime',qualify)

    class FakeEngine:
        native_state_enabled=False
        def __init__(self,*args,**kwargs): self.__dict__.update(kwargs)

    task_calls=[]
    class FakeTaskEval:
        def __init__(self,spec,out,engine,*,prompt_rendering,stop_requested):
            self.engine=engine
            self.stop_requested=stop_requested
            assert callable(stop_requested)
        def cadence(self,step,*,final):
            assert self.stop_requested() is False
            task_calls.append((step,final))
            return {'run':'fixture','step':step,'eligible':True,'passed':True,
                    'task_successes':0,'outcome_counts':{'invalid_action':1}}

    monkeypatch.setattr(engine_module,'Engine',FakeEngine)
    monkeypatch.setattr(store_module,'TensorStore',lambda:object())
    monkeypatch.setattr(task_cadence,'TaskEval',FakeTaskEval)
    monkeypatch.setattr(native_state,'task_execution_receipt',lambda summary,identity,*,checkpoint_sha256:{
        'schema':'natlang.native-task-execution-receipt/1','passed':True,'complete':True,
        'eligible':True,'native_identity':identity})

    base=tmp_path/'base';tiny_source(base)
    _,_,roles,config=load_native_model(base,dtype=torch.float32,rank=4)
    fp=config['tokenizer_fingerprint']
    old_pair=authenticated_pair('foundation',roles=roles.ids,fingerprint=fp)
    original=tmp_path/'original';original.mkdir()
    original_manifest=source_manifest(original,[dict(graph('ordinary-a'),tokenizer_fingerprint=fp)],[old_pair])
    held=tmp_path/'held.jsonl'
    held.write_text(json.dumps(dict(graph('held'),split='test',tokenizer_fingerprint=fp))+'\n')
    packet=tmp_path/'packet.jsonl';packet.write_text('{}\n')
    task_spec=json.dumps({'packet':{'path':str(packet),'sha256':'0'*64},'cases':[0]})
    first_out=tmp_path/'first'
    args=['--base',str(base),'--dtype','float32','--rank','4','--device','cpu',
        '--native-graphs',str(original_manifest),'--foundation-native-graphs',str(original_manifest),
        '--communication-foundation','--held-native-graphs',str(held),'--epochs','1',
        '--stage-id','full','--eval-every','1','--task-eval',task_spec,'--task-eval-every','0','--out',str(first_out),
        '--forward-atol','1e-4','--gradient-atol','1e-5','--checkpoint-every','1']
    assert trainer.main(args)==0
    first=torch.load(first_out/'checkpoint.pt',weights_only=False)
    first_report=json.loads((first_out/'report.json').read_text())
    cohort_sha=first_report['communication_foundation']['cohort_sha256']
    frame=collect_sft_progress(first['training'])
    if acknowledged:
        frame['full']['completed_foundations']=[cohort_sha]
    first['training']['sft_progress']=frame
    parent=tmp_path/'parent.pt';torch.save(first,parent)

    expanded=tmp_path/'expanded';expanded.mkdir()
    ordinary_a=expanded/'ordinary-a.jsonl'
    pairs=expanded/'foundation-pair.jsonl'
    ordinary_b=expanded/'ordinary-b.jsonl'
    ordinary_a.write_bytes((original/'ordinary.jsonl').read_bytes())
    pairs.write_bytes((original/'pairs.jsonl').read_bytes())
    ordinary_b.write_text(json.dumps(dict(graph('ordinary-b'),tokenizer_fingerprint=fp))+'\n')
    from natlang_neuralese.common.hashing import sha256_file_hex
    expanded_manifest=expanded/'source.json'
    expanded_manifest.write_text(json.dumps({'schema':'natlang.native-graph-sources/1','sources':[
        {'kind':'native-graphs','path':ordinary_a.name,'sha256':sha256_file_hex(ordinary_a)},
        {'kind':'semantic-pairs','path':pairs.name,'sha256':sha256_file_hex(pairs),'model':'lfm'},
        {'kind':'native-graphs','path':ordinary_b.name,'sha256':sha256_file_hex(ordinary_b)}]}))
    resumed=tmp_path/'resumed'
    continued=list(args)
    continued[continued.index('--native-graphs')+1]=str(expanded_manifest)
    continued[continued.index('--out')+1]=str(resumed)
    continued += ['--continue-from',str(parent),'--objective-continuation',
                  '--continued-native-graphs',str(original_manifest),'--refresh-at-checkpoint',
                  '--refresh-after-updates','1','--refresh-foundation-readiness']
    before=len(calls);before_task=len(task_calls)
    assert trainer.main(continued)==0
    report=json.loads((resumed/'report.json').read_text())
    receipt=json.loads((resumed/'native-training-receipt.json').read_text())
    assert report['communication_foundation']['cohort_sha256']==cohort_sha
    assert report['foundation_ready'] is True and receipt['foundation_ready'] is True
    assert report['completed'] is False
    assert report['mandatory_sft']['complete'] is False
    assert report['mandatory_sft']['remaining_graph_visits'] > 0
    assert report['continuation_boundary']['new_optimizer_updates']==1
    runtime=json.loads(Path(report['runtime_certificate']['path']).read_text())
    assert runtime['native_identity']['checkpoint_sha256']==report['checkpoint_sha256']
    assert len(calls)-before==1  # one terminal qualification, not one per checkpoint/update
    assert calls[-1]==report['checkpoint_sha256']
    assert task_calls[before_task:]==[(report['task_eval']['step'],True)]
