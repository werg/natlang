"""Task references are operational recipe changes, never current qualifications."""
import copy,json,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/coreweave'))
import prepare_native_task_reference as m
from natlang_neuralese.common.jsonio import write_json_durable as write


def fixture(tmp):
    packet={'path':'/work/native-production/held/cases.jsonl','sha256':'a'*64}
    policy={'packet':packet,'cases':[0,5],'case_seconds':7200,'workers':2,'max_turns':160,
        'max_model_requests':128,'reference_successes':None,'runtime':'/work/native-runtime/02b/ts-host'}
    recipe=tmp/'recipe.json';value={'id':'actual-source-fixture','source_provenance':{'unchanged':True},
        'stages':[{'id':name,'kind':'native_state_sft','parameters':{'task_eval':copy.deepcopy(policy),
            'task_eval_every':0,'learning_rate':1e-5,'batch_tokens':65536}}for name in ('full','later')]}
    write(recipe,value)
    result={'historical':True,'qualifies_current_weights':False,'checkpoint_sha256':'c'*64,'step':24575,
        'result':{'schema':'natlang.in-run-task-eval/1','eligible':True,'complete':False,'step':24575,
            'task_successes':1,'cases':2,'case_indices':[0,5],'packet_sha256':'a'*64,
            'evaluation_provenance':{'checkpoint':{'sha256':'c'*64},'evaluation_spec_sha256':'d'*64,
                'model_config':{'architecture':'mellum','base_inventory_sha256':'b'*64,'dtype':'torch.bfloat16','native_shared_execution':'stable-v1'}}}}
    source=tmp/'observation.json';write(source,result)
    return recipe,source,value,result


@pytest.mark.parametrize('role',['lfm','mellum'])
def test_only_reference_fields_change_all_actual_stage_policies(tmp_path,role):
    recipe,source,original,_=fixture(tmp_path)
    receipt=m.prepare(m.binding(recipe),role,m.binding(source),tmp_path/'prepared',owner_reason='Explicit supervisor-selected continuation floor; actual current checks remain mandatory.')
    future=json.loads(Path(receipt['recipe']['path']).read_text())
    for before,after in zip(original['stages'],future['stages'],strict=True):
        selected=after['parameters']['task_eval']
        assert selected.pop('reference_successes')==(0 if role=='lfm'else 1)
        proof=selected.pop('reference_provenance');assert proof['qualifies_current_weights'] is proof['base_measurement'] is False
        before['parameters']['task_eval'].pop('reference_successes')
        assert before==after
    assert receipt['optimizer_updates']==receipt['new_exposure']==receipt['checkpoint_decodes']==0
    assert receipt['activated'] is False


@pytest.mark.parametrize('fault',['current','wrong-architecture','wrong-checkpoint','wrong-count','packet','cases','bytes'])
def test_wrong_or_unbound_observation_fails_before_recipe_write(tmp_path,fault):
    recipe,source,original,result=fixture(tmp_path)
    if fault=='current':result['qualifies_current_weights']=True
    elif fault=='wrong-architecture':result['result']['evaluation_provenance']['model_config']['architecture']='lfm2'
    elif fault=='wrong-checkpoint':result['checkpoint_sha256']='0'*64
    elif fault=='wrong-count':result['result']['task_successes']=True
    elif fault=='packet':result['result']['packet_sha256']='e'*64
    elif fault=='cases':result['result']['case_indices']=[0,6]
    write(source,result);ref=m.binding(source)
    if fault=='bytes':ref['bytes']=0
    with pytest.raises(ValueError):m.prepare(m.binding(recipe),'mellum',ref,tmp_path/'forbidden',owner_reason='selected actual floor')
    assert not(tmp_path/'forbidden').exists()


@pytest.mark.parametrize('fault',[None,'summary','hash','bytes'])
def test_portable_evidence_is_exact_small_derived_snapshot(tmp_path,fault):
    recipe,source,_,_=fixture(tmp_path);reason='Explicit historical continuation judgment'
    _,proof=m.reference('mellum',m.binding(source),owner_reason=reason)
    original=proof.pop('source')
    proof.update(original_source_path=original['path'],original_source_sha256=original['sha256'],original_source_bytes=source.stat().st_size)
    if fault=='summary':proof['observed_successes']=0
    snapshot=tmp_path/'evidence.json';write(snapshot,{'schema':'natlang.native-task-reference-evidence/1','roles':{'mellum':proof}})
    local=m.binding(snapshot);remote={'path':'/work/native-production/data/reference/evidence.json','sha256':local['sha256'],'bytes':snapshot.stat().st_size}
    if fault=='hash':remote['sha256']='0'*64
    if fault=='bytes':remote['bytes']+=1
    if fault:
        with pytest.raises(ValueError):m.prepare(m.binding(recipe),'mellum',m.binding(source),tmp_path/'out',owner_reason=reason,portable_evidence={'local':local,'remote':remote})
    else:
        result=m.prepare(m.binding(recipe),'mellum',m.binding(source),tmp_path/'out',owner_reason=reason,portable_evidence={'local':local,'remote':remote})
        provenance=result['reference_provenance'];assert provenance['source']==remote
        assert provenance['original_source_path']==str(source)
        assert not any(isinstance(value,dict)and'path'in value for key,value in provenance.items()if key!='source')
