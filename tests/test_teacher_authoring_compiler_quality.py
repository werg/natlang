import copy
import json
import pytest
from scripts.neuralese_data import teacher_file_quality as q


def captured():
    runs=[{'args':[3],'expected':6},{'args':[8],'expected':16}]
    marker='This `nl` expression was not compiled by the natlang compiler.'
    return {'id':'teacher:compiler','task':{'program_ir':{'split':'train','source_groups':['source:train'],
        'semantics':{'expected':None,'authoring':{'module':'app.ts','export':'double','runs':runs}}}},
        'outcome':{'status':'done','accepted':False,'files':{'app.ts':'export function double(value:number):number { return value*2; }'},
            'oracle':{'accepted':False,'level':'exact'},'rejection_reasons':['files'],
            'checks':{k:k!='files'for k in ('seeded_failure_requirement','expected_status','honest_stop','effects','files','world','file_return_consistency','answer')},
            'authoring':{'ok':False,'results':[None,None],'problems':[f'run {i}: authoring_check: quiesced: error: The authored export failed: '+marker for i in range(2)]}},
        'trajectory':[{'raw_model_text':'preserved authored output'}],'provenance':{'actor':'old-base-teacher'}}


@pytest.fixture
def legacy(monkeypatch):
    # Unit fixture isolates historical custody; actual43e replay is tested by the operational capture.
    monkeypatch.setattr(q,'historical_authoring_verdict',lambda row,ref:copy.deepcopy(row['outcome']['authoring']))
    return {'path':'fixture','inventory_sha256':q.HISTORICAL_AUTHORING_RUNTIME}


def test_actual_current_all_runs_and_canonical_original_reconstruction(tmp_path,legacy):
    from scripts.neuralese_data.build_verified_teacher_trajectory_graphs import verify_original_row
    row=captured();original=copy.deepcopy(row);p=tmp_path/'raw.jsonl';p.write_text(json.dumps(row)+'\n');ref={'path':str(p),'sha256':q.sha256_file_hex(p)}
    view=q.continue_authoring_compiler_quality(row,ref,legacy)
    assert row==original and view['task']==row['task'] and view['trajectory']==row['trajectory']
    assert view['outcome']['authoring']=={'ok':True,'problems':[],'results':[6,16]}
    assert verify_original_row(view,ref)==p
    view['outcome']['value']='changed output'
    with pytest.raises(ValueError,match='differs'):verify_original_row(view,ref)


@pytest.mark.parametrize('fault',['held','groups','root','other_check','oracle','pending','rejection','captured_ok','captured_results','missing_problem','other_problem','empty_runs','malformed_run','wrong_result','wrong_files'])
def test_no_compiler_waiver(fault,legacy):
    row=captured();p=row['task']['program_ir'];o=row['outcome'];s=p['semantics']
    if fault=='held':p['split']='test'
    elif fault=='groups':p['source_groups']=[]
    elif fault=='root':o['status']='error'
    elif fault=='other_check':o['checks']['world']=False
    elif fault=='oracle':o['oracle']['level']='partial'
    elif fault=='pending':o['quality_pending']=['held']
    elif fault=='rejection':o['rejection_reasons']=['files','other']
    elif fault=='captured_ok':o['authoring']['ok']=True
    elif fault=='captured_results':o['authoring']['results'][0]=6
    elif fault=='missing_problem':o['authoring']['problems'].pop()
    elif fault=='other_problem':o['authoring']['problems'][1]='wrong answer'
    elif fault=='empty_runs':s['authoring']['runs']=[]
    elif fault=='malformed_run':s['authoring']['runs'][0].pop('expected')
    elif fault=='wrong_result':s['authoring']['runs'][1]['expected']=17
    else:o['files']['app.ts']=o['files']['app.ts'].replace('value*2','value*3')
    with pytest.raises(ValueError):q.continue_authoring_compiler_quality(row,{'path':'raw','sha256':'a'*64},legacy)


def test_historical_reproduction_is_mandatory(monkeypatch,legacy):
    monkeypatch.setattr(q,'historical_authoring_verdict',lambda row,ref:{'ok':False,'problems':['other'],'results':[None,None]})
    with pytest.raises(ValueError,match='historical compiler verdict'):q.continue_authoring_compiler_quality(captured(),{},legacy)


def test_arbitrary_runtime_is_not_historical_authority():
    with pytest.raises(ValueError,match='captured compiler epoch'):q.historical_authoring_verdict(captured(),{'path':'arbitrary','inventory_sha256':'a'*64})


def test_exact_optional_nl_presence_false_negative_only(legacy):
    row=captured();row['task']['program_ir']['semantics']['authoring']['requires']={'nl':True}
    row['outcome']['files']['app.ts']='export async function double(value:number):Promise<number> { const f=nl.with<number>({})`Double value.`; return await f(value); }'
    row['task']['program_ir']['semantics']['authoring']['oracle']=[{'match':['3'],'value':6},{'match':['8'],'value':16}]
    row['outcome']['authoring']['problems'].insert(0,'app.ts creates no nl function')
    assert q.continue_authoring_compiler_quality(row,{},legacy)['outcome']['accepted']is True
    row['outcome']['authoring']['problems'][0]='other requirement failed'
    with pytest.raises(ValueError):q.continue_authoring_compiler_quality(row,{},legacy)
