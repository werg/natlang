import copy
import json
import pytest
from scripts.neuralese_data import teacher_file_quality as quality


def captured():
    files={'app.ts':'export async function double(value: number): Promise<number> { return value * 2; }'}
    spec={'module':'app.ts','export':'double','runs':[{'args':[3],'expected':6},{'args':[8],'expected':16}]}
    return {'id':'teacher:authoring','task':{'program_ir':{'id':'program:authoring','split':'train',
        'source_groups':['authoring:train'], 'semantics':{'expected':None,'authoring':spec,
        'folder_files':{'app.ts':'export async function double(value: number): Promise<number> { return 0; }'}}}},
        'outcome':{'status':'done','accepted':True,'value':'done','files':files,
            'authoring':{'ok':True,'problems':[],'results':[6,16]},
            'oracle':{'accepted':False,'level':'exact'},'rejection_reasons':[],
            'checks':{key:True for key in ('seeded_failure_requirement','expected_status','honest_stop','effects','files','world','file_return_consistency','answer')}},
        'trajectory':[{'raw_model_text':'original authored code'}], 'provenance':{'actor':'unchanged'}}


def test_actual_authoring_rerun_and_original_converter_reconstruction(tmp_path):
    from scripts.neuralese_data.build_verified_teacher_trajectory_graphs import verify_original_row
    row=captured();original=copy.deepcopy(row);p=tmp_path/'raw.jsonl';p.write_text(json.dumps(row)+'\n')
    binding={'path':str(p),'sha256':quality.sha256_file_hex(p)}
    view=quality.continue_authoring_quality(row,binding)
    assert row==original
    assert view['task']==row['task'] and view['trajectory']==row['trajectory'] and view['provenance']==row['provenance']
    changed=copy.deepcopy(view);changed.pop('source_quality_continuation');changed['outcome']['oracle']=row['outcome']['oracle']
    assert changed==row
    assert verify_original_row(view,binding)==p
    view['outcome']['value']='silently changed root reply'
    with pytest.raises(ValueError,match='differs'):verify_original_row(view,binding)


@pytest.mark.parametrize('fault',['wrong_files','wrong_expected','false_check','pending','rejection','held','empty_runs','claimed_results','oracle_other','accepted_false'])
def test_no_authoring_waiver_for_wrong_execution_or_other_debt(fault):
    row=captured();o=row['outcome'];s=row['task']['program_ir']['semantics']
    if fault=='wrong_files':o['files']['app.ts']=o['files']['app.ts'].replace('value * 2','value * 3')
    elif fault=='wrong_expected':s['authoring']['runs'][0]['expected']=9
    elif fault=='false_check':o['checks']['world']=False
    elif fault=='pending':o['quality_pending']=['held']
    elif fault=='rejection':o['rejection_reasons']=['other']
    elif fault=='held':row['task']['program_ir']['split']='test'
    elif fault=='empty_runs':s['authoring']['runs']=[]
    elif fault=='claimed_results':o['authoring']['results']=[9,16]
    elif fault=='oracle_other':o['oracle']['level']='normalized'
    else:o['accepted']=False
    with pytest.raises(ValueError):quality.continue_authoring_quality(row,{'path':'original','sha256':'a'*64})


def test_historical_verifier_is_provenance_but_current_execution_required(monkeypatch):
    row=captured();ref={'path':'original','sha256':'a'*64};view=quality.continue_authoring_quality(row,ref)
    real=quality.authoring_verdict
    def rebuilt(row):
        verdict,_=real(row);return verdict,{'files':1,'inventory_sha256':'b'*64}
    monkeypatch.setattr(quality,'authoring_verdict',rebuilt)
    quality.verify_quality_continuation(view,row,ref)
    def rejects(row):
        verdict,identity=rebuilt(row);return {**verdict,'ok':False},identity
    monkeypatch.setattr(quality,'authoring_verdict',rejects)
    with pytest.raises(ValueError,match='fresh executable'):quality.verify_quality_continuation(view,row,ref)
