import copy
from types import SimpleNamespace
import pytest
from scripts.coreweave.checkpoint_exposure_summary import summarize_cohorts


def test_exact_readonly_cohorts_distinguish_missing_admission_and_zero_visits():
    frame={'options':{'stage_id':'ordinary'},'step':8,'source_row_bindings':{'a':'aa','b':'bb'},'exposure_counts':{'a':2},'required_visits':{'a':2,'b':1}}
    state={'step':12,'training':frame};before=copy.deepcopy(state)
    spec={'stage_id':'ordinary','cohorts':{'old':{'path':'old','sha256':'1'},'new':{'path':'new','sha256':'2'}}}
    rows={'old':[{'id':'a','binding':'aa'},{'id':'b','binding':'bb'}],'new':[{'id':'c','binding':'cc'}]}
    result=summarize_cohorts(state,spec,lambda ref:SimpleNamespace(entries=rows[ref['path']]))
    assert result['cohorts']['old']['actual_visits']==2
    assert result['cohorts']['old']['completed_rows']==1
    assert result['cohorts']['old']['never_visited_admitted_rows']==1
    assert result['cohorts']['new']['not_admitted_rows']==1
    assert result['cohorts']['new']['never_visited_admitted_rows']==0
    assert state==before


def test_carried_ordinary_history_is_selected_without_current_dpo_credits():
    ordinary={'step':3,'source_row_bindings':{'a':'aa'},'exposure_counts':{'a':1},'required_visits':{'a':1}}
    state={'step':8,'training':{'options':{'stage_id':'dpo'},'exposure_counts':{'a':999},'sft_progress':{'ordinary':ordinary}}}
    result=summarize_cohorts(state,{'stage_id':'ordinary','cohorts':{'x':{}}},lambda _:SimpleNamespace(entries=[{'id':'a','binding':'aa'}]))
    assert result['stage_step']==3 and result['cohorts']['x']['actual_visits']==1


@pytest.mark.parametrize('damage',['binding','credit','overlap'])
def test_conflicting_or_duplicated_row_custody_is_rejected(damage):
    state={'step':1,'training':{'options':{'stage_id':'ordinary'},'source_row_bindings':{'a':'aa'},'exposure_counts':{'a':1}}}
    rows=[{'id':'a','binding':'bb'if damage=='binding'else'aa'}]
    if damage=='credit':state['training']['exposure_counts']['a']=-1
    cohorts={'x':{}}
    if damage=='overlap':cohorts['y']={}
    with pytest.raises(ValueError):summarize_cohorts(state,{'stage_id':'ordinary','cohorts':cohorts},lambda _:SimpleNamespace(entries=rows))


def test_cursor_captures_summary_from_one_existing_safe_decode_and_suppresses_loader_stdout(tmp_path):
    import hashlib,json,subprocess,sys,torch
    from scripts.coreweave.cursor_checkpoint_export import remote_code
    runtime=tmp_path/'runtime';package=runtime/'natlang_neuralese'
    for sub in ['', 'data','common','train']:
        p=package/sub;p.mkdir(parents=True,exist_ok=True);(p/'__init__.py').write_text('')
    (package/'data/native_sources.py').write_text("class NativeGraphSource:\n def __init__(self,*args,**kwargs):\n  print('existing source verifier cache diagnostic')\n  self.entries=[{'id':'a','binding':'aa'}]\n")
    (package/'common/hashing.py').write_text("import hashlib\ndef sha256_file_hex_cached(path):return hashlib.sha256(path.read_bytes()).hexdigest()\n")
    (package/'train/native_resident.py').write_text('def _check_indexes(witnesses):\n assert witnesses==[\"bound-test-index\"]\n')
    cohort=tmp_path/'cohort.json';cohort.write_text('immutable source')
    specification={'stage_id':'ordinary','cohorts':{'new':{'path':str(cohort),'sha256':hashlib.sha256(cohort.read_bytes()).hexdigest()}},'tokenizer_fingerprint':'test','role_ids':[1,2,3,4],'row_index_cache':str(tmp_path/'cache'),'hash_cache':str(tmp_path/'hashes'),'indexes':['bound-test-index']}
    sp=tmp_path/'spec.json';sp.write_text(json.dumps(specification));ref={'path':str(sp),'sha256':hashlib.sha256(sp.read_bytes()).hexdigest()}
    root=tmp_path/'root';slot=root/'rounds/lfm/production/round-1/learner/checkpoint.pt';slot.parent.mkdir(parents=True);pins=tmp_path/'pins';pins.mkdir()
    state={'step':42,'training':{'options':{'stage_id':'ordinary'},'optimizer':{'moment':torch.tensor([1.])},'rng':{'cpu':torch.get_rng_state()},'source_receipts':[{}],'source_row_bindings':{'a':'aa'},'exposure_counts':{'a':1}}};torch.save(state,slot)
    code=remote_code(role='lfm',root=root,known={},last=0,extra=None,pin_directory=pins,minimum_available_bytes=0,training_summary_spec=ref,training_summary_runtime=runtime)
    # The same original mmap decode serves both full-state custody and reporting.
    assert code.count('torch.load(')==1
    completed=subprocess.run([sys.executable,'-c',code],text=True,capture_output=True,timeout=30)
    assert completed.returncode==0,completed.stderr
    output=json.loads(completed.stdout);summary=output['chosen']['training_summary']
    assert summary['global_step']==42 and summary['cohorts']['new']['completed_rows']==1
    assert 'cache diagnostic'in completed.stderr and 'cache diagnostic'not in completed.stdout
    # A rejected optional report cannot prevent the genuine full checkpoint pin.
    sp.write_text('changed spec');other=pins/'lfm-native-checkpoint-step-42.pt';other.unlink()
    completed=subprocess.run([sys.executable,'-c',code],text=True,capture_output=True,timeout=30)
    assert completed.returncode==0,completed.stderr
    assert json.loads(completed.stdout)['chosen']['training_summary']['assigned_repair']is True
    assert other.is_file()
