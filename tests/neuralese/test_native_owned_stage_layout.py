"""A captured real owner under learner retains active-stage contracts without markers."""
import copy,json
from pathlib import Path
import pytest
from test_native_external_resident_handoff import external,ordinary,bare,later,owned,_rejoin,_assert_next_update
from natlang_neuralese.rl import native_raw_handoff as handoff
from natlang_neuralese.train import native_resident as resident
from natlang_neuralese.train.recipe import unstarted_stage_input_changes,lineage_handoff
from natlang_neuralese.common.jsonio import write_json_durable as write

@pytest.mark.parametrize('external',['owner-layout'],indirect=True)
@pytest.mark.parametrize('later',['lfm','mellum'],indirect=True)
def test_actual_external_learner_layout_rejoins_and_keeps_next_update(external):
    x=external;assert not(x['external_root']/'sft').exists()
    proof=handoff.verify_drained_recipe_handoff(x['transition'],handoff=x['handoff'])
    layout=proof['external_runtime']['started_stage_layout'];assert layout['started_stage_ids']==['sft']
    plan=json.loads(Path(proof['new_plan']['path']).read_text())
    assert plan['handoffs'][-1]['authenticated_started_stages']==['sft']
    _rejoin(x);_assert_next_update(x)
    assert not(x['external_root']/'sft').exists()

@pytest.mark.parametrize('external',['owner-layout'],indirect=True)
@pytest.mark.parametrize('tamper',['out','stage-id','receipt-stage','checkpoint','exposure','layout-stage'])
def test_real_external_layout_rejects_changed_control_or_exposure(external,tamper,tmp_path):
    x=external;proof=json.loads(Path(x['transition']['path']).read_text());layout=proof['external_runtime']['started_stage_layout']
    if tamper in('out','stage-id'):
        control=proof['external_runtime']['launch_control'];argv=json.loads(Path(control['argv']['path']).read_text())
        argv[argv.index('--'+tamper)+1]=str(tmp_path/'foreign')if tamper=='out'else'not-sft'
        changed=tmp_path/'changed-argv.json';write(changed,argv);control['argv']=resident.binding(changed)
    elif tamper=='layout-stage':layout['stage_id']='not-sft'
    else:
        receipt=json.loads(Path(layout['receipt']['path']).read_text())
        if tamper=='receipt-stage':receipt['stage_id']='not-sft'
        elif tamper=='checkpoint':receipt['checkpoint']['sha256']='0'*64
        else:receipt['mandatory_sft']['stages']['sft']['graph_visits']+=1
        changed=tmp_path/'changed-receipt.json';write(changed,receipt);layout['receipt']=resident.binding(changed)
    path=tmp_path/'changed-proof.json';write(path,proof)
    with pytest.raises(ValueError):handoff.verify_drained_recipe_handoff(resident.binding(path),handoff=x['handoff'])

@pytest.mark.parametrize('external',['owner-layout'],indirect=True)
def test_active_source_stays_active_under_authenticated_external_layout(external):
    x=external;proof=json.loads(Path(x['transition']['path']).read_text());old=json.loads(Path(proof['old_plan']['path']).read_text());new=copy.deepcopy(old)
    new['stage_inputs']['sft']['native_graphs']['sha256']='0'*64
    assert unstarted_stage_input_changes(old,new,x['external_root'])['sft']['kind']=='unstarted_stage'
    with pytest.raises(ValueError):
        unstarted_stage_input_changes(old,new,x['external_root'],started_stage_ids=('sft',))
    with pytest.raises(ValueError,match='authenticated started stages'):
        lineage_handoff(old,old,x['newpackage'],x['newpackage'],x['external_root']/'unused.json',dry_run=True,started_stage_ids=('foreign',))


@pytest.mark.parametrize('external',['owner-layout-null-script','owner-layout-wrong-script'],indirect=True)
@pytest.mark.parametrize('later',['lfm'],indirect=True)
def test_present_invalid_supervisor_script_binding_is_not_omission(external):
    reference,recipe,out,package=external['rejected_layout_request']
    with pytest.raises(ValueError,match='exact captured supervisor'):
        handoff.prepare_drained_recipe_handoff(reference,recipe,out,package=package,external_package=package)
