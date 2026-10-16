"""No persistent tensor cache: exact unchanged bootstrap avoids model unpickling."""
import json
from pathlib import Path
import pytest
from test_native_raw_handoff import actual_drain,drained
from natlang_neuralese.rl import native_raw_rounds as raw,production
from natlang_neuralese.train.checkpoint_policy import CheckpointPolicy
from natlang_neuralese.common.jsonio import write_json_durable as write


def test_unchanged_config_rechecks_sources_and_ownership_without_deserializing_parent(actual_drain,tmp_path,monkeypatch):
    handoff=drained(actual_drain);record=json.loads(Path(handoff['path']).read_text())
    intent=json.loads(Path(record['intent']['path']).read_text())
    runtime=tmp_path/'ts-host';(runtime/'dist/teacher').mkdir(parents=True)
    for p,value in [('dist/teacher/cli.js','// runtime'),('prelude.js','// prelude'),('package.json','{}'),
        ('../package.json','{}'),('../package-lock.json','{}'),('../training/source-reviews/holds.jsonl',''),
        ('../training/source-reviews/family-holds.json','{}'),('../training/source-reviews/datasets.json','{}')]:
        path=runtime/p;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(value)
    refs={}
    for name in ('train','held','recipe','cases'):
        p=tmp_path/name;p.write_text(json.dumps({'id':name,'split':'train'})+'\n');refs[name]=raw.binding(p)
    declaration={'model':'fixture','production_stage':'sft','runtime':str(runtime),'base':str(tmp_path/'base'),'port':8097,
        'cases':refs['cases'],'production_recipe':refs['recipe'],'seed_native_graphs':refs['train'],'held_native_graphs':refs['held'],
        'handoff_requirement':{'model_id':'fixture:'+'a'*64,'recipe_sha256':refs['recipe']['sha256'],
            'trainer_launch':intent['source_launch'],'trainer_exit':record['source_terminal']['path']}}
    plan=tmp_path/'plan.json';write(plan,{'models':[declaration]})
    def configure(args):
        return {'out':str(args.out),'checkpoint':str(args.checkpoint),'production_recipe':str(args.production_recipe),
            'seed_native_graphs':str(args.seed_native_graphs),'held_native_graphs':str(args.held_native_graphs),
            'trainer_parameters':{},'seed_trainer_parameters':{}}
    monkeypatch.setattr(production,'configure_native',configure)
    actual=CheckpointPolicy.load;calls=[]
    def load(path,*args,**kwargs):calls.append(path);return actual(path,*args,**kwargs)
    monkeypatch.setattr(CheckpointPolicy,'load',load)
    out=tmp_path/'configured'
    path=raw.configure_model(plan,declaration,handoff,out)
    assert len(calls)==1 and (out/'bootstrap-verification.json').is_file()
    assert raw.configure_model(plan,declaration,handoff,out)==path
    assert len(calls)==1
    # Host configuration tamper invalidates reuse and re-runs the full verifier.
    value=json.loads(path.read_text());value['operational_note']='changed';write(path,value)
    raw.configure_model(plan,declaration,handoff,out)
    assert len(calls)==2
    seed=Path(refs['train']['path']);original=seed.read_bytes();seed.write_text('changed graph bytes')
    with pytest.raises(ValueError,match='changed'):raw.configure_model(plan,declaration,handoff,out)
    assert len(calls)==3  # Changed graph bytes trigger the complete original verifier.
    seed.write_bytes(original)
    raw.configure_model(plan,declaration,handoff,out)
    assert len(calls)==3
    Path(refs['cases']['path']).write_text('{"id":"changed","split":"test"}\n')
    with pytest.raises(ValueError,match='changed'):raw.configure_model(plan,declaration,handoff,out)
    assert len(calls)==3  # No tensor read is required to reject changed task bytes.


def test_hot_task_plan_keeps_latest_committed_resident_sources(actual_drain,tmp_path,monkeypatch):
    from test_native_resident import setup
    from natlang_neuralese.train import native_resident as resident
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE',str(tmp_path/'cache'))
    root=tmp_path/'source';root.mkdir()
    paths,resident_dir,manager,new_recipe=setup(root)
    old_recipe=Path(manager.recipe_binding['path'])
    resident.prepare(resident_dir,new_recipe,'sft',tokenizer_fingerprint='tokenizer-sha',role_ids=())
    assert manager.poll(7) is not None
    adopted=resident.current(resident_dir)
    snapshot=tmp_path/'immutable-resident';snapshot.mkdir()
    (snapshot/'current.json').write_bytes((resident_dir/'current.json').read_bytes())
    handoff=drained(actual_drain);record=json.loads(Path(handoff['path']).read_text())
    intent=json.loads(Path(record['intent']['path']).read_text())
    runtime=tmp_path/'runtime';(runtime/'dist/teacher').mkdir(parents=True)
    (runtime/'dist/teacher/cli.js').write_text('// test-only collector')
    cases=tmp_path/'cases';cases.write_text('{"id":"one","split":"train"}\n')
    declaration={'model':'fixture','production_stage':'sft','runtime':str(runtime),'base':str(tmp_path/'base'),'port':8097,
        'cases':raw.binding(cases),'production_recipe':raw.binding(old_recipe),
        'seed_native_graphs':raw.binding(paths['old']),'held_native_graphs':raw.binding(paths['held']),
        'handoff_requirement':{'model_id':'fixture:'+'a'*64,'recipe_sha256':manager.recipe_binding['sha256'],
            'stage_ids':['sft'],'trainer_launch':intent['source_launch'],'trainer_exit':record['source_terminal']['path']}}
    seen=[]
    def configure(args):
        seen.append(args)
        return {'out':str(args.out),'checkpoint':str(args.checkpoint),'production_recipe':str(args.production_recipe),
            'seed_native_graphs':str(args.seed_native_graphs),'held_native_graphs':str(args.held_native_graphs),
            'trainer_parameters':{},'seed_trainer_parameters':{}}
    monkeypatch.setattr(production,'configure_native',configure)
    plan=tmp_path/'plan.json';write(plan,{'models':[declaration]})
    out=tmp_path/'configured';config_path=raw.configure_model(plan,declaration,handoff,out)
    assert seen[-1].seed_native_graphs==paths['old']
    write(out/'cycle.json',{'phase':'rl','cycle':1,'resident_current':raw.binding(snapshot/'current.json')})
    second=tmp_path/'next-cases';second.write_text('{"id":"second","split":"train"}\n')
    declaration={**declaration,'cases':raw.binding(second)};write(plan,{'models':[declaration]})
    raw.configure_model(plan,declaration,handoff,out)
    updated=json.loads(config_path.read_text())
    assert updated['production_recipe']==adopted['payload']['recipe']['path']
    assert updated['seed_native_graphs']==adopted['payload']['train']['path']
    assert seen[-1].cases==second and seen[-1].seed_native_graphs==paths['new']
    assert updated['recipe_lineage_sha256']==manager.recipe_binding['lineage_sha256']
