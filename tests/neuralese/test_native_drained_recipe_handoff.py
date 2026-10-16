"""A real drained owner crosses a verified priority handoff without losing history."""
import copy
import json
from pathlib import Path
import shutil
import time
import pytest
import torch
from test_native_raw_handoff import actual_drain
from test_native_resident import setup
from natlang_neuralese.rl import native_raw_handoff as handoff, native_raw_rounds as raw
from natlang_neuralese.train import native_resident as resident
from natlang_neuralese.common.jsonio import write_json_durable as write


@pytest.mark.parametrize('interrupt',[False,True])
def test_drained_resident_selects_new_priority_plan_and_preserves_exact_parent(actual_drain,tmp_path,monkeypatch,interrupt):
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE',str(tmp_path/'cache'))
    root=tmp_path/'source';root.mkdir()
    paths,directory,manager,new_recipe=setup(root,manifest_train=True)
    manifest=root/'adopted.json';write(manifest,{'schema':'natlang.native-graph-sources/1',
        'sources':[{'kind':'native-graphs',**resident.binding(paths['new'])}]})
    recipe=json.loads(new_recipe.read_text());recipe['input_bindings']['train']=resident.binding(manifest);write(new_recipe,recipe)
    resident.prepare(directory,new_recipe,'sft',tokenizer_fingerprint='tokenizer-sha',role_ids=())
    assert manager.poll(7) is not None
    state=resident.current(directory)
    (directory.parent/'sft').mkdir()  # The actual drained supervised stage is started.
    frozen=directory.parent/'runtime/natlang_neuralese'
    package=Path(resident.__file__).parents[1]
    shutil.copytree(package,frozen,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    process,intent,terminal,slot,pin=actual_drain
    tree=json.loads(Path(intent['path']).read_text());initial=json.loads((directory/'initial.json').read_text())
    initial['owner']=tree['children'][0];write(directory/'initial.json',initial)
    saved=torch.load(slot,weights_only=False)
    saved['training'].update(global_step=7,source_receipts=[state['payload']['train']],
        options={'resident_adoption':{'generation':state['generation'],'history':state['history']}})
    torch.save(saved,slot);shutil.copyfile(slot,pin)
    process.communicate();write(terminal,{'at':time.time(),'returncode':0})
    original=handoff.complete_drain(intent,resident.binding(terminal),resident.binding(pin),tmp_path/'drained.json',
        original_slot=resident.binding(slot),resident_directory=directory)
    oldbytes=Path(json.loads(Path(original['path']).read_text())['resident']['current']['path']).read_bytes()
    parentbytes=pin.read_bytes()
    resident.retire(directory,resident.binding(pin),resident.binding(terminal))
    priority=root/'priority.json';value=json.loads(manifest.read_text());value['sources'][0]['training_priority']=1;write(priority,value)
    next_recipe=root/'next-recipe.json';recipe=copy.deepcopy(recipe)
    recipe['input_bindings']['train']=resident.binding(priority)
    recipe['input_bindings']['prior']=resident.binding(manifest)
    write(next_recipe,recipe)
    if interrupt:
        original_write=handoff.write_json_durable
        def fail_receipt(path,value):
            if path==tmp_path/'transition.json':raise OSError('receipt interruption after plan commit')
            original_write(path,value)
        with monkeypatch.context() as patch:
            patch.setattr(handoff,'write_json_durable',fail_receipt)
            with pytest.raises(OSError,match='receipt interruption'):
                handoff.prepare_drained_recipe_handoff(original,next_recipe,tmp_path/'transition.json',package=package)
    proof=handoff.prepare_drained_recipe_handoff(original,next_recipe,tmp_path/'transition.json',package=package)
    checked=handoff.verify_drained_recipe_handoff(proof,handoff=original)
    newplan=json.loads(Path(checked['new_plan']['path']).read_text())
    changes=newplan['handoffs'][-1]['stage_input_changes']
    assert changes['sft']['kind']=='scheduling_priority_continuation'
    assert resident.current(directory) is None and pin.read_bytes()==parentbytes
    assert Path(json.loads(Path(original['path']).read_text())['resident']['current']['path']).read_bytes()==oldbytes
    assert handoff.prepare_drained_recipe_handoff(original,next_recipe,tmp_path/'transition.json',package=package)==proof
    runtime=tmp_path/'runtime';(runtime/'dist/teacher').mkdir(parents=True);(runtime/'dist/teacher/cli.js').write_text('/* not executed */')
    cases=tmp_path/'cases';cases.write_text('{"id":"private","split":"train"}\n')
    declaration={'model':'lfm','production_stage':'sft','runtime':str(runtime),'base':str(tmp_path/'base'),'port':8097,
        'drained_recipe_handoff':proof,'handoff_requirement':{'model_id':'fixture:'+'a'*64,
        'recipe_sha256':manager.recipe_binding['sha256'],'stage_ids':['sft'],
        'trainer_launch':tree['source_launch'],'trainer_exit':str(terminal)},
        'cases':resident.binding(cases),'production_recipe':resident.binding(new_recipe),
        'seed_native_graphs':state['payload']['train'],'held_native_graphs':state['payload']['held']}
    from natlang_neuralese.rl import production
    calls=[]
    def configure(args):
        calls.append(args)
        return {'native_state':True,'checkpoint':str(args.checkpoint),'out':str(args.out),
            'production_recipe':str(args.production_recipe),'seed_trainer_parameters':{},'trainer_parameters':{}}
    monkeypatch.setattr(production,'configure_native',configure)
    plan=tmp_path/'bootstrap.json';write(plan,{'models':[declaration]})
    raw.configure_model(plan,declaration,original,tmp_path/'raw')
    assert calls[0].production_recipe==next_recipe and calls[0].seed_native_graphs==priority
    assert raw._release_requirement(declaration,{})['recipe_sha256']==checked['recipe']['sha256']
    for field in ('checkpoint','new_code','new_plan'):
        forged=copy.deepcopy(checked)
        if field=='checkpoint':forged[field]=resident.binding(slot)
        elif field=='new_code':forged[field]={}
        else:
            bad=tmp_path/'bad-plan.json';other=copy.deepcopy(newplan);other['stage_inputs']['sft']['native_graphs']['sha256']='f'*64;write(bad,other)
            forged[field]=resident.binding(bad)
        path=tmp_path/('forged-'+field+'.json');write(path,forged)
        with pytest.raises(ValueError):handoff.verify_drained_recipe_handoff(resident.binding(path),handoff=original)
