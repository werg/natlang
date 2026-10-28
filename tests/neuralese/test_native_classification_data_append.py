"""New data and its proof append atomically without replaying historic indexes."""
import copy
import json
import shutil
from pathlib import Path

import pytest

from natlang_neuralese.common import row_index
from natlang_neuralese.train import native_resident as resident, recipe
from natlang_neuralese.train import native_source_classification as migration
from natlang_neuralese.data.native_sources import NativeGraphSource
from test_native_resident import setup


def prepared(tmp_path,monkeypatch,*,fingerprint='tokenizer-sha',role_ids=(),geometry_identity=None):
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE',str(tmp_path/'cache'))
    paths,directory,previous,new_recipe=setup(tmp_path,fingerprint,role_ids,geometry_identity,manifest_train=True)
    package=Path(migration.__file__).resolve().parents[1]
    old_package=tmp_path/'old-package'
    for name in migration.POLICY_FILES:
        target=old_package/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(package/name,target)
    with (old_package/'data/native_sources.py').open('a')as stream:stream.write('\n# historical policy\n')
    def part(source):
        witness=resident._source_witness(source)[0]
        current=row_index.load(witness['cache']['path'])
        key={**current['key'],'policy_sha256':migration.source_policy(old_package)}
        entries=[{k:v for k,v in entry.items()if k not in migration.CLASS_FIELDS}for entry in current['entries']]
        for entry in entries:entry['communication']=False
        prior=tmp_path/(source.path.name+'.prior.pkl')
        row_index.save(prior,{'key':key,'entries':entries,'metadata_sha256':resident.digest(entries)})
        return migration.migrate_index(source.path,prior,old_package=old_package,new_package=package)
    old_part=part(previous.train.children[0]);held_part=part(previous.held)
    extra=tmp_path/'extra.jsonl';extra.write_text(paths['new'].read_text().splitlines()[-1]+'\n')
    extra_source=NativeGraphSource(extra,fingerprint,required_split='train',role_ids=role_ids)
    extra_part=part(extra_source)
    combined=tmp_path/'combined.json';resident.write_json(combined,{'schema':'natlang.native-graph-sources/1',
        'sources':[{'kind':'native-graphs',**resident.binding(paths['old'])},{'kind':'native-graphs',**resident.binding(extra)}]})
    proof={'schema':migration.SCHEMA,'train_sha256':resident.sha(paths['old']),
        'held':migration._binding(paths['held']),'indexes':[old_part],'held_indexes':[held_part],
        'model_allocations':0,'optimizer_updates':0,'student_pointer_writes':0,'graph_position_expansions':0}
    old_proof=tmp_path/'old-proof.json';resident.write_json(old_proof,proof)
    new_proof=tmp_path/'new-proof.json';resident.write_json(new_proof,{**proof,
        'train_sha256':resident.sha(combined),'indexes':[old_part,extra_part]})
    old_recipe=tmp_path/'old-recipe.json';value=json.loads(old_recipe.read_text())
    value['input_bindings']['classification']=resident.binding(old_proof)
    value['stages'][0]['inputs']['source_classification_migration']='classification'
    resident.write_json(old_recipe,value)
    plan=copy.deepcopy(previous.plan);plan['recipe']=recipe.load_recipe(old_recipe)
    plan['recipe_sha256']=resident.sha(old_recipe)
    plan['stage_inputs']['sft']=recipe.resolve_stage_inputs(plan['recipe'],plan['recipe']['stages'][0],{})
    plan['inputs'][str(old_proof)]=resident.sha(old_proof)
    resident.write_json(directory.parent/'recipe-plan.json',plan)
    shutil.rmtree(directory)  # discard only this test's first control fixture
    manager=resident.ResidentAdoption(directory,stage_id='sft',
        recipe_binding=recipe.native_recipe_binding(old_recipe,plan['recipe_sha256']),
        train=previous.train,held=previous.held,geometry_qualification=tmp_path/'geometry-capacity.json',geometry_identity=geometry_identity)
    proposed=copy.deepcopy(value)
    proposed['input_bindings'].update(train=resident.binding(combined),prior=resident.binding(paths['old']),
                                      classification=resident.binding(new_proof))
    proposed['stages'][0]['inputs']['continued_native_graphs']='prior'
    resident.write_json(new_recipe,proposed)
    return directory,manager,new_recipe,new_proof,extra


@pytest.mark.parametrize('delta',[False,True])
def test_actual_prepare_poll_and_recover_joint_classified_data(tmp_path,monkeypatch,delta):
    directory,manager,path,proof,extra=prepared(tmp_path,monkeypatch)
    old_verifier=migration.migration_classes;verified=[]
    def track(path,source):
        verified.append([entry['id'] for entry in source.entries])
        return old_verifier(path,source)
    monkeypatch.setattr(migration,'migration_classes',track)
    resident.prepare(directory,path,'sft',tokenizer_fingerprint='tokenizer-sha',role_ids=(),
        delta_sources={'train':resident.binding(extra)} if delta else None)
    train,held=manager.poll(19)
    assert len(train)==3 and len(held)==1
    assert verified and all(rows==['c'] for rows in verified)
    current=resident.recover(directory)
    assert current['at_global_step']==19 and 'exposure_counts' not in current
    assert current['payload']['plan']['stage_inputs']['sft']['source_classification_migration']['sha256']==resident.sha(proof)
    assert manager.plan==current['payload']['plan']
    assert not (directory/'rejections').exists()


@pytest.mark.parametrize('fault',['rewrite-prefix','rewrite-held','omit-new','foreign-new'])
def test_joint_proof_rejects_before_publishing_request(tmp_path,monkeypatch,fault):
    directory,manager,path,proof,extra=prepared(tmp_path,monkeypatch)
    value=json.loads(proof.read_text())
    if fault=='rewrite-prefix':value['indexes'][0]['rows']+=1
    elif fault=='rewrite-held':value['held_indexes']=[]
    elif fault=='omit-new':value['indexes']=value['indexes'][:1]
    else:value['indexes'].append(copy.deepcopy(value['held_indexes'][0]))
    resident.write_json(proof,value)
    configured=json.loads(path.read_text());configured['input_bindings']['classification']=resident.binding(proof)
    resident.write_json(path,configured)
    with pytest.raises(ValueError):
        resident.prepare(directory,path,'sft',tokenizer_fingerprint='tokenizer-sha',role_ids=())
    assert resident.current(directory)is None and not(directory/'pending.json').exists()


def test_identical_relocated_leaf_keeps_both_custody_bindings(tmp_path,monkeypatch):
    directory,manager,path,proof,extra=prepared(tmp_path,monkeypatch)
    alias=tmp_path/'verified-transfer-copy.jsonl';shutil.copyfile(extra,alias)
    value=json.loads(proof.read_text());value['indexes'][-1]['source']=migration._binding(alias)
    resident.write_json(proof,value)
    configured=json.loads(path.read_text());configured['input_bindings']['classification']=resident.binding(proof)
    resident.write_json(path,configured)
    resident.prepare(directory,path,'sft',tokenizer_fingerprint='tokenizer-sha',role_ids=())
    train,held=manager.poll(7)
    receipts=list((directory/'classification-append-proofs').glob('*.json'))
    assert receipts
    for receipt in receipts:
        scope=json.loads(receipt.read_text())['resident_append_scope']
        assert scope['actual_source']['path']==str(extra) and scope['proof_source']['path']==str(alias)
        assert scope['actual_source']['sha256']==scope['proof_source']['sha256']
    assert len(train)==3


@pytest.mark.parametrize('crash_after_commit',[False,True])
def test_actual_sgd_and_old_checkpoint_resume_select_committed_proof(tmp_path,monkeypatch,crash_after_commit):
    import torch
    from test_native_model import tiny_source
    from natlang_neuralese.model.native_model import load_native_model
    from natlang_neuralese.train import native_state_sft as trainer
    base=tmp_path/'base';tiny_source(base)
    _,_,roles,config=load_native_model(base,dtype=torch.float32,rank=4)
    identity={'model_config':config,'batch_graphs':1,'batch_tokens':0,'checkpoint_layers':True,
        'chunk_tokens':1024,'checkpoint_input_offload':False,'cotangent_accumulator_backend':'plain',
        'share_causal_prefixes':True,
        'fp8':{'kind':'off','recipe':'delayed','amax_history_len':16,'input_gradient_precision':None},
        'runtime':{'torch':torch.__version__,'cuda':torch.version.cuda,'device':'cpu'}}
    directory,manager,path,proof,extra=prepared(tmp_path,monkeypatch,
        fingerprint=config['tokenizer_fingerprint'],role_ids=roles.ids,geometry_identity=identity)
    old=manager.plan['stage_inputs']['sft'];out=directory.parent/'sft'
    argv=['--base',str(base),'--dtype','float32','--rank','4','--device','cpu',
        '--native-graphs',old['native_graphs']['path'],'--held-native-graphs',old['held_native_graphs']['path'],
        '--source-classification-migration',old['source_classification_migration']['path'],
        '--epochs','1','--eval-every','1','--checkpoint-every','1','--out',str(out),
        '--resident-data-directory',str(directory),'--resident-geometry-qualification',str(tmp_path/'geometry-capacity.json'),
        '--recipe-binding',json.dumps(manager.recipe_binding),'--stage-id','sft']
    due=trainer.CheckpointPolicy.due;poll=resident.ResidentAdoption.poll;published=[]
    def prepare_on_update(policy,step,**kwargs):
        if step==1 and not published:
            published.append(resident.prepare(directory,path,'sft',tokenizer_fingerprint=config['tokenizer_fingerprint'],role_ids=roles.ids))
        return due(policy,step,**kwargs)
    class Crash(BaseException):pass
    def crash(self,step):
        result=poll(self,step)
        if result is not None:raise Crash()
        return result
    with monkeypatch.context()as patch:
        patch.setattr(trainer.CheckpointPolicy,'due',prepare_on_update)
        if crash_after_commit:
            patch.setattr(resident.ResidentAdoption,'poll',crash)
            with pytest.raises(Crash):trainer.main(argv)
        else:assert trainer.main(argv)==0
    if crash_after_commit:
        saved=torch.load(out/'checkpoint.pt',weights_only=False)
        assert saved['training']['global_step']==1
        assert saved['training']['options']['source_classification_migration']==old['source_classification_migration']['path']
        assert trainer.main(argv)==0  # deliberately unchanged stale launch arguments
    saved=torch.load(out/'checkpoint.pt',weights_only=False);training=saved['training']
    assert training['global_step']==3 and training['exposure_counts']=={'a':1,'b':1,'c':1}
    assert training['options']['source_classification_migration']==str(proof)
    assert training['options']['source_classification_history'][-1]=={
        'proof_sha256':resident.sha(proof),'graph_visits_created':0}
