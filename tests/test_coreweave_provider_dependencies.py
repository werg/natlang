import importlib.util
import json
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('provider_dependencies',Path(__file__).parents[1]/'scripts/coreweave/provider_dependencies.py')
deps=importlib.util.module_from_spec(spec);spec.loader.exec_module(deps)

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value) if isinstance(value,dict) else value)
    return {'path':str(path),'sha256':deps.sha256_file_hex_cached(path),'bytes':path.stat().st_size}

def test_mandatory_sft_seed_closure_contains_nested_ordinary_and_raw_pairs(tmp_path):
    ordinary=write(tmp_path/'ordinary.jsonl','{"id":"one"}\n')
    pair=write(tmp_path/'pairs.jsonl','{"id":"paired"}\n')
    nested=write(tmp_path/'nested.json',{'schema':'natlang.native-graph-sources/1','sources':[{'kind':'native-graphs',**ordinary}]})
    seed=write(tmp_path/'seed.json',{'schema':'natlang.native-graph-sources/1','sources':[{'kind':'native-graphs',**nested},{'kind':'semantic-pairs',**pair,'model':'exact-port-id'}]})
    held=write(tmp_path/'held.jsonl','{"id":"held"}\n')
    config=write(tmp_path/'config.json',{'mandatory_sft_only':True,'seed_native_graphs':seed['path'],'held_native_graphs':held['path']})
    closure=deps.dependency_closure({'config':config},namespace=str(tmp_path))
    assert {row['path'] for row in closure['files']}=={row['path'] for row in [config,seed,nested,ordinary,pair,held]}
    Path(pair['path']).write_text('changed source')
    with pytest.raises(ValueError,match='hash mismatch'):
        deps.dependency_closure({'config':config},namespace=str(tmp_path))

def test_asset_preflight_verifies_content_not_directory_existence(tmp_path):
    task=tmp_path/'ts-host';write(task/'dist/teacher/cli.js','actual compiled runtime')
    config=write(tmp_path/'config.json',{'runtime':str(task)})
    closure=deps.dependency_closure({'config':config},namespace=str(tmp_path))
    deps.verify_assets(closure['assets'])
    (task/'dist/teacher/cli.js').write_text('other runtime')
    with pytest.raises(ValueError,match='asset content differs'):
        deps.verify_assets(closure['assets'])

def test_config_cycle_and_external_payloads_fail_before_launch(tmp_path):
    path=tmp_path/'config.json';config=write(path,{'semantic_refresh_config':str(path)})
    with pytest.raises(ValueError,match='cycle'):
        deps.dependency_closure({'config':config},namespace=str(tmp_path))
    config=write(path,{'seed_native_graphs':str(tmp_path.parent/'external.jsonl')})
    with pytest.raises(ValueError,match='escaped'):
        deps.dependency_closure({'config':config},namespace=str(tmp_path))

def test_composite_optimizer_and_rng_sidecars_are_not_lost(tmp_path):
    base=write(tmp_path/'parent.pt','immutable-binary-parent')
    parameter=write(tmp_path/'parameters.pt','parameter-update')
    optimizer=write(tmp_path/'optimizer.pt','optimizer-moments')
    rng=write(tmp_path/'training.pt','rng-and-mandatory-sft-progress')
    composite=write(tmp_path/'checkpoint.json',{
        'schema':'natlang.neuralese-native-composite-checkpoint/1',
        'base_checkpoint':base,
        'parameter_artifact':{**parameter,'path':'parameters.pt'},
        'optimizer_state_artifact':{**optimizer,'path':'optimizer.pt'},
        'training_state':{**rng,'path':'training.pt'}})
    closure=deps.dependency_closure({'parent_checkpoint':composite},namespace=str(tmp_path))
    assert {row['path'] for row in closure['files']}=={row['path'] for row in [base,parameter,optimizer,rng,composite]}
    Path(rng['path']).unlink()
    with pytest.raises((ValueError,FileNotFoundError)):
        deps.dependency_closure({'parent_checkpoint':composite},namespace=str(tmp_path))

def test_prior_qualified_semantic_sources_and_block_bundles_survive_refresh(tmp_path):
    checkpoint=write(tmp_path/'parent.pt','immutable-semantic-parent')
    token=write(tmp_path/'tokenizer/tokenizer.json','tokenizer-bytes')
    pairs=write(tmp_path/'old-pairs.jsonl','old-qualified-pair')
    held=write(tmp_path/'old-held-pairs.jsonl','old-qualified-held-pair')
    block=write(tmp_path/'native-block.safetensors','encoded-native-state')
    bundle=write(tmp_path/'blocks.json',{'schema':'natlang.rl-block-bundle/1','blocks':{'actual-block-id':{**block,'path':'native-block.safetensors'}}})
    config={'pairs':pairs,'held_pairs':held,'external_bundles':{'actual-model-id':bundle}}
    value={'schema':'natlang.native-semantic-adapter/1','checkpoint':checkpoint,
        'models':{'actual-model-id':{'native_identity':{'checkpoint_sha256':checkpoint['sha256']}}},
        'tokenizers':{'actual-model-id':{'path':'tokenizer','files':{'tokenizer.json':token['sha256']}}},
        'qualified_curriculum':{'config':config,'sha256':deps.canonical_json_sha256_hex(config)}}
    adapter=write(tmp_path/'adapter.json',value)
    closure=deps.dependency_closure({'semantic_adapter':adapter},namespace=str(tmp_path))
    assert {row['path'] for row in closure['files']}=={row['path'] for row in [checkpoint,token,pairs,held,block,bundle,adapter]}
    value['qualified_curriculum']['config']['pairs']=held
    adapter=write(tmp_path/'adapter.json',value)
    with pytest.raises(ValueError,match='qualified semantic curriculum hash'):
        deps.dependency_closure({'semantic_adapter':adapter},namespace=str(tmp_path))

def test_mmap_checkpoint_declares_exact_public_base_preflight(tmp_path):
    import torch
    base=tmp_path/'public-base'
    write(base/'config.json',{'model_type':'lfm2'})
    write(base/'model.safetensors','pinned-model-source')
    from natlang_neuralese.model.native_model import base_inventory
    expected=base_inventory(base)
    checkpoint=tmp_path/'native.pt'
    torch.save({'model_config':{'base':str(base),'base_inventory':expected},'backbone_state':{'x':torch.ones(2)}},checkpoint)
    binding={'path':str(checkpoint),'sha256':deps.sha256_file_hex_cached(checkpoint),'bytes':checkpoint.stat().st_size}
    closure=deps.dependency_closure({'parent_checkpoint':binding},namespace=str(tmp_path))
    assert len(closure['assets'])==1
    deps.verify_assets(closure['assets'])
    (base/'model.safetensors').write_text('different source')
    with pytest.raises(ValueError,match='pinned base inventory changed'):
        deps.dependency_closure({'parent_checkpoint':binding},namespace=str(tmp_path))
