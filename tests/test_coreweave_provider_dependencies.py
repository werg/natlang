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


def test_withdrawal_relative_evidence_stays_in_exact_provider_closure(tmp_path):
    evidence=write(tmp_path/'receipt.json',{'audit':'rejected targets'})
    proof=write(tmp_path/'model/withdrawal.json',{'schema':'natlang.native-graph-withdrawal/1',
        'evidence':{'path':'../receipt.json','sha256':evidence['sha256']},'rows':[]})
    config=write(tmp_path/'config.json',{'withdrawn_native_graphs':proof})
    closure=deps.dependency_closure({'config':config},namespace=str(tmp_path))
    assert {x['path'] for x in closure['files']}=={x['path'] for x in [evidence,proof,config]}

def test_asset_preflight_verifies_content_not_directory_existence(tmp_path):
    task=tmp_path/'ts-host';write(task/'dist/teacher/cli.js','actual compiled runtime')
    write(task/'package.json',{'name':'task-runtime'});write(tmp_path/'package.json',{'name':'root'});write(tmp_path/'package-lock.json',{'lockfileVersion':3})
    (task/'prelude.js').write_text('// exact runtime prelude\n')
    for name in ['holds.jsonl','family-holds.json','datasets.json']:
        write(tmp_path/'training/source-reviews'/name,{})
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


def test_explicit_participant_base_relocation_verifies_saved_inventory_not_old_path(tmp_path):
    import torch
    from natlang_neuralese.model.native_model import base_inventory
    base=tmp_path/'relocated-base'
    write(base/'config.json',{'model_type':'lfm2'})
    write(base/'model.safetensors','actual-pinned-base')
    expected=base_inventory(base)
    checkpoint=tmp_path/'shared.pt'
    torch.save({'participants':{'student':{'model_config':{'base':str(tmp_path/'old-absent-base'),
        'base_inventory':expected}}}},checkpoint)
    binding={'path':str(checkpoint),'sha256':deps.sha256_file_hex_cached(checkpoint),'bytes':checkpoint.stat().st_size}
    config=write(tmp_path/'semantic.json',{'models':{'student':{'base':str(base)}}})
    closure=deps.dependency_closure({'semantic_config':config,'parent_checkpoint':binding},namespace=str(tmp_path))
    assert [item['path'] for item in closure['assets']]==[str(base)]
    (base/'model.safetensors').write_text('wrong-alias-weights')
    with pytest.raises(ValueError,match='pinned base inventory changed'):
        deps.dependency_closure({'semantic_config':config,'parent_checkpoint':binding},namespace=str(tmp_path))

def test_refresh_update_includes_consumed_readiness_and_report(tmp_path):
    checkpoint=write(tmp_path/'update/checkpoint.pt','closed-updated-state')
    report=write(tmp_path/'update/report.json',{'completed':True,'numerics_ok':True})
    receipt=write(tmp_path/'update/native-training-receipt.json',{'training_receipt':report})
    closure=deps.dependency_closure({'updates':{'student':checkpoint}},namespace=str(tmp_path))
    assert {row['path'] for row in closure['files']}=={row['path'] for row in [checkpoint,report,receipt]}
    Path(receipt['path']).unlink()
    with pytest.raises(FileNotFoundError):
        deps.dependency_closure({'updates':{'student':checkpoint}},namespace=str(tmp_path))

def test_carried_remaining_ce_frames_include_complete_old_sources(tmp_path):
    import torch
    old=write(tmp_path/'old-train.jsonl','old untouched training span')
    held=write(tmp_path/'held.jsonl','held span')
    frame={'required_visits':{'untouched':1},'exposure_counts':{},'source_receipts':[old,held]}
    state=tmp_path/'training.pt';torch.save({'sft_progress':{'prior-ce':frame}},state)
    binding={'path':str(state),'sha256':deps.sha256_file_hex_cached(state),'bytes':state.stat().st_size}
    closure=deps.dependency_closure({'parent_checkpoint':binding},namespace=str(tmp_path))
    assert {str(state),old['path'],held['path']}=={row['path'] for row in closure['files']}
    # A fully consumed old cohort is metadata, not a mandatory live file dependency.
    frame['exposure_counts']['untouched']=1
    torch.save({'sft_progress':{'prior-ce':frame}},state)
    Path(old['path']).unlink();binding.update(sha256=deps.sha256_file_hex_cached(state),bytes=state.stat().st_size)
    assert [row['path'] for row in deps.dependency_closure({'parent_checkpoint':binding},namespace=str(tmp_path))['files']]==[str(state)]

def test_actual_task_packet_and_foundation_manifest_are_consumed_inputs(tmp_path):
    packet=write(tmp_path/'tasks.jsonl','protected task control')
    graph=write(tmp_path/'foundation.jsonl','raw communication graph')
    config=write(tmp_path/'config.json',{'foundation_native_graphs':graph,'trainer_parameters':{'task_eval':json.dumps({'packet':packet})}})
    closure=deps.dependency_closure({'config':config},namespace=str(tmp_path))
    assert {row['path'] for row in closure['files']}=={row['path'] for row in [packet,graph,config]}


def test_compiled_task_runtime_requires_and_binds_external_prelude(tmp_path):
    root=tmp_path/'runtime';host=root/'ts-host';(host/'dist/teacher').mkdir(parents=True)
    (host/'dist/teacher/cli.js').write_text('// compiled actual CLI import graph\n')
    for path in [host/'package.json',root/'package.json',root/'package-lock.json']:
        path.write_text('{}')
    reviews=root/'training/source-reviews';reviews.mkdir(parents=True)
    for name in ['holds.jsonl','family-holds.json','datasets.json']:
        (reviews/name).write_text('{}')
    with pytest.raises(ValueError,match='built CLI and pinned npm manifests'):
        deps.task_runtime_inventory(host)
    (host/'prelude.js').write_text('// first exact prelude\n')
    inventory=deps.task_runtime_inventory(host)
    asset={'path':str(host),'kind':'task-runtime','inventory':inventory,
           'sha256':deps.canonical_json_sha256_hex(inventory)}
    deps.verify_assets([asset])
    (host/'prelude.js').write_text('// different runtime semantics\n')
    with pytest.raises(ValueError,match='pre-staged asset content differs'):
        deps.verify_assets([asset])


def test_pending_source_declarations_and_complete_cohort_are_carried_dependencies(tmp_path):
    import torch
    full_rows=write(tmp_path/'all.jsonl','all admitted rows')
    long_rows=write(tmp_path/'long.jsonl','capacity deferred rows')
    evidence=write(tmp_path/'partition.json',{'exact_unchanged':True})
    full=write(tmp_path/'full.json',{'schema':'natlang.native-graph-sources/1','sources':[
        {'kind':'native-graphs',**full_rows}]})
    pending=write(tmp_path/'long.json',{'schema':'natlang.native-graph-sources/1','sources':[
        {'kind':'native-graphs',**long_rows}]})
    declaration=write(tmp_path/'pending.json',{'schema':'natlang.native-pending-sources/1',
        'tokenizer_fingerprint':'1'*64,'sources':[{'full_cohort':full,'pending_native_graphs':pending,
            'evidence':[evidence],'rows':[{'id':'long','binding':'2'*64,'required_visits':1}]}]})
    frame={'required_visits':{'long':1},'exposure_counts':{},'source_receipts':[],
        'pending_source_declarations':{'declaration':{'binding':declaration}}}
    checkpoint=tmp_path/'state.pt';torch.save({'sft_progress':{'ce':frame}},checkpoint)
    binding={'path':str(checkpoint),'sha256':deps.sha256_file_hex_cached(checkpoint)}
    closure=deps.dependency_closure({'parent_checkpoint':binding},namespace=str(tmp_path))
    assert {row['path'] for row in closure['files']}=={str(checkpoint),*[
        item['path'] for item in (declaration,full,pending,full_rows,long_rows,evidence)]}
    config=write(tmp_path/'config.json',{'pending_native_sources':declaration})
    closure=deps.dependency_closure({'config':config},namespace=str(tmp_path))
    assert declaration['path'] in {row['path'] for row in closure['files']}
    Path(long_rows['path']).write_text('changed')
    with pytest.raises(ValueError,match='content hash mismatch'):
        deps.dependency_closure({'parent_checkpoint':binding},namespace=str(tmp_path))


@pytest.mark.parametrize('condition',['valid','changed','missing','complete'])
def test_carried_offline_preference_objective_is_exact_consumed_dependency(tmp_path,condition):
    import torch
    source=write(tmp_path/'paired-graphs.jsonl','complete chosen and rejected histories')
    objective=write(tmp_path/'preference.json',{'rows':{'chosen':{'mask':[0,1]},'rejected':{'mask':[0,1]}}})
    frame={'required_visits':{'chosen':1,'rejected':1},'exposure_counts':{},'source_receipts':[source],
        'identity':{'objective_kind':'native-offline-pairwise',
            'objective':{**objective,'kind':'native-offline-pairwise','beta':.1,'margin':0.}}}
    if condition=='complete':frame['exposure_counts']={'chosen':1,'rejected':1}
    checkpoint=tmp_path/'state.pt';torch.save({'sft_progress':{'preference':frame}},checkpoint)
    binding={'path':str(checkpoint),'sha256':deps.sha256_file_hex_cached(checkpoint)}
    if condition=='changed':Path(objective['path']).write_text('changed mask')
    if condition in ('missing','complete'):Path(objective['path']).unlink()
    if condition=='changed':
        with pytest.raises(ValueError,match='dependency hash changed'):
            deps.dependency_closure({'parent_checkpoint':binding},namespace=str(tmp_path))
    elif condition=='missing':
        with pytest.raises(FileNotFoundError):
            deps.dependency_closure({'parent_checkpoint':binding},namespace=str(tmp_path))
    else:
        result={row['path'] for row in deps.dependency_closure({'parent_checkpoint':binding},namespace=str(tmp_path))['files']}
        assert result==({str(checkpoint)} if condition=='complete' else {str(checkpoint),source['path'],objective['path']})
