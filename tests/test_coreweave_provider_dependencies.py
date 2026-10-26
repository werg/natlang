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


def test_round_mandatory_stage_keeps_inherited_adaptation_opaque_and_actual_inputs_strict(tmp_path,monkeypatch):
    from natlang_neuralese.rl import native_completed_supervised as completed
    def no_replay(*args,**kwargs):
        raise AssertionError('worker must not replay its historical controller invocation')
    monkeypatch.setattr(completed,'verify',no_replay)
    historical=write(tmp_path/'historical.json',{'unconsumed_old_invocation':'preserved exactly'})
    train=write(tmp_path/'train.jsonl','actual admitted source\n')
    seed=write(tmp_path/'seed.json',{'schema':'natlang.native-graph-sources/1',
        'sources':[{'kind':'native-graphs',**train}]})
    held=write(tmp_path/'held.jsonl','protected held source\n')
    config=write(tmp_path/'stage.json',{'mandatory_sft_only':True,
        'completed_supervised_adaptation':historical,'seed_native_graphs':seed['path'],
        'held_native_graphs':held['path']})
    request={'kind':'round','config':config}
    closure=deps.dependency_closure(request,namespace=str(tmp_path))
    assert {r['path']for r in closure['files']}=={r['path']for r in [config,historical,seed,train,held]}
    Path(historical['path']).write_text('changed provenance')
    with pytest.raises(ValueError,match='consumed dependency hash changed'):deps.dependency_closure(request,namespace=str(tmp_path))
    write(Path(historical['path']),{'unconsumed_old_invocation':'preserved exactly'})
    Path(train['path']).write_text('changed current training source')
    with pytest.raises(ValueError,match='hash mismatch'):deps.dependency_closure(request,namespace=str(tmp_path))


@pytest.mark.parametrize('kind,mandatory',[('round',False),('round',1),('semantic',True),(None,True)])
def test_other_contexts_still_validate_completed_adaptation(tmp_path,monkeypatch,kind,mandatory):
    from natlang_neuralese.rl import native_completed_supervised as completed
    def reject(reference):raise ValueError('active controller adaptation rejected')
    monkeypatch.setattr(completed,'verify',reject)
    proof=write(tmp_path/'proof.json',{'preserved':True})
    config=write(tmp_path/'config.json',{'mandatory_sft_only':mandatory,'completed_supervised_adaptation':proof})
    with pytest.raises(ValueError,match='active controller adaptation rejected'):
        deps.dependency_closure({'kind':kind,'config':config},namespace=str(tmp_path))


def test_withdrawal_relative_evidence_stays_in_exact_provider_closure(tmp_path):
    evidence=write(tmp_path/'receipt.json',{'audit':'rejected targets'})
    proof=write(tmp_path/'model/withdrawal.json',{'schema':'natlang.native-graph-withdrawal/1',
        'evidence':{'path':'../receipt.json','sha256':evidence['sha256']},'rows':[]})
    config=write(tmp_path/'config.json',{'withdrawn_native_graphs':proof})
    closure=deps.dependency_closure({'config':config},namespace=str(tmp_path))
    assert {x['path'] for x in closure['files']}=={x['path'] for x in [evidence,proof,config]}


def test_internal_bound_parent_reference_normalizes_without_losing_content_authority(tmp_path):
    directory=tmp_path/'admission';directory.mkdir()
    source=write(tmp_path/'source.jsonl','same immutable source\n')
    relative={**source,'path':str(directory/'..'/'source.jsonl')}
    config=write(tmp_path/'config.json',{'operational_inputs':[relative]})
    closure=deps.dependency_closure({'config':config},namespace=str(tmp_path))
    assert {x['path']for x in closure['files']}=={config['path'],source['path']}
    Path(source['path']).write_text('changed')
    with pytest.raises(ValueError,match='hash changed'):deps.dependency_closure({'config':config},namespace=str(tmp_path))


def test_parent_traversal_cannot_change_symlink_meaning_or_escape_namespace(tmp_path):
    root=tmp_path/'root';root.mkdir();outside=tmp_path/'outside';(outside/'child').mkdir(parents=True)
    source=write(root/'source.jsonl','same source\n');write(outside/'source.jsonl','same source\n')
    alias=root/'alias';alias.symlink_to(outside/'child',target_is_directory=True)
    ref={**source,'path':str(alias/'..'/'source.jsonl')}
    config=write(root/'config.json',{'operational_inputs':[ref]})
    with pytest.raises(ValueError,match='symlink target'):deps.dependency_closure({'config':config},namespace=str(root))
    ref['path']=str(root/'..'/'outside'/'source.jsonl');config=write(root/'config.json',{'operational_inputs':[ref]})
    with pytest.raises(ValueError,match='escaped'):deps.dependency_closure({'config':config},namespace=str(root))

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


def test_shared_checkpoint_custodies_current_source_task_baseline_and_runtime(tmp_path):
    import torch
    from natlang_neuralese.eval.native_state import task_execution_receipt
    from natlang_neuralese.eval.task_packet import TASK_OUTCOMES
    identity={'checkpoint_sha256':'a'*64,'tokenizer_fingerprint':'b'*64,
        'embedding_weight_sha256':'c'*64,'architecture':'lfm2'}
    run=tmp_path/'baseline';run.mkdir()
    packet='d'*64;counts={name:0 for name in TASK_OUTCOMES};counts['invalid_action']=1
    score={'schema':'natlang.in-run-task-eval/1','step':0,'run':str(run),'cases':1,'case_indices':[0],
        'packet_sha256':packet,'outcome_counts':counts,'complete':True,'eligible':True,'task_successes':0,
        'cases_detail':[{'id':'case-0','outcome':'invalid_action'}]}
    score_path=run/'task-score.json';write(score_path,score)
    receipt=task_execution_receipt({key:value for key,value in score.items()if key!='cases_detail'},identity,
        checkpoint_sha256=identity['checkpoint_sha256'])
    runtime_file=write(tmp_path/'runtime-module.py','runtime code bytes\n')
    runtime={'schema':'natlang.task-eval-runtime-identity/1','repo':str(tmp_path),
        'runtime':str(tmp_path/'ts-host'),'code_commit':'e'*40,'files':[runtime_file]}
    receipt.update(source_baseline=True,baseline_comparison_floor=0,
        task_eval_policy_sha256='f'*64,task_runtime_identity=runtime)
    receipt_path=run/'native-task-execution-receipt.json';write(receipt_path,receipt)
    baseline={'native_identity':identity,'task_successes':0,'task_eval_policy_sha256':'f'*64,
        'task_runtime_identity':runtime,'task_execution_receipt':write(receipt_path,receipt),
        'score':write(score_path,score)}
    checkpoint=tmp_path/'checkpoint.pt'
    torch.save({'schema':'natlang.neuralese-native-checkpoint/1',
        'training':{'source_task_baselines':{'lfm:base':baseline}}},checkpoint)
    checkpoint_ref={'path':str(checkpoint),'sha256':deps.sha256_file_hex_cached(checkpoint),
        'bytes':checkpoint.stat().st_size}
    closure=deps.dependency_closure({'parent_checkpoint':checkpoint_ref},namespace=str(tmp_path))
    files={row['path']:row for row in closure['files']}
    assert receipt_path.as_posix() in files and score_path.as_posix() in files
    assert runtime_file['path'] in files
    runtime_path=Path(runtime_file['path']);runtime_path.write_text('runtime changed\n')
    with pytest.raises(ValueError,match='hash changed'):
        deps.dependency_closure({'parent_checkpoint':checkpoint_ref},namespace=str(tmp_path))

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


@pytest.mark.parametrize('location', ['curriculum', 'pending', 'checkpoint-history', 'relocated-curriculum'])
def test_live_policy_source_closure_preserves_behavior_semantics_and_every_archive_member(tmp_path, location, monkeypatch):
    checkpoint=write(tmp_path/'behavior.pt','immutable-semantic-parent')
    base_files=[]
    if location=='relocated-curriculum':
        import torch
        from natlang_neuralese.model.native_model import base_inventory
        base=tmp_path/'current-base'
        base_files=[write(base/'config.json',{'model_type':'lfm2'}),write(base/'model.safetensors','same-base')]
        path=Path(checkpoint['path'])
        torch.save({'participant':'student','model_config':{'base':str(tmp_path/'old-absent-base'),
            'base_inventory':base_inventory(base)}},path)
        checkpoint={'path':str(path),'sha256':deps.sha256_file_hex_cached(path)}
    token=write(tmp_path/'tokenizer/tokenizer.json','behavior-tokenizer-bytes')
    adapter=write(tmp_path/'adapter.json',{'schema':'natlang.native-semantic-adapter/1',
        'checkpoint':checkpoint,'models':{'student':{'native_identity':{'checkpoint_sha256':checkpoint['sha256']}}},
        'tokenizers':{'student':{'path':'tokenizer','files':{'tokenizer.json':token['sha256']}}}})
    rollout=write(tmp_path/'rollouts.jsonl','whole-episode')
    raw=write(tmp_path/'producer.state','exact-state-and-derivation')
    dependency=write(tmp_path/'adapter.tensor','exact-recursive-artifact')
    original_read=Path.read_bytes
    def no_payload_read(path):
        if str(path) in (raw['path'],dependency['path']):
            raise AssertionError('metadata closure materialized raw KV bytes')
        return original_read(path)
    monkeypatch.setattr(Path,'read_bytes',no_payload_read)
    archive=write(tmp_path/'blocks.json',{'schema':'natlang.rl-block-bundle/1',
        'policy_id':checkpoint['sha256'],'blocks':{'producer':{**raw,'path':'producer.state'},
            'dependency':{**dependency,'path':'adapter.tensor'}}})
    source={'rollouts':rollout,'block_bundle':archive,'adapter':adapter,'mode':'rft','collector_participant':'student'}
    calls=[];original=deps.composite_dependencies
    def observed(path, **kwargs):
        calls.append((str(path),kwargs.get('semantic_only',False)))
        return original(path,**kwargs)
    monkeypatch.setattr(deps,'composite_dependencies',observed)
    if location in ('curriculum','relocated-curriculum'):
        value={'policy_training':{'sources':[source]}}
        if location=='relocated-curriculum':value['models']={'student':{'base':str(base)}}
        control=write(tmp_path/'control.json',value)
        request={'config':control}
    elif location=='pending':
        control=write(tmp_path/'state.json',{'pending_semantic_source':source})
        request={'state':control}
    else:
        import torch
        path=tmp_path/'current.pt';torch.save({'training':{'policy_source_history':[source]}},path)
        control={'path':str(path),'sha256':deps.sha256_file_hex_cached(path)}
        request={'parent_checkpoint':control}
    closure=deps.dependency_closure(request,namespace=str(tmp_path))
    assert {row['path'] for row in closure['files']}=={row['path'] for row in
        [control,checkpoint,token,adapter,rollout,raw,dependency,archive,*base_files]}
    assert (checkpoint['path'],True) in calls
    Path(dependency['path']).write_text('changed recursive artifact')
    with pytest.raises(ValueError,match='hash changed'):
        deps.dependency_closure(request,namespace=str(tmp_path))


def test_archive_metadata_verifier_rejects_invalid_member_size(tmp_path):
    from natlang_neuralese.rl.graph_replay import verified_bundle_path
    raw=write(tmp_path/'raw.state','same-bytes')
    bundle=write(tmp_path/'bundle.json',{'schema':'natlang.rl-block-bundle/1','policy_id':'actor',
        'blocks':{'member':{**raw,'path':'raw.state','bytes':raw['bytes']+1}}})
    with pytest.raises(ValueError,match='size/hash changed'):
        verified_bundle_path(bundle['path'],'member',policy_id='actor')
    config=write(tmp_path/'config.json',{'external_bundles':{'student':bundle}})
    with pytest.raises(ValueError,match='size changed'):
        deps.dependency_closure({'config':config},namespace=str(tmp_path))


@pytest.mark.parametrize('location', ['request', 'state'])
def test_raw_partial_checkpoint_uses_none_participant_base_relocation(tmp_path, location):
    import torch
    from natlang_neuralese.model.native_model import base_inventory
    base=tmp_path/'current-base'
    write(base/'config.json',{'model_type':'lfm2'})
    write(base/'model.safetensors','exact-native-base')
    path=tmp_path/'partial.pt'
    torch.save({'model_config':{'base':str(tmp_path/'absent-saved-base'),
        'base_inventory':base_inventory(base)}},path)
    partial={'path':str(path),'sha256':deps.sha256_file_hex_cached(path)}
    config=write(tmp_path/'config.json',{'base':str(base),'native_participant':None})
    request={'kind':'round','config':config}
    if location=='state':
        request['state']=write(tmp_path/'state.json',{'partial_checkpoint':partial})
    else:
        request['partial_checkpoint']=partial
    closure=deps.dependency_closure(request,namespace=str(tmp_path))
    assert str(path) in {row['path'] for row in closure['files']}
    assert [row['path'] for row in closure['assets']]==[str(base)]

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


def test_same_full_checkpoint_is_decoded_once_per_closure_and_keeps_alias_paths(tmp_path,monkeypatch):
    import torch
    from natlang_neuralese.train.checkpoint_policy import CheckpointPolicy
    parent=tmp_path/'parent.pt';torch.save({'schema':'native-bare-test','training':{}},parent)
    alias=tmp_path/'alias.pt';alias.symlink_to(parent)
    reference={'path':str(parent),'sha256':deps.sha256_file_hex_cached(parent)}
    config=write(tmp_path/'config.json',{'models':{'student':{'checkpoint':str(parent)}}})
    state=write(tmp_path/'state.json',{'checkpoint':str(parent)})
    actual=CheckpointPolicy.load;calls=[]
    def load(path,*args,**kwargs):
        calls.append(str(Path(path).resolve()));return actual(path,*args,**kwargs)
    monkeypatch.setattr(CheckpointPolicy,'load',load)
    request={'config':config,'parent_checkpoint':{**reference,'path':str(alias)},'state':state}
    closure=deps.dependency_closure(request,namespace=str(tmp_path))
    assert calls==[str(parent)]
    assert {str(parent),str(alias)}<={item['path'] for item in closure['files']}
    calls.clear();deps.dependency_closure(request,namespace=str(tmp_path))
    assert calls==[str(parent)]  # Nothing persists across invocations.
    alias.unlink();alias.write_text('changed alias')
    with pytest.raises(ValueError,match='hash changed'):deps.dependency_closure(request,namespace=str(tmp_path))


def test_checkpoint_participant_context_is_not_coalesced(tmp_path,monkeypatch):
    import torch
    from natlang_neuralese.train.checkpoint_policy import CheckpointPolicy
    parent=tmp_path/'parent.pt';torch.save({'training':{}},parent)
    config=write(tmp_path/'config.json',{'models':{'student':{'checkpoint':str(parent)}}})
    reference={'path':str(parent),'sha256':deps.sha256_file_hex_cached(parent),'participant':'other'}
    actual=CheckpointPolicy.load;calls=[]
    def load(path,*args,**kwargs):calls.append(path);return actual(path,*args,**kwargs)
    monkeypatch.setattr(CheckpointPolicy,'load',load)
    deps.dependency_closure({'config':config,'parent_checkpoint':reference},namespace=str(tmp_path))
    assert len(calls)==2


@pytest.mark.parametrize('disabled',[None,'','null'])
@pytest.mark.parametrize('key',['task_eval','trainer_parameters','seed_trainer_parameters'])
def test_disabled_task_specs_match_shared_trainer_parser(tmp_path,disabled,key):
    value={key:disabled}if key=='task_eval'else{key:{'task_eval':disabled}}
    config=write(tmp_path/'config.json',value)
    assert [r['path']for r in deps.dependency_closure({'config':config},namespace=str(tmp_path))['files']]==[config['path']]


@pytest.mark.parametrize('location',['seed','literal','equals','placeholder'])
def test_actual_empty_trainer_task_spec_retains_explicit_protected_override(tmp_path,location):
    packet=write(tmp_path/'held.jsonl','exact protected packet')
    root=tmp_path/'runtime';host=root/'ts-host';(host/'dist/teacher').mkdir(parents=True)
    (host/'dist/teacher/cli.js').write_text('// exact built collector')
    for path in(host/'package.json',root/'package.json',root/'package-lock.json'):path.write_text('{}')
    reviews=root/'training/source-reviews';reviews.mkdir(parents=True)
    for name in('holds.jsonl','family-holds.json','datasets.json'):(reviews/name).write_text('{}')
    (host/'prelude.js').write_text('// exact protected runtime prelude')
    specification=json.dumps({'packet':packet,'runtime':str(host)})
    value={'trainer_parameters':{'task_eval':''}}
    if location=='seed':value['seed_trainer_parameters']={'task_eval':specification}
    elif location=='literal':value['seed_command']=['python','-m','trainer','--task-eval',specification]
    elif location=='equals':value['learner_command']=['python','-m','trainer','--task-eval='+specification]
    else:value.update(task_eval=specification,seed_command=['trainer','--task-eval','{task_eval}'])
    config=write(tmp_path/'config.json',value)
    closure=deps.dependency_closure({'config':config},namespace=str(tmp_path))
    assert {r['path']for r in closure['files']}=={config['path'],packet['path']}
    assert [a['path']for a in closure['assets']]==[str(host)]
    assert closure['assets'][0]['inventory']==deps.task_runtime_inventory(host)
    Path(packet['path']).write_text('changed protected bytes')
    with pytest.raises(ValueError,match='dependency hash changed'):deps.dependency_closure({'config':config},namespace=str(tmp_path))


@pytest.mark.parametrize('command',[['trainer','--task-eval'],['trainer','--task-eval','{'],['trainer','--task-eval','{missing_spec}']])
def test_task_command_missing_or_malformed_spec_rejected_before_launch(tmp_path,command):
    config=write(tmp_path/'config.json',{'seed_command':command})
    with pytest.raises(ValueError):deps.dependency_closure({'config':config},namespace=str(tmp_path))



def test_carried_dpo_external_inputs_are_verified_in_place_not_ferried(tmp_path):
    import torch
    namespace=tmp_path/'production';namespace.mkdir()
    readonly=tmp_path/'d24'
    source=write(readonly/'paired-graphs.jsonl','complete chosen and rejected histories')
    objective=write(readonly/'objective.json',{'rows':{}})
    frame={'required_visits':{'pair':1},'exposure_counts':{},'source_receipts':[source],
        'identity':{'objective_kind':'native-offline-pairwise','objective':objective}}
    checkpoint=namespace/'state.pt';torch.save({'sft_progress':{'preference':frame}},checkpoint)
    binding={'path':str(checkpoint),'sha256':deps.sha256_file_hex_cached(checkpoint)}
    with pytest.raises(ValueError,match='escaped'):
        deps.dependency_closure({'parent_checkpoint':binding},namespace=str(namespace))
    closure=deps.dependency_closure({'parent_checkpoint':binding},namespace=str(namespace),readonly_input_roots=[str(readonly)])
    assert [item['path'] for item in closure['files']]==[str(checkpoint)]
    assert {item['path'] for item in closure['assets']}=={source['path'],objective['path']}
    assert all(item['kind']=='input-file' for item in closure['assets'])
    deps.verify_assets(closure['assets'],readonly_input_roots=[str(readonly)])
    with pytest.raises(ValueError,match='checkpoint escaped'):
        deps.dependency_closure({'parent_checkpoint':source},namespace=str(namespace),readonly_input_roots=[str(readonly)])


@pytest.mark.parametrize('failure',['undeclared','escape','hash','size','missing','boolean-size'])
def test_readonly_input_file_assets_fail_closed(tmp_path,failure):
    root=tmp_path/'declared';item=write(root/'input.json','exact source bytes')
    item['kind']='input-file'
    roots=[str(root)]
    if failure=='undeclared':roots=[]
    elif failure=='escape':
        outside=tmp_path/'outside';outside.write_text('exact source bytes')
        Path(item['path']).unlink();Path(item['path']).symlink_to(outside)
    elif failure=='hash':item['sha256']='0'*64
    elif failure=='size':item['bytes']+=1
    elif failure=='missing':Path(item['path']).unlink()
    elif failure=='boolean-size':item['bytes']=True
    with pytest.raises(ValueError):deps.verify_assets([item],readonly_input_roots=roots)


def test_readonly_root_configuration_rejects_broad_relative_and_symlink_roots(tmp_path):
    target=tmp_path/'actual';target.mkdir();alias=tmp_path/'alias';alias.symlink_to(target,target_is_directory=True)
    for roots in ['not-list',['/'],['relative'],[str(alias)],[str(target),str(target)], [str(target)+'/../actual']]:
        with pytest.raises(ValueError):deps.verify_assets([],readonly_input_roots=roots)


@pytest.mark.parametrize('target_kind',['production','other-readonly'])
def test_consumed_readonly_alias_preserves_binding_across_declared_namespaces(tmp_path,target_kind):
    import torch
    namespace=tmp_path/'production';namespace.mkdir()
    readonly=tmp_path/'historical-cache';readonly.mkdir()
    target_root=namespace if target_kind=='production' else tmp_path/'canonical-readonly'
    target=write(target_root/'rows.jsonl','{"id":"complete-source"}\n')
    alias=readonly/'rows.jsonl';alias.symlink_to(target['path'])
    source={**target,'path':str(alias)}
    objective=write(readonly/'objective.json',{'rows':{}})
    frame={'required_visits':{'pair':1},'exposure_counts':{},'source_receipts':[source],
        'identity':{'objective_kind':'native-offline-pairwise','objective':objective}}
    checkpoint=namespace/'state.pt';torch.save({'sft_progress':{'preference':frame}},checkpoint)
    binding={'path':str(checkpoint),'sha256':deps.sha256_file_hex_cached(checkpoint)}
    roots=[str(readonly)]+([str(target_root)]if target_kind=='other-readonly'else[])
    closure=deps.dependency_closure({'parent_checkpoint':binding},namespace=str(namespace),readonly_input_roots=roots)
    asset=next(item for item in closure['assets']if item['path']==str(alias))
    assert asset=={**source,'kind':'input-file'}
    deps.verify_assets(closure['assets'],namespace=str(namespace),readonly_input_roots=roots)
    # A declared target never authorizes an undeclared lexical source path.
    with pytest.raises(ValueError,match='escaped'):
        deps.verify_assets([asset],namespace=str(namespace),readonly_input_roots=[])
    outside=tmp_path/'outside';outside.write_bytes(Path(target['path']).read_bytes())
    alias.unlink();alias.symlink_to(outside)
    with pytest.raises(ValueError,match='escaped'):
        deps.verify_assets([asset],namespace=str(namespace),readonly_input_roots=roots)
    with pytest.raises(ValueError,match='escaped'):
        deps.dependency_closure({'parent_checkpoint':binding},namespace=str(namespace),readonly_input_roots=roots)


def test_private_readonly_helper_retains_same_root_contract_without_namespace(tmp_path):
    left=tmp_path/'left';left.mkdir()
    target=write(tmp_path/'right'/'index.pkl','exact index')
    alias=left/'index.pkl';alias.symlink_to(target['path'])
    roots=(left,tmp_path/'right')
    with pytest.raises(ValueError,match='escaped'):deps._readonly_file(alias,roots)
    assert deps._readonly_file(alias,roots,namespace=tmp_path/'production')==alias


@pytest.mark.parametrize('change',['hash','size','boolean-size'])
def test_declared_production_alias_still_requires_exact_content(tmp_path,change):
    namespace=tmp_path/'production'
    target=write(namespace/'cache.pkl','exact serialized source index')
    readonly=tmp_path/'historical-cache';readonly.mkdir()
    alias=readonly/'cache.pkl';alias.symlink_to(target['path'])
    asset={**target,'path':str(alias),'kind':'input-file'}
    if change=='hash':asset['sha256']='0'*64
    elif change=='size':asset['bytes']+=1
    else:asset['bytes']=True
    with pytest.raises(ValueError,match='content differs'):
        deps.verify_assets([asset],namespace=str(namespace),readonly_input_roots=[str(readonly)])



def test_checkpoint_dependency_cannot_use_readonly_input_escape(tmp_path):
    namespace=tmp_path/'production';readonly=tmp_path/'d24'
    external=write(readonly/'checkpoint.pt','immutable binary checkpoint')
    alias=namespace/'offloaded.pt';alias.parent.mkdir();alias.symlink_to(external['path'])
    parent=write(namespace/'checkpoint.json',{'schema':'natlang.neuralese-native-composite-checkpoint/1',
        'base_checkpoint':{**external,'path':str(alias)}})
    with pytest.raises(ValueError,match='checkpoint dependency escaped'):
        deps.dependency_closure({'parent_checkpoint':parent},namespace=str(namespace),readonly_input_roots=[str(readonly)])



def test_separately_bound_operational_inputs_are_in_actual_file_closure(tmp_path):
    script=write(tmp_path/'control/profile.py','exact read-only update observer')
    config=write(tmp_path/'config.json',{'operational_inputs':[script]})
    closure=deps.dependency_closure({'config':config},namespace=str(tmp_path))
    assert {row['path']for row in closure['files']}=={script['path'],config['path']}
    Path(script['path']).write_text('changed observer')
    with pytest.raises(ValueError,match='dependency hash changed'):
        deps.dependency_closure({'config':config},namespace=str(tmp_path))


def test_declared_semantic_participants_use_actual_complete_checkpoint_closure(tmp_path,monkeypatch):
    checkpoint=write(tmp_path/'checkpoint.pt','test-binary-input')
    calls=[]
    def dependencies(path,**kwargs):
        calls.append((path,kwargs));return {checkpoint['sha256']:Path(checkpoint['path'])}
    monkeypatch.setattr(deps,'composite_dependencies',dependencies)
    closure=deps.dependency_closure({'semantic_participants':[checkpoint]},namespace=str(tmp_path))
    assert closure['files']==[checkpoint] and not closure['assets']
    assert len(calls)==1 and calls[0][1]['semantic_only']is True
    Path(checkpoint['path']).write_text('changed')
    with pytest.raises(ValueError):
        deps.dependency_closure({'semantic_participants':[checkpoint]},namespace=str(tmp_path))


def test_foundation_request_is_an_exact_optional_worker_input(tmp_path):
    request=write(tmp_path/'request.json',{'schema':'natlang.native-foundation-qualification-request/1'})
    config=write(tmp_path/'config.json',{'foundation_qualification_request':request,
        'trainer_parameters':{'foundation_qualification_request':request['path']}})
    closure=deps.dependency_closure({'config':config},namespace=str(tmp_path))
    assert {row['path'] for row in closure['files']}=={config['path'],request['path']}
    config=write(tmp_path/'config.json',{'trainer_parameters':{'foundation_qualification_request':request['path']}})
    with pytest.raises(ValueError,match='lacks its exact bound input'):
        deps.dependency_closure({'config':config},namespace=str(tmp_path))
    config=write(tmp_path/'config.json',{'foundation_qualification_request':{**request,'bytes':request['bytes']+1}})
    with pytest.raises(ValueError,match='size changed'):
        deps.dependency_closure({'config':config},namespace=str(tmp_path))
    config=write(tmp_path/'config.json',{})
    assert deps.dependency_closure({'config':config},namespace=str(tmp_path))['files']==[config]


def test_snapshot_adoption_ferries_exact_current_status_pointer_and_proof(tmp_path):
    from natlang_neuralese.rl import native_role_release as roles
    exchange=tmp_path/'exchange';exchange.mkdir()
    assignment={name:{'source_host_id':name+'-fixture','owned_models':[name]} for name in ('lfm','mellum')}
    plan=write(tmp_path/'plan.json',{'models':[{'model':name}for name in assignment],
        'owned_models':['mellum'],'role_exchange':{'schema':roles.EXCHANGE,'role':'mellum',
            'semantic_consumer':'mellum','directory':str(exchange),'roles':assignment}})
    release=write(exchange/'release.json',{'schema':roles.SCHEMA,'role':'lfm','kind':'released',
        'optimizer_released':True,'scope':'typed closure fixture; not numerical qualification'})
    pointer=write(exchange/'lfm-status.json',release)
    config=write(tmp_path/'config.json',{'snapshot_adoption':{'schema':'natlang.native-snapshot-adoption/1',
        'plan':plan,'releases':[release]}})
    closure=deps.dependency_closure({'config':config},namespace=str(tmp_path))
    assert {row['path']for row in closure['files']}=={row['path']for row in (config,plan,release,pointer)}
    write(exchange/'lfm-status.json',{**release,'sha256':'a'*64})
    with pytest.raises(ValueError,match='retired role release'):
        deps.dependency_closure({'config':config},namespace=str(tmp_path))


def test_snapshot_ferry_restores_only_declared_closure_with_original_host_unavailable(tmp_path):
    """Actual Torch decode + same-namespace receiver, with no source files left.

    This checks transport closure, not numerical CUDA model qualification.
    """
    import shutil
    import torch
    from natlang_neuralese.train.checkpoint_policy import CheckpointPolicy
    from natlang_neuralese.train.native_continuation import collect_sft_progress
    source=tmp_path/'production';source.mkdir()
    ordinary=write(source/'ordinary.jsonl','ordinary mandatory training row\n')
    pairs=write(source/'pairs.jsonl','genuine communication source row\n')
    nested=write(source/'nested.json',{'schema':'natlang.native-graph-sources/1',
        'sources':[{'kind':'native-graphs',**ordinary},{'kind':'semantic-pairs',**pairs,'model':'student'}]})
    old_recipe=write(source/'old-recipe.json',{'policy':'historical identity; not read by metadata recovery'})
    completed=write(source/'completed-source.jsonl','already consumed row\n')
    frame={'required_visits':{'ordinary':2,'communication':1},'exposure_counts':{'ordinary':1},
        'source_receipts':[nested]}
    state={'training':{'options':{'recipe_binding':old_recipe},'sft_progress':{'mandatory':frame,
        'completed':{'required_visits':{'done':1},'exposure_counts':{'done':1},'source_receipts':[completed]}},
        'optimizer':{'actual-buffer':torch.arange(8.)},'rng':{'torch':torch.get_rng_state()}}}
    path=source/'checkpoint.pt';torch.save(state,path)
    binding={'path':str(path),'sha256':deps.sha256_file_hex_cached(path),'bytes':path.stat().st_size}
    closure=deps.dependency_closure({'semantic_participants':[binding]},namespace=str(source))
    expected={str(path),nested['path'],ordinary['path'],pairs['path']}
    assert {row['path'] for row in closure['files']}==expected
    transport=tmp_path/'transport';transport.mkdir()
    for row in closure['files']:
        shutil.copyfile(row['path'],transport/row['sha256'])
    # Delete every original, then construct the receiver solely from the ferry
    # inventory. Real sandboxes use the same /work/native-production namespace.
    shutil.rmtree(source)
    for row in closure['files']:
        destination=Path(row['path']);destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(transport/row['sha256'],destination)
        assert destination.stat().st_size==row['bytes']
        assert deps.sha256_file_hex_cached(destination)==row['sha256']
    assert not Path(old_recipe['path']).exists() and not Path(completed['path']).exists()
    checked=deps.dependency_closure({'semantic_participants':[binding]},namespace=str(source))
    assert checked==closure
    loaded=CheckpointPolicy.load(path)
    assert collect_sft_progress(loaded['training'])['mandatory']==frame
    torch.testing.assert_close(loaded['training']['optimizer']['actual-buffer'],
        state['training']['optimizer']['actual-buffer'],rtol=0,atol=0)
    assert torch.equal(loaded['training']['rng']['torch'],state['training']['rng']['torch'])


@pytest.mark.parametrize('mutation',[None,'historical','future_source','future_held'])
def test_verified_future_keeps_inherited_adaptation_as_bound_provenance(tmp_path,monkeypatch,mutation):
    from natlang_neuralese.rl import native_completed_supervised as completed
    historical=write(tmp_path/'old-adaptation.json',{'old_invocation':'historical defaults'})
    source=write(tmp_path/'future.jsonl','future admitted source\n')
    held=write(tmp_path/'held.jsonl','protected source\n')
    source_manifest=write(tmp_path/'sources.json',{'schema':'natlang.native-graph-sources/1',
        'sources':[{'kind':'native-graphs',**source}]})
    held_manifest=write(tmp_path/'held-sources.json',{'schema':'natlang.native-graph-sources/1',
        'sources':[{'kind':'native-graphs',**held}]})
    future=write(tmp_path/'future-config.json',{'completed_supervised_adaptation':historical,
        'seed_native_graphs':source_manifest['path'],'held_native_graphs':held_manifest['path']})
    active=write(tmp_path/'active-adaptation.json',{'actual':'current invocation'})
    config=write(tmp_path/'config.json',{'completed_supervised_adaptation':active})
    calls=[]
    def verified(reference):
        calls.append(reference)
        assert reference==active,'must not reinterpret historical defaults'
        return {'future_config':future},{active['path']:active,future['path']:future}
    monkeypatch.setattr(completed,'verified_dependencies',verified)
    if mutation:
        target={'historical':historical,'future_source':source,'future_held':held}[mutation]
        Path(target['path']).write_text('changed bound bytes')
        with pytest.raises(ValueError,match='hash (changed|mismatch)'):
            deps.dependency_closure({'kind':'round','config':config},namespace=str(tmp_path))
    else:
        closure=deps.dependency_closure({'kind':'round','config':config},namespace=str(tmp_path))
        assert {r['path']for r in closure['files']}=={r['path']for r in
            [config,active,historical,future,source_manifest,source,held_manifest,held]}
    assert calls==[active]


@pytest.mark.parametrize('complete',[False,True])
def test_foundation_recovery_and_header_are_in_checkpoint_dependency_closure(tmp_path,complete):
    import torch
    payload=tmp_path/'staged'/'graphs.jsonl'
    payload.parent.mkdir();payload.write_text('{"id":"actual-graph"}\n')
    original='/retired/source/graphs.jsonl'
    source_map=write(tmp_path/'.natlang-source-map.json',{'schema':'natlang.staged-source-map/1',
        'paths':{original:str(payload)}})
    header=write(tmp_path/'foundation.json',{'schema':'natlang.native-graph-sources/1',
        'sources':[{'kind':'native-graphs','path':original,'sha256':deps.sha256_file_hex_cached(payload)}]})
    from natlang_neuralese.train.native_foundation_recovery import native_graph_bindings
    header['dependencies']=[item for item in native_graph_bindings(header)
        if item['path']!=str(Path(header['path']).resolve())]
    historical_recipe=write(tmp_path/'historical-recipe.json',{
        'checkpoint_path':'/pruned/old/checkpoint.pt','declared':'documentary only'})
    metadata=write(tmp_path/'historical-metadata.json',{
        'checkpoint_path':'/also-pruned/checkpoint.pt','historical':'documentary only'})
    verification=write(tmp_path/'verification.json',{'checkpoint_path':'/pruned/verify-target.pt'})
    proof=write(tmp_path/'recovery.json',{'schema':'natlang.native-foundation-declaration-recovery/1',
        'stage_id':'full','historical_recipe':historical_recipe,'historical_training_metadata':metadata,
        'historical_checkpoint_verification':verification,'foundation_sources':header,
        'reason':'exact declaration recovery'})
    options={'communication_foundation':True,'foundation_native_graphs':header['path'],'foundation_native_graphs_binding':header,
        'foundation_declaration_recovery':proof['path'],
        'foundation_declaration_recovery_binding':proof}
    historical={'options':options,'required_visits':{} if complete else {'actual-graph':1},
        'source_row_order':['actual-graph'],'exposure_counts':{'actual-graph':0},
        'source_row_bindings':{'actual-graph':'a'*64}}
    training={'schema':'natlang.neuralese-native-sft-training/1','identity':{'objective':None},
        'options':{'stage_id':'next-stage'},'sft_progress':{'full':historical},
        'global_step':0,'source_row_order':[],'exposure_counts':{},'source_row_bindings':{},
        'required_visits':{},'completed_foundations':[]}
    checkpoint=tmp_path/'state.pt';torch.save({'training':training},checkpoint)
    binding={'path':str(checkpoint),'sha256':deps.sha256_file_hex_cached(checkpoint)}
    closure=deps.dependency_closure({'parent_checkpoint':binding},namespace=str(tmp_path))
    paths={item['path'] for item in closure['files']}
    assert paths >= {checkpoint.resolve().as_posix(),Path(proof['path']).as_posix(),
        Path(historical_recipe['path']).as_posix(),Path(metadata['path']).as_posix(),
        Path(verification['path']).as_posix(),Path(header['path']).as_posix(),
        payload.resolve().as_posix(),Path(source_map['path']).as_posix()}
    assert not any('/pruned/' in item or '/also-pruned/' in item for item in paths)


def test_foundation_recovery_dependency_binding_omission_and_tampering_fail_closed(tmp_path):
    import torch
    header=write(tmp_path/'header.json',{'schema':'natlang.native-graph-sources/1',
        'sources':[{'kind':'semantic-pairs','path':'pairs.jsonl','sha256':deps.sha256_file_hex_cached(
            write(tmp_path/'pairs.jsonl','exact').get('path')),'model':'port'}]})
    from natlang_neuralese.train.native_foundation_recovery import native_graph_bindings
    header['dependencies']=[item for item in native_graph_bindings(header)
        if item['path']!=str(Path(header['path']).resolve())]
    docs=[write(tmp_path/name,{'historical':'opaque'}) for name in
        ('recipe.json','metadata.json','verification.json')]
    proof=write(tmp_path/'proof.json',{'schema':'natlang.native-foundation-declaration-recovery/1',
        'stage_id':'full','historical_recipe':docs[0],'historical_training_metadata':docs[1],
        'historical_checkpoint_verification':docs[2],'foundation_sources':header,'reason':'recover'})
    training={'options':{'foundation_declaration_recovery':proof['path'],'foundation_native_graphs':header['path'],
        'foundation_native_graphs_binding':header},'sft_progress':{}}
    checkpoint=tmp_path/'state.pt';torch.save({'training':training},checkpoint)
    binding={'path':str(checkpoint),'sha256':deps.sha256_file_hex_cached(checkpoint)}
    training['options']['communication_foundation']=True
    del training['options']['foundation_native_graphs_binding']
    torch.save({'training':training},checkpoint)
    binding['sha256']=deps.sha256_file_hex_cached(checkpoint)
    with pytest.raises(ValueError,match='active communication foundation lacks'):
        deps.dependency_closure({'parent_checkpoint':binding},namespace=str(tmp_path))
    training['options']['foundation_native_graphs_binding']=header
    torch.save({'training':training},checkpoint)
    binding['sha256']=deps.sha256_file_hex_cached(checkpoint)
    with pytest.raises(ValueError,match='selected foundation recovery lacks'):
        deps.dependency_closure({'parent_checkpoint':binding},namespace=str(tmp_path))
    training['options']['foundation_declaration_recovery_binding']=proof
    torch.save({'training':training},checkpoint)
    Path(docs[1]['path']).write_text('tampered')
    binding['sha256']=deps.sha256_file_hex_cached(checkpoint)
    with pytest.raises(ValueError,match='foundation recovery artifact changed'):
        deps.dependency_closure({'parent_checkpoint':binding},namespace=str(tmp_path))


def test_unselected_stale_foundation_bindings_are_not_runtime_dependencies(tmp_path):
    import torch
    absent={'path':str(tmp_path/'obsolete-proof.json'),'sha256':'a'*64,'bytes':1}
    checkpoint=tmp_path/'state.pt'
    torch.save({'training':{'options':{'communication_foundation':False,
        'foundation_native_graphs':None,'foundation_declaration_recovery':None,
        'foundation_native_graphs_binding':absent,'foundation_declaration_recovery_binding':absent}}},checkpoint)
    closure=deps.dependency_closure({'parent_checkpoint':{'path':str(checkpoint),
        'sha256':deps.sha256_file_hex_cached(checkpoint)}},namespace=str(tmp_path))
    assert {item['path']for item in closure['files']}=={str(checkpoint)}
