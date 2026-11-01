import hashlib,json,os,tempfile
from pathlib import Path
import pytest
from natlang_neuralese.train.checkpoint_offload import RecoveryExportQueue
from scripts.coreweave.retention_leases import change_lease
from natlang_neuralese.common.jsonio import write_json_durable


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def attach_qualification(plan,path):
    from natlang_neuralese.train.checkpoint_offload import recovery_closure_identity
    plan.setdefault('require_static_contract',True);plan.setdefault('sft_static_contract',{})
    if 'control_snapshot' not in plan:
        plan['control_snapshot']=[]
        for name in ('build_local_recovery_plan.py','qualify_local_checkpoint_recovery.py','verify_local_recovery_export.py'):
            helper=path.parent/('snapshot-'+name);helper.write_bytes(name.encode())
            ref={'path':str(helper),'sha256':sha(helper),'source_path':'/test/'+name,'source_commit':'0'*40}
            plan['control_snapshot'].append(ref);plan['closure'].append({'role':'controls','path':ref['path'],'sha256':ref['sha256']})
    old=plan.get('restore_qualification')
    if old:plan['closure']=[x for x in plan['closure'] if not (x['role']=='controls' and x['path']==old['path'] and x['sha256']==old['sha256'])]
    evidence_path=path.with_name(path.stem+'-evidence.json')
    if not evidence_path.exists():write_json_durable(evidence_path,{'test':'simulated qualification evidence'})
    evidence_ref={'path':str(evidence_path),'sha256':sha(evidence_path)}
    if not any(x['path']==str(evidence_path) for x in plan['closure']):plan['closure'].append({'role':'controls',**evidence_ref})
    plan['restore_qualification']={'path':str(path),'sha256':'0'*64}
    checks={key:True for key in ('all_payloads_reread','source_closure_resolved','environment_imports','tokenizer_loaded','checkpoint_full_state_loaded','checkpoint_plan_custody','node_collector_imported','task_assets_read')}
    write_json_durable(path,{'schema':'natlang.local-recovery-restore-qualification/1','status':'passed','checkpoint':{'sha256':'a'*64,'training_custody':{'source_roots':[plan['inputs']['train'],plan['inputs']['held']],'ce':plan['inputs']['ce']}},'inputs':plan['inputs'],'require_static_contract':plan.get('require_static_contract'),'sft_static_contract':plan.get('sft_static_contract'),'control_snapshot':plan.get('control_snapshot'),'restore':plan.get('restore'),'additional_inputs':plan.get('additional_inputs',[]),'controller_config':plan.get('controller',{}).get('config'),'closure_identity':recovery_closure_identity(plan),'checks':checks,'evidence':{key:evidence_ref for key in checks}})
    plan['restore_qualification']['sha256']=sha(path)
    plan['closure'].append({'role':'controls',**plan['restore_qualification']})
    return plan


@pytest.fixture
def setup(tmp_path):
    with tempfile.TemporaryDirectory(dir='/dev/shm') as destination:
        slot=tmp_path/'checkpoint.pt';slot.write_bytes(b'full state 1')
        refs=[]
        for role in ('runtime','environment','system-python','system-libraries','node','interpreter','base','tokenizer','task-assets','controls','sources'):
            p=tmp_path/role;p.write_bytes(role.encode());refs.append({'role':role,'path':str(p),'sha256':sha(p)})
        inputs={role:{'path':str(tmp_path/role),'sha256':sha(tmp_path/role)} for role in ('runtime','environment','system-python','system-libraries','node','interpreter','base','tokenizer','task-assets','controls','sources')}
        bindings={'train':inputs['sources'],'foundation':inputs['sources'],'held':inputs['runtime'],'ce':inputs['controls']}
        controller=tmp_path/'controller.json';controller.write_text(json.dumps({'phase':'updating','checkpoint':str(slot),'round':0,'offset':0,'trained_rounds':0}))
        plan={'controller':{'state_path':str(controller),'config':inputs['controls']},'inputs':bindings,'schema':'natlang.local-full-recovery-plan/1','session':'s1','destination':destination,'journal_directory':str(tmp_path/'journal'),'closure':refs}
        plan['restore']={'runtime_archive':str(tmp_path/'runtime'),'environment_archive':str(tmp_path/'environment'),'system_python_archive':str(tmp_path/'system-python'),'base_path':str(tmp_path),'node_path':str(tmp_path/'node'),'interpreter_path':str(tmp_path/'interpreter'),'sample_graph':str(tmp_path/'sources'),'tokenizer_fingerprint':'a'*64,'asset_guard':str(tmp_path/'controls'),'asset_inventory':str(tmp_path/'controls'),'asset_archives':[{'env':'NATLANG_RL_TASK_ASSETS','original_root':str(tmp_path/'asset1'),'archive':str(tmp_path/'task-assets')},{'env':'NATLANG_RL_TASK_ASSETS_V3','original_root':str(tmp_path/'asset2'),'archive':str(tmp_path/'task-assets')}]}
        attach_qualification(plan,tmp_path/'qualification.json')
        yield slot,plan


def submit(q,slot,step):q.submit(slot,{'training':{'optimizer':{'x':1},'rng':{'x':1},'global_step':step,'source_receipts':[q.inputs['train'],q.inputs['held']],'options':{'ce_supervision_policy_binding':q.inputs['ce'],'native_graphs':q.inputs['train']['path'],'foundation_native_graphs':q.inputs['foundation']['path'] if q.inputs['foundation'] else None,'held_native_graphs':q.inputs['held']['path']}}},{'previous_preserved_until_replace':True})


def test_coalesced_pins_recovery_and_verified_copy(setup):
    slot,plan=setup;q=RecoveryExportQueue(slot,plan,start_worker=False,plan_validator=lambda plan:None,checkpoint_validator=lambda path,custody:None);submit(q,slot,1)
    first=q._stat(q.pin);next_slot=slot.with_name('next');next_slot.write_bytes(b'full state 2');os.replace(next_slot,slot);submit(q,slot,2)
    assert q._stat(q.pin)==first and q.state['candidate']['global_step']==2
    q.close();q=RecoveryExportQueue(slot,plan,start_worker=False,plan_validator=lambda plan:None,checkpoint_validator=lambda path,custody:None)
    assert q._export_one() and q.state['inflight']['global_step']==2
    assert q._export_one() and q.state['inflight'] is None and q.state['latest_verified']['global_step']==2
    assert not q.pin.exists()
    r=json.loads(Path(q.state['latest_verified']['path']).read_text());assert r['full_state'] and r['independent_readback'] and Path(r['checkpoint']['path']).read_bytes()==b'full state 2'
    assert len(r['closure'])==len(plan['closure'])+1
    q.close()


def test_lease_blocks_retirement_and_recovers(setup):
    slot,plan=setup;q=RecoveryExportQueue(slot,plan,start_worker=False,plan_validator=lambda plan:None,checkpoint_validator=lambda path,custody:None);submit(q,slot,1);r=q.state['inflight'];d=q.directory;role=f"local-recovery-pin:{plan['session']}:{r['global_step']}:{r['identity']['device']}:{r['identity']['inode']}";args={'lock_path':d/'standalone-cloud-retention.lock','state_path':d/'standalone-retention-consumer-leases.json','retirement_state_path':d/'local-recovery-retirement.json','role':role,'consumer_id':'test','identity':{'id':'one'},'write_durable':write_json_durable}
    change_lease(**args,release=False)
    with pytest.raises(RuntimeError,match='lease'):q._export_one()
    assert q.pin.exists() and q.state['inflight']
    change_lease(**args,release=True);assert q._export_one();assert not q.pin.exists()
    with pytest.raises(ValueError,match='retirement'):change_lease(**args,release=False)
    q.close()


def test_reject_changed_plan_and_incomplete_state(setup):
    slot,plan=setup;q=RecoveryExportQueue(slot,plan,start_worker=False,plan_validator=lambda plan:None,checkpoint_validator=lambda path,custody:None)
    with pytest.raises(ValueError,match='optimizer'):q.submit(slot,{'training':{'global_step':1}},{'previous_preserved_until_replace':True})
    q.close()
    with pytest.raises(ValueError,match='another session'):RecoveryExportQueue(slot,plan|{'session':'other'},start_worker=False,plan_validator=lambda plan:None,checkpoint_validator=lambda path,custody:None)


def test_inflight_keeps_old_closure_when_source_plan_changes(setup):
    slot,plan=setup;q=RecoveryExportQueue(slot,plan,start_worker=False,plan_validator=lambda plan:None,checkpoint_validator=lambda path,custody:None);submit(q,slot,1)
    old_hash=q.state['inflight']['plan_sha256'];new_source=slot.parent/'newsource';new_source.write_bytes(b'new source')
    next_plan=json.loads(json.dumps(plan));new_ref={'role':'sources','path':str(new_source),'sha256':sha(new_source)}
    next_plan['closure']=[x for x in next_plan['closure'] if x['role']!='sources']+[new_ref]
    next_plan['inputs']['train']={'path':str(new_source),'sha256':sha(new_source)};next_plan['inputs']['foundation']=next_plan['inputs']['train'];next_plan['restore']['sample_graph']=str(new_source)
    q.update_plan(next_plan);replacement=slot.with_name('replacement');replacement.write_bytes(b'full state 2');os.replace(replacement,slot);submit(q,slot,2)
    assert q._export_one();old_receipt=json.loads(Path(q.state['latest_verified']['path']).read_text());assert old_receipt['plan_sha256']==old_hash
    assert q._export_one();new_receipt=json.loads(Path(q.state['latest_verified']['path']).read_text());assert new_receipt['inputs']['train']['sha256']==sha(new_source)
    q.close()


def test_orphan_link_after_crash_is_reconciled_by_exact_candidate(setup):
    slot,plan=setup;q=RecoveryExportQueue(slot,plan,start_worker=False,plan_validator=lambda plan:None,checkpoint_validator=lambda path,custody:None)
    q.state['candidate']={'identity':q._stat(slot),'global_step':1,'plan':plan,'plan_sha256':q._plan_digest(plan),'custody':{},'checkpoint_sha256':sha(slot),'controller_state':q._capture_controller()};q._persist();os.link(slot,q.pin);q.close()
    q=RecoveryExportQueue(slot,plan,start_worker=False,plan_validator=lambda plan:None,checkpoint_validator=lambda path,custody:None);assert q.state['inflight']['global_step']==1 and q.state['candidate'] is None
    assert q._export_one();q.close()


def test_completed_unlink_crash_recovers_without_duplicate_pin(setup):
    slot,plan=setup;q=RecoveryExportQueue(slot,plan,start_worker=False,plan_validator=lambda plan:None,checkpoint_validator=lambda path,custody:None);submit(q,slot,1)
    q.state['inflight']['verified_receipt']={'path':str(slot),'sha256':sha(slot),'checkpoint_sha256':'a'*64};q._persist();q.pin.unlink();q.close()
    with pytest.raises(ValueError):RecoveryExportQueue(slot,plan,start_worker=False,plan_validator=lambda plan:None,checkpoint_validator=lambda path,custody:None)


def test_real_dependency_contract_rejects_missing_transitive_file(setup,monkeypatch):
    from natlang_neuralese.train import recipe,native_supervision
    from natlang_neuralese.train.checkpoint_offload import validate_recovery_plan_dependencies
    slot,plan=setup;missing=slot.parent/'transitive';missing.write_bytes(b'metadata')
    monkeypatch.setattr(recipe,'native_source_dependencies',lambda path:[{'path':str(missing),'sha256':sha(missing)}])
    monkeypatch.setattr(native_supervision,'ce_policy_dependencies',lambda ref:[ref])
    with pytest.raises(ValueError,match='transitive'):validate_recovery_plan_dependencies(plan)
    plan['closure'].append({'role':'sources','path':str(missing),'sha256':sha(missing)})
    attach_qualification(plan,slot.parent/'qualification.json')
    validate_recovery_plan_dependencies(plan)


def test_checkpoint_must_match_qualified_source(setup):
    slot,plan=setup;q=RecoveryExportQueue(slot,plan,start_worker=False,plan_validator=lambda plan:None,checkpoint_validator=lambda path,custody:None)
    state={'training':{'optimizer':{'x':1},'rng':{'x':1},'global_step':1,'source_receipts':[q.inputs['train']|{'sources':[{'path':'/omitted','sha256':'a'*64}]},q.inputs['held']],'options':{'ce_supervision_policy_binding':q.inputs['ce'],'native_graphs':q.inputs['train']['path'],'foundation_native_graphs':q.inputs['foundation']['path'] if q.inputs['foundation'] else None,'held_native_graphs':q.inputs['held']['path']}}}
    with pytest.raises(ValueError,match='closure omits'):q.submit(slot,state,{'previous_preserved_until_replace':True})
    assert not q.pin.exists();q.close()


def test_verified_receipt_unlink_crash_is_recovered(setup):
    from natlang_neuralese.train.checkpoint_offload import verified_copy
    slot,plan=setup;q=RecoveryExportQueue(slot,plan,start_worker=False,plan_validator=lambda plan:None,checkpoint_validator=lambda path,custody:None);submit(q,slot,1);r=q.state['inflight'];target=Path(plan['destination'])/'state.pt';copy=verified_copy(q.pin,target)
    receipt={'schema':'natlang.local-full-recovery-export/1','session':plan['session'],'global_step':1,'source_identity':r['identity'],'plan_sha256':r['plan_sha256'],'plan':r['plan'],'training_custody':r['custody'],'controller_state':r['controller_state'],'checkpoint':{'path':str(target),'sha256':copy['sha256']},'full_state':True,'independent_readback':True,'closure':q._copy_closure(r['plan'],r['controller_state'])}
    path=Path(plan['destination'])/'receipt.json';write_json_durable(path,receipt);r['verified_receipt']={'path':str(path),'sha256':sha(path),'checkpoint_sha256':copy['sha256']};q._persist();q.pin.unlink();q.close()
    q=RecoveryExportQueue(slot,plan,start_worker=False,plan_validator=lambda plan:None,checkpoint_validator=lambda path,custody:None);assert q.state['inflight'] is None;q.close()


def test_same_step_saved_again_coalesces_but_regression_is_rejected(setup):
    slot,plan=setup;q=RecoveryExportQueue(slot,plan,start_worker=False,plan_validator=lambda plan:None,checkpoint_validator=lambda path,custody:None);submit(q,slot,1)
    next_slot=slot.with_name('same');next_slot.write_bytes(b'resaved');os.replace(next_slot,slot);submit(q,slot,1)
    assert q.state['candidate']['global_step']==1
    with pytest.raises(ValueError,match='advance'):submit(q,slot,0)
    assert q._export_one() and q._export_one();q.close()


def test_real_graph_and_ce_closure_validation(tmp_path):
    from natlang_neuralese.train.checkpoint_offload import validate_recovery_plan_dependencies
    leaf=tmp_path/'leaf.jsonl';leaf.write_text('{}\n')
    manifest=tmp_path/'train.json';manifest.write_text(json.dumps({'schema':'natlang.native-graph-sources/1','sources':[{'kind':'native-graphs','path':str(leaf),'sha256':sha(leaf)}]}))
    held=tmp_path/'held.json';held.write_text(manifest.read_text())
    ce=tmp_path/'ce.json';ce.write_text(json.dumps({'schema':'natlang.native-ce-supervision-policy/1','sources':[{'sha256':sha(leaf),'context_weight':0.1,'encoding':'native-row-objectives-v1'}]}))
    inputs={'train':{'path':str(manifest),'sha256':sha(manifest)},'foundation':{'path':str(manifest),'sha256':sha(manifest)},'held':{'path':str(held),'sha256':sha(held)},'ce':{'path':str(ce),'sha256':sha(ce)}}
    plan={'inputs':inputs,'controller':{'config':inputs['ce']},'closure':[{'role':'sources',**ref} for ref in inputs.values()]}
    with pytest.raises(ValueError,match='transitive'):validate_recovery_plan_dependencies(plan)
    plan['closure'].append({'role':'sources','path':str(leaf),'sha256':sha(leaf)})
    attach_qualification(plan,tmp_path/'qualification.json')
    validate_recovery_plan_dependencies(plan)


def test_policy_automatically_submits_only_after_commit(tmp_path,monkeypatch):
    from natlang_neuralese.train import checkpoint_offload
    from natlang_neuralese.train.checkpoint_policy import CheckpointPolicy
    class Recorder:
        def __init__(self):self.seen=[];self.state={'latest_verified':None}
        def submit(self,path,state,event):
            assert Path(path).is_file() and event['previous_preserved_until_replace']
            self.seen.append((Path(path).read_bytes(),state['training']['global_step']))
    record=Recorder();(tmp_path/'checkpoint-recovery-plan.json').write_text('{}')
    monkeypatch.setattr(checkpoint_offload,'recovery_queue_for',lambda slot,plan,state=None:record)
    monkeypatch.setattr(checkpoint_offload,'_read_recovery_selection',lambda plan,state=None:{})
    policy=CheckpointPolicy(tmp_path);policy.save({'training':{'global_step':3,'optimizer':{'x':1},'rng':{'x':1}}});policy.close()
    assert len(record.seen)==1 and record.seen[0][1]==3 and record.seen[0][0]


def test_policy_required_plan_fails_closed(tmp_path,monkeypatch):
    from natlang_neuralese.train.checkpoint_policy import CheckpointPolicy
    monkeypatch.setenv('NATLANG_REQUIRE_FULL_RECOVERY','1')
    with pytest.raises(ValueError,match='plan is missing'):CheckpointPolicy(tmp_path)


def test_independent_checkpoint_decodes_full_exposure_and_detects_mismatch(setup):
    import torch
    from natlang_neuralese.train.checkpoint_offload import recovery_training_custody,validate_full_recovery_checkpoint
    slot,plan=setup;q=RecoveryExportQueue(slot,plan,start_worker=False,plan_validator=lambda plan:None)
    state={'training':{'global_step':3,'optimizer':{'slots':{'w':torch.tensor([1.0])}},'rng':{'seed':1},'source_row_bindings':{'row':'binding'},'held_row_bindings':{},'exposure_counts':{'row':2},'sft_progress':{},'source_receipts':[q.inputs['train'],q.inputs['held']],'options':{'ce_supervision_policy_binding':q.inputs['ce'],'native_graphs':q.inputs['train']['path'],'foundation_native_graphs':q.inputs['foundation']['path'] if q.inputs['foundation'] else None,'held_native_graphs':q.inputs['held']['path']}}}
    torch.save(state,slot);q.submit(slot,state,{'previous_preserved_until_replace':True});assert q._export_one()
    receipt=json.loads(Path(q.state['latest_verified']['path']).read_text());assert receipt['training_custody']==recovery_training_custody(state)
    changed=recovery_training_custody(state);changed['global_step']=4
    with pytest.raises(ValueError,match='custody differs'):validate_full_recovery_checkpoint(receipt['checkpoint']['path'],changed)
    q.close()


def test_changed_closure_blocks_pin_retirement(setup):
    slot,plan=setup
    def mutate(path,custody):
        target=Path(plan['destination'])/'closure'/plan['closure'][0]['sha256'];target.write_bytes(b'changed')
    q=RecoveryExportQueue(slot,plan,start_worker=False,plan_validator=lambda plan:None,checkpoint_validator=mutate);submit(q,slot,1)
    with pytest.raises(RuntimeError,match='dependency changed'):q._export_one()
    assert q.pin.exists() and q.state['inflight'] is not None;q.close()


def test_reject_session_path_escape(setup):
    slot,plan=setup
    with pytest.raises(ValueError,match='session identity'):RecoveryExportQueue(slot,plan|{'session':'../bad'},start_worker=False,plan_validator=lambda plan:None)


def test_real_policy_hook_exports_full_checkpoint_and_closure(tmp_path,monkeypatch):
    import time,torch
    from natlang_neuralese.train import checkpoint_offload
    from natlang_neuralese.train.checkpoint_policy import CheckpointPolicy
    with tempfile.TemporaryDirectory(dir='/dev/shm') as destination:
        out=tmp_path/'learner';out.mkdir();leaf=tmp_path/'leaf.jsonl';leaf.write_text('{}\n')
        manifest=tmp_path/'source.json';manifest.write_text(json.dumps({'schema':'natlang.native-graph-sources/1','sources':[{'kind':'native-graphs','path':str(leaf),'sha256':sha(leaf)}]}))
        held=tmp_path/'held.json';held.write_text(manifest.read_text())
        ce=tmp_path/'ce.json';ce.write_text(json.dumps({'schema':'natlang.native-ce-supervision-policy/1','sources':[{'sha256':sha(leaf),'context_weight':0.1,'encoding':'native-row-objectives-v1'}]}))
        runtime=tmp_path/'runtime';runtime.write_text('runtime')
        bind=lambda p:{'path':str(p),'sha256':sha(p)}
        inputs={'train':bind(manifest),'foundation':bind(manifest),'held':bind(held),'ce':bind(ce)}
        controller=tmp_path/'controller.json';controller.write_text(json.dumps({'phase':'updating','checkpoint':str(out/'checkpoint.pt'),'round':0,'offset':0,'trained_rounds':0}))
        plan={'controller':{'state_path':str(controller),'config':bind(ce)},'schema':'natlang.local-full-recovery-plan/1','session':'integration','destination':destination,'journal_directory':str(out/'recovery'),'inputs':inputs,'closure':[{'role':role,**bind(path)} for role,path in [('sources',manifest),('sources',held),('sources',leaf),('controls',ce),('runtime',runtime),('environment',runtime),('system-python',runtime),('system-libraries',runtime),('node',runtime),('interpreter',runtime),('base',runtime),('tokenizer',runtime),('task-assets',runtime)]]}
        plan['restore']={'runtime_archive':str(runtime),'environment_archive':str(runtime),'system_python_archive':str(runtime),'base_path':str(tmp_path),'node_path':str(runtime),'interpreter_path':str(runtime),'sample_graph':str(leaf),'tokenizer_fingerprint':'a'*64,'asset_guard':str(ce),'asset_inventory':str(ce),'asset_archives':[{'env':'NATLANG_RL_TASK_ASSETS','original_root':str(tmp_path/'asset1'),'archive':str(runtime)},{'env':'NATLANG_RL_TASK_ASSETS_V3','original_root':str(tmp_path/'asset2'),'archive':str(runtime)}]}
        attach_qualification(plan,tmp_path/'qualification.json')
        plan_path=out/'immutable-plan.json';write_json_durable(plan_path,plan);plan_ref={'path':str(plan_path),'sha256':sha(plan_path)}
        write_json_durable(out/'checkpoint-recovery-plan.json',{'schema':'natlang.local-full-recovery-catalog/1','controller_config':bind(ce),'initial_plan':plan_ref,'plans':[plan_ref]})
        monkeypatch.setenv('NATLANG_REQUIRE_FULL_RECOVERY','1');monkeypatch.setenv('NATLANG_RECOVERY_CONTROL_SHA',sha(ce))
        state={'training':{'global_step':3,'optimizer':{'slots':{'w':torch.tensor([1.0])}},'rng':{'seed':1},'source_row_bindings':{'row':'binding'},'held_row_bindings':{},'exposure_counts':{'row':2},'sft_progress':{},'source_receipts':[inputs['train']|{'sources':[bind(leaf)]},inputs['held']|{'sources':[bind(leaf)]}],'options':{'ce_supervision_policy_binding':inputs['ce'],'native_graphs':inputs['train']['path'],'foundation_native_graphs':inputs['foundation']['path'],'held_native_graphs':inputs['held']['path']}}}
        policy=CheckpointPolicy(out);policy.save(state);queue=policy.recovery_queue
        deadline=time.monotonic()+30
        while (not queue.state.get('latest_verified') or queue.state.get('inflight')) and time.monotonic()<deadline:time.sleep(.05)
        assert queue.state.get('latest_verified'),queue.state.get('last_error')
        receipt=json.loads(Path(queue.state['latest_verified']['path']).read_text());assert receipt['training_custody']['global_step']==3 and receipt['independent_readback']
        assert not queue.pin.exists();policy.close();queue.close()
        checkpoint_offload._RECOVERY_QUEUES.pop(str(policy.slot.resolve()))


def test_missing_latest_is_not_reported_as_verified_on_restart(setup):
    slot,plan=setup;q=RecoveryExportQueue(slot,plan,start_worker=False,plan_validator=lambda plan:None,checkpoint_validator=lambda path,custody:None);submit(q,slot,1);assert q._export_one()
    receipt=json.loads(Path(q.state['latest_verified']['path']).read_text());Path(receipt['closure'][0]['path']).unlink();q.close()
    q=RecoveryExportQueue(slot,plan,start_worker=False,plan_validator=lambda plan:None,checkpoint_validator=lambda path,custody:None)
    assert q.state['latest_verified_status']=='unavailable' and q.state['latest_verified'] is not None;q.close()


def test_strong_policy_never_uses_legacy_mirror_when_slot_missing(tmp_path,monkeypatch):
    from natlang_neuralese.train import checkpoint_offload,checkpoint_policy
    class Queue:state={'latest_verified':{'global_step':3}}
    (tmp_path/'checkpoint-recovery-plan.json').write_text('{}')
    monkeypatch.setattr(checkpoint_offload,'recovery_queue_for',lambda slot,plan:Queue())
    monkeypatch.setattr(checkpoint_policy,'recover_latest',lambda slot:pytest.fail('legacy mirror must not be selected'))
    with pytest.raises(ValueError,match='restore a verified full-state export'):checkpoint_policy.CheckpointPolicy(tmp_path)


def test_rl_leaf_keeps_carried_sft_source_and_ce_custody(setup):
    slot,plan=setup;old=plan['inputs']['train'];held=plan['inputs']['held'];ce=plan['inputs']['ce']
    rl=slot.parent/'rollout.jsonl';rl.write_text('{}\n');rlref={'path':str(rl),'sha256':sha(rl)}
    plan['inputs']['train']=rlref;plan['inputs']['foundation']=None;plan['closure'].append({'role':'sources',**rlref})
    plan['additional_inputs']=[{'kind':'source',**old}]
    frame={'source_receipts':[old,held], 'options':{'foundation_native_graphs':old['path'],'ce_supervision_policy_binding':ce},'source_row_bindings':{'old':'binding'},'exposure_counts':{'old':2}}
    state={'training':{'global_step':5,'optimizer':{'x':1},'rng':{'x':1},'source_receipts':[rlref,held],
                       'options':{'native_graphs':rlref['path'],'foundation_native_graphs':None,'held_native_graphs':held['path'],'ce_supervision_policy_binding':ce},'sft_progress':{'joint':frame}}}
    from natlang_neuralese.train.checkpoint_offload import recovery_static_sft_contract
    plan['sft_static_contract']=recovery_static_sft_contract(state['training'])
    attach_qualification(plan,slot.parent/'qualification.json')
    q=RecoveryExportQueue(slot,plan,start_worker=False,plan_validator=lambda plan:None,checkpoint_validator=lambda path,custody:None)
    q.submit(slot,state,{'previous_preserved_until_replace':True});assert q.state['inflight']['custody']['sft_frames_sha256']['joint']
    assert q._export_one();q.close()
    plan['additional_inputs']=[];attach_qualification(plan,slot.parent/'qualification.json')
    plan['journal_directory']=str(slot.parent/'other-journal');q=RecoveryExportQueue(slot,plan,start_worker=False,plan_validator=lambda plan:None,checkpoint_validator=lambda path,custody:None)
    with pytest.raises(ValueError,match='carried SFT source'):q.submit(slot,state,{'previous_preserved_until_replace':True})
    q.close()


def test_restore_qualification_rejects_changed_restore_contract(setup):
    from natlang_neuralese.train.checkpoint_offload import validate_restore_qualification
    _,plan=setup;validate_restore_qualification(plan)
    plan['restore']['base_path']='/changed/base'
    with pytest.raises(ValueError,match='complete closure'):validate_restore_qualification(plan)


def test_restore_qualification_rejects_checkpoint_from_another_source(setup):
    from natlang_neuralese.train.checkpoint_offload import validate_restore_qualification
    _,plan=setup;ref=plan['restore_qualification'];path=Path(ref['path']);proof=json.loads(path.read_text())
    proof['checkpoint']['training_custody']['source_roots'][0]['sha256']='b'*64
    write_json_durable(path,proof);old=ref['sha256'];ref['sha256']=sha(path)
    for item in plan['closure']:
        if item['path']==str(path) and item['sha256']==old:item['sha256']=ref['sha256']
    with pytest.raises(ValueError,match='complete closure'):validate_restore_qualification(plan)


def test_catalog_selects_actual_saved_source_without_early_promotion(setup,monkeypatch):
    from natlang_neuralese.train.checkpoint_offload import _read_recovery_selection
    slot,old=setup;oldpath=slot.parent/'old-plan.json';write_json_durable(oldpath,old);oldref={'path':str(oldpath),'sha256':sha(oldpath)}
    new=json.loads(json.dumps(old));source=slot.parent/'new';source.write_bytes(b'new');ref={'path':str(source),'sha256':sha(source)}
    new['inputs']['train']=new['inputs']['foundation']=ref;new['closure'].append({'role':'sources',**ref});new['restore']['sample_graph']=str(source);attach_qualification(new,slot.parent/'new-qualification.json')
    newpath=slot.parent/'new-plan.json';write_json_durable(newpath,new);newref={'path':str(newpath),'sha256':sha(newpath)}
    control=old['controller']['config'];catalog=slot.parent/'catalog.json';write_json_durable(catalog,{'schema':'natlang.local-full-recovery-catalog/1','controller_config':control,'initial_plan':oldref,'plans':[oldref,newref]})
    monkeypatch.setenv('NATLANG_REQUIRE_FULL_RECOVERY','1');monkeypatch.setenv('NATLANG_RECOVERY_CONTROL_SHA',control['sha256'])
    def state(plan):
        i=plan['inputs'];return {'training':{'source_receipts':[i['train'],i['held']],'options':{'native_graphs':i['train']['path'],'foundation_native_graphs':i['foundation']['path'],'held_native_graphs':i['held']['path'],'ce_supervision_policy_binding':i['ce']}}}
    assert _read_recovery_selection(catalog)['inputs']==old['inputs']
    assert _read_recovery_selection(catalog,state(old))['inputs']==old['inputs']
    assert _read_recovery_selection(catalog,state(new))['inputs']==new['inputs']
    bad=state(new);bad['training']['source_receipts'][0]['sha256']='a'*64
    with pytest.raises(ValueError,match='no unique'): _read_recovery_selection(catalog,bad)


def test_resident_candidate_requires_qualified_matching_plan(setup, monkeypatch):
    from natlang_neuralese.train import checkpoint_offload as offload
    slot, plan = setup
    path = slot.parent / 'catalog.json'
    write_json_durable(path, plan)
    checked = []
    monkeypatch.setattr(offload, 'validate_recovery_plan_dependencies', lambda value: checked.append(value))
    assert offload.require_recovery_candidate(path, plan['inputs']) == plan['restore_qualification']
    assert checked == [plan]
    inputs = dict(plan['inputs'], train={'path': '/unqualified', 'sha256': 'a' * 64})
    with pytest.raises(ValueError, match='no unique staged recovery plan'):
        offload.require_recovery_candidate(path, inputs)
    assert checked == [plan]


def test_copy_child_inherits_blob_and_owner_locks(tmp_path, monkeypatch):
    import fcntl
    from natlang_neuralese.train import checkpoint_offload as offload
    source = tmp_path / 'source'; source.write_bytes(b'complete recovery bytes')
    target = tmp_path / 'blobs' / sha(source)
    owner = (tmp_path / 'owner.lock').open('w')
    fcntl.flock(owner, fcntl.LOCK_EX)
    calls = []
    real = offload.subprocess.Popen
    def checked(*args, **kwargs):
        descriptors = kwargs['pass_fds']
        assert owner.fileno() in descriptors and len(descriptors) == 2
        calls.append(descriptors)
        return real(*args, **kwargs)
    monkeypatch.setattr(offload.subprocess, 'Popen', checked)
    offload.verified_recovery_blob(source, target, sha(source), owner_fd=owner.fileno())
    assert target.read_bytes() == source.read_bytes() and len(calls) == 1
    owner.close()


def test_inherited_owner_lock_survives_parent_close(tmp_path):
    import fcntl, subprocess, sys
    path = tmp_path / 'owner.lock'
    owner = path.open('w')
    fcntl.flock(owner, fcntl.LOCK_EX)
    child = subprocess.Popen([sys.executable, '-c', 'import sys; sys.stdin.read()'],
                             stdin=subprocess.PIPE, pass_fds=(owner.fileno(),))
    try:
        owner.close()
        with path.open('w') as contender:
            with pytest.raises(BlockingIOError):
                fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
        child.stdin.close()
        assert child.wait(timeout=5) == 0
        with path.open('w') as contender:
            fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        if child.poll() is None:
            child.kill(); child.wait()


def test_export_verifier_uses_only_independent_bytes(setup, monkeypatch):
    from scripts.neuralese_data import verify_local_recovery_export as verifier
    slot, plan = setup
    q=RecoveryExportQueue(slot,plan,start_worker=False,plan_validator=lambda plan:None,
                          checkpoint_validator=lambda path,custody:None)
    submit(q,slot,1); assert q._export_one()
    ref=q.state['latest_verified']; q.close()
    for item in plan['closure']:
        Path(item['path']).unlink(missing_ok=True)
    monkeypatch.setattr(verifier,'validate_full_recovery_checkpoint',lambda path,custody:{})
    monkeypatch.setattr(verifier,'_plan_matches_checkpoint',lambda plan,state:True)
    result=verifier.verify(ref['path'],ref['sha256'])
    assert result['global_step']==1
    receipt=json.loads(Path(ref['path']).read_text()); Path(receipt['closure'][0]['path']).write_bytes(b'tampered')
    with pytest.raises(ValueError,match='closure bytes'):
        verifier.verify(ref['path'],ref['sha256'])


def test_checkpoint_copy_uses_low_best_effort_io(monkeypatch):
    from natlang_neuralese.train import checkpoint_offload as offload
    calls=[]
    monkeypatch.setattr(offload.platform,'machine',lambda:'aarch64')
    monkeypatch.setattr(offload.os,'setpriority',lambda *args:calls.append(('nice',args)))
    class Lib:
        def syscall(self,*args):calls.append(('ioprio',args));return 0
    monkeypatch.setattr(offload.ctypes,'CDLL',lambda *args,**kwargs:Lib())
    offload._lower_priority(123)
    assert calls==[('nice',(offload.os.PRIO_PROCESS,123,19)),('ioprio',(30,1,123,(2<<13)|7))]


def test_static_recovery_contract_allows_earned_progress_but_rejects_quota_change():
    from copy import deepcopy
    from natlang_neuralese.train.checkpoint_offload import recovery_static_sft_contract
    frame={'source_row_bindings':{'row':'binding'},'source_row_order':['row'],
           'held_row_bindings':{},'required_visits':{'row':7},'source_hash':'source',
           'identity':{'stage':'joint'},'source_receipts':[{'path':'/train','sha256':'a'*64}],
           'options':{'epochs':1,'_required_visits':{'row':7}},'exposure_counts':{'row':1},
           'rng':{'seed':1},'optimizer':{'momentum':1},'cursor':1,'global_step':68}
    training={'sft_progress':{'joint':frame}}
    original=recovery_static_sft_contract(training)
    advanced=deepcopy(training);other=advanced['sft_progress']['joint']
    other.update(exposure_counts={'row':2},rng={'seed':2},optimizer={'momentum':2},cursor=2,global_step=69)
    assert recovery_static_sft_contract(advanced)==original
    other['required_visits']['row']=8
    assert recovery_static_sft_contract(advanced)!=original


def test_recovery_plan_cannot_omit_static_contract(setup):
    from natlang_neuralese.train.checkpoint_offload import validate_recovery_plan_shape
    _,plan=setup;plan.pop('sft_static_contract')
    with pytest.raises(ValueError,match='static SFT contract'):
        validate_recovery_plan_shape(plan)


def test_direct_queue_submit_rejects_changed_static_quota(setup):
    from natlang_neuralese.train.checkpoint_offload import recovery_static_sft_contract
    slot,plan=setup
    frame={'source_row_bindings':{},'required_visits':{'row':1},'options':{},'source_receipts':[]}
    state={'training':{'global_step':4,'optimizer':{'m':1},'rng':{'r':1},
        'source_receipts':[plan['inputs']['train'],plan['inputs']['held']],
        'options':{'native_graphs':plan['inputs']['train']['path'],
                   'foundation_native_graphs':plan['inputs']['foundation']['path'],
                   'held_native_graphs':plan['inputs']['held']['path'],
                   'ce_supervision_policy_binding':plan['inputs']['ce']},'sft_progress':{'joint':frame}}}
    plan['sft_static_contract']=recovery_static_sft_contract(state['training']);attach_qualification(plan,slot.parent/'qualification.json')
    q=RecoveryExportQueue(slot,plan,start_worker=False,plan_validator=lambda plan:None,checkpoint_validator=lambda path,custody:None)
    frame['required_visits']['row']=2
    with pytest.raises(ValueError,match='static SFT contract'):
        q.submit(slot,state,{'previous_preserved_until_replace':True})
    assert not q.pin.exists();q.close()


def test_prospective_append_intent_binds_owner_sources_and_execution(setup):
    from copy import deepcopy
    from natlang_neuralese.train.native_recovery_append import validate_append_intent
    _,old=setup;new=deepcopy(old)
    source=Path(old['inputs']['train']['path']).parent/'appended';source.write_bytes(b'appended')
    ref={'path':str(source),'sha256':sha(source)}
    new['inputs']['train']=ref;new['inputs']['foundation']=ref
    new['closure'].append({'role':'sources',**ref})
    new['additional_inputs']=[{'kind':'source',**old['inputs']['train']}]
    roles={'native_graphs':ref,'foundation_native_graphs':ref,'held_native_graphs':old['inputs']['held'],
           'ce_supervision_policy':old['inputs']['ce']}
    payload={'stage_id':'joint','predecessor':'old-generation','prior_train':old['inputs']['train'],
             'prior_held':old['inputs']['held'],'train':ref,'held':old['inputs']['held'],
             'plan':{'stage_inputs':{'joint':roles}}}
    options={'stage_id':'joint','predecessor_generation':'old-generation',
             'predecessor_validator':lambda p:None,'successor_validator':lambda p:None}
    intent=validate_append_intent(old,new,payload,**options)
    assert intent['inputs']==new['inputs'] and 'prospective only' in intent['scope']
    changed=deepcopy(new);changed['closure'][0]['sha256']='b'*64
    with pytest.raises(ValueError,match='execution closure'):
        validate_append_intent(old,changed,payload,**options)
    changed=deepcopy(payload);changed['prior_train']['sha256']='b'*64
    with pytest.raises(ValueError,match='predecessor sources'):
        validate_append_intent(old,new,changed,**options)


def test_resident_transition_seals_actual_successor_without_fake_restore(setup, monkeypatch):
    from copy import deepcopy
    import shutil
    from natlang_neuralese.train import checkpoint_offload as offload
    from natlang_neuralese.train.native_recovery_append import seal_append_transition
    slot,old=setup;old_path=slot.parent/'old-qualified.json';write_json_durable(old_path,old);old_ref={'path':str(old_path),'sha256':sha(old_path)}
    new=deepcopy(old);source=slot.parent/'next-source';source.write_bytes(b'next')
    ref={'path':str(source),'sha256':sha(source)};new['inputs']['train']=new['inputs']['foundation']=ref
    new['closure'].append({'role':'sources',**ref});new['additional_inputs']=[{'kind':'source',**old['inputs']['train']}]
    roles={'native_graphs':ref,'foundation_native_graphs':ref,'held_native_graphs':old['inputs']['held'],'ce_supervision_policy':old['inputs']['ce']}
    payload={'stage_id':'joint','predecessor':'old-generation','prior_train':old['inputs']['train'],'prior_held':old['inputs']['held'],'train':ref,'held':old['inputs']['held'],'plan':{'stage_inputs':{'joint':roles}}}
    payload.update(entries_sha256='c'*64,held_entries_sha256='d'*64)
    from natlang_neuralese.train.native_resident import digest
    committed={'schema':'natlang.native-resident-adoption/1','generation':digest(payload),'payload':payload,
               'validation':{'entries_sha256':'c'*64,'held_entries_sha256':'d'*64},'at_global_step':3}
    draft_path=slot.parent/'successor-draft.json';write_json_durable(draft_path,new);draft_ref={'path':str(draft_path),'sha256':sha(draft_path)}
    blobs=Path(new['destination'])/'closure';blobs.mkdir(exist_ok=True);objects=[]
    for digest_value in sorted({item['sha256'] for item in new['closure']}):
        item=next(item for item in new['closure'] if item['sha256']==digest_value);target=blobs/digest_value;shutil.copyfile(item['path'],target);s=target.stat()
        objects.append({'path':str(target),'sha256':digest_value,'identity':[s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]})
    stage_path=slot.parent/'staged.json';write_json_durable(stage_path,{'schema':'natlang.local-recovery-staged-closure/1','plan_sha256':RecoveryExportQueue._plan_digest(new),'objects':objects});stage_ref={'path':str(stage_path),'sha256':sha(stage_path)}
    wrapper_path=slot.parent/'intent.json';write_json_durable(wrapper_path,{'schema':'natlang.local-recovery-resident-append-intent/1','predecessor_plan':old_ref,'successor_draft':draft_ref,'staged_closure':stage_ref,'predecessor_generation':'old-generation','generation':digest(payload),'resident_directory':str(slot.parent)});wrapper_ref={'path':str(wrapper_path),'sha256':sha(wrapper_path)}
    current_path=slot.parent/'resident-current.json';write_json_durable(current_path,committed);current_ref={'path':str(current_path),'sha256':sha(current_path)}
    state={'training':{'global_step':3,'optimizer':{'m':1},'rng':{'r':1},'source_receipts':[ref,old['inputs']['held']],
           'options':{'native_graphs':ref['path'],'foundation_native_graphs':ref['path'],'held_native_graphs':old['inputs']['held']['path'],'ce_supervision_policy_binding':old['inputs']['ce']}}}
    result=seal_append_transition(old_ref,new,payload,committed,state,stage_id='joint',predecessor_generation='old-generation',
        proof_directory=slot.parent/'proof',extra_controls=(wrapper_ref,stage_ref,draft_ref,current_ref),predecessor_validator=lambda p:None,successor_validator=lambda p:None)
    assert seal_append_transition(old_ref,new,payload,committed,state,stage_id='joint',predecessor_generation='old-generation',
        proof_directory=slot.parent/'proof',extra_controls=(wrapper_ref,stage_ref,draft_ref,current_ref),predecessor_validator=lambda p:None,successor_validator=lambda p:None)==result
    plan=json.loads(Path(result['path']).read_text());proof=json.loads(Path(plan['restore_qualification']['path']).read_text())
    assert proof['mode']=='resident-append/1' and 'successor checkpoint/export must be independently verified' in proof['scope']
    monkeypatch.setattr(offload,'validate_recovery_plan_dependencies',lambda p,**kwargs:None)
    offload.validate_restore_qualification(plan)
    # Rebind every enclosing hash after tampering, so validation exercises the
    # transition semantics rather than merely noticing a changed file digest.
    proof_path=Path(plan['restore_qualification']['path']);proof=json.loads(proof_path.read_text())
    evidence_path=Path(proof['transition_evidence']['path']);evidence=json.loads(evidence_path.read_text())
    old_evidence_sha=proof['transition_evidence']['sha256'];evidence['actual_global_step']+=1;write_json_durable(evidence_path,evidence)
    proof['transition_evidence']['sha256']=sha(evidence_path)
    for item in plan['closure']:
        if item['path']==str(evidence_path) and item['sha256']==old_evidence_sha:item['sha256']=proof['transition_evidence']['sha256']
    proof['closure_identity']=offload.recovery_closure_identity(plan)
    old_proof_sha=plan['restore_qualification']['sha256'];write_json_durable(proof_path,proof);plan['restore_qualification']['sha256']=sha(proof_path)
    for item in plan['closure']:
        if item['path']==str(proof_path) and item['sha256']==old_proof_sha:item['sha256']=plan['restore_qualification']['sha256']
    with pytest.raises(ValueError,match='transition proof differs|not been committed'):offload.validate_restore_qualification(plan)
    bad=deepcopy(committed);bad['generation']='wrong'
    with pytest.raises(ValueError,match='not been committed'):
        seal_append_transition(old_ref,new,payload,bad,state,stage_id='joint',predecessor_generation='old-generation',
            proof_directory=slot.parent/'bad',extra_controls=(wrapper_ref,stage_ref,draft_ref,current_ref),predecessor_validator=lambda p:None,successor_validator=lambda p:None)


def test_staged_append_closure_rejects_changed_objects(setup):
    import shutil
    from natlang_neuralese.train.native_recovery_append import validate_staged_closure
    _,plan=setup;root=Path(plan['destination'])/'closure';root.mkdir(exist_ok=True);objects=[]
    for digest in sorted({item['sha256'] for item in plan['closure']}):
        source=next(Path(item['path']) for item in plan['closure'] if item['sha256']==digest)
        target=root/digest;shutil.copyfile(source,target);s=target.stat()
        objects.append({'path':str(target),'sha256':digest,'identity':[s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]})
    receipt={'schema':'natlang.local-recovery-staged-closure/1','plan_sha256':RecoveryExportQueue._plan_digest(plan),'objects':objects}
    validate_staged_closure(plan,receipt)
    Path(objects[0]['path']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed since independent readback'):
        validate_staged_closure(plan,receipt)


def test_queue_close_keeps_owner_until_staging_worker_exits(setup,monkeypatch):
    import threading,time
    from natlang_neuralese.train import native_recovery_append as append
    slot,plan=setup;started=threading.Event();release=threading.Event()
    def delayed(queue,catalog,payload,directory,*,stop_requested):
        started.set();assert release.wait(5)
        assert stop_requested()
        raise RuntimeError('stopped')
    monkeypatch.setattr(append,'stage_append_intent',delayed)
    q=RecoveryExportQueue(slot,plan,start_worker=False,plan_validator=lambda p:None,checkpoint_validator=lambda p,c:None)
    q.schedule_resident_append(slot.parent/'catalog',{'x':1},slot.parent);assert started.wait(5)
    q.close()
    with pytest.raises(BlockingIOError):
        RecoveryExportQueue(slot,plan,start_worker=False,plan_validator=lambda p:None,checkpoint_validator=lambda p,c:None)
    release.set();deadline=time.monotonic()+5
    while not q._owner.closed and time.monotonic()<deadline:time.sleep(.01)
    assert q._owner.closed
    replacement=RecoveryExportQueue(slot,plan,start_worker=False,plan_validator=lambda p:None,checkpoint_validator=lambda p,c:None);replacement.close()


def test_historical_program_custody_checks_receipt_and_archive_not_original(setup,monkeypatch):
    from natlang_neuralese.train import recipe,native_supervision
    from natlang_neuralese.train.checkpoint_offload import validate_recovery_plan_dependencies
    slot,plan=setup;original=slot.parent/'mutable-original.py';original.write_bytes(b'new code')
    archive=slot.parent/'archived-old.py';archive.write_bytes(b'old code')
    receipt=slot.parent/'history.json';write_json_durable(receipt,{'schema':'natlang.historical-source-program-preservation/1','programs':[{'original_path':str(original),'path':str(archive),'sha256':sha(archive),'bytes':archive.stat().st_size}]})
    archive_ref={'path':str(archive),'sha256':sha(archive)};receipt_ref={'path':str(receipt),'sha256':sha(receipt)}
    plan['historical_programs']=[{'original_path':str(original),'archive':archive_ref,'receipt':receipt_ref}]
    plan['closure'] += [{'role':'controls',**archive_ref},{'role':'controls',**receipt_ref}]
    monkeypatch.setattr(recipe,'native_source_dependencies',lambda path:[])
    monkeypatch.setattr(native_supervision,'ce_policy_dependencies',lambda ref:[ref])
    validate_recovery_plan_dependencies(plan,require_restore=False)
    original.unlink();validate_recovery_plan_dependencies(plan,require_restore=False)
    archive.write_bytes(b'changed')
    with pytest.raises(ValueError,match='archive or receipt changed'):
        validate_recovery_plan_dependencies(plan,require_restore=False)
