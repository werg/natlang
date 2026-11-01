"""Publish a signal-stopped direct SFT checkpoint without granting stage completion."""
from __future__ import annotations
import fcntl
import copy
import json
from pathlib import Path
from ..common.jsonio import write_json_durable
from ..train.native_resident import binding
from ..train.native_readiness import _bound


SESSION_RECONCILIATION = 'natlang.rl-production-session-reconciliation/1'
SESSION_PLAN = 'natlang.rl-production-session-plan/1'


def _strong_binding(path):
    """Hash a file's current bytes without relying on the mutable hash cache."""
    from ..common.hashing import sha256_file_hex
    path=Path(path).resolve(strict=True)
    if not path.is_file():raise ValueError('session reconciliation bindings must name regular files')
    before=path.stat()
    digest=sha256_file_hex(path)
    after=path.stat()
    if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(
            after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):
        raise ValueError('session reconciliation input changed while hashing')
    return {'path':str(path),'bytes':before.st_size,'sha256':digest}


def _verify_strong_ref(reference):
    if (not isinstance(reference,dict) or set(reference)!={'path','bytes','sha256'}
            or type(reference['bytes']) is not int or reference['bytes']<=0
            or not isinstance(reference['sha256'],str) or len(reference['sha256'])!=64):
        raise ValueError('session reconciliation needs an exact path/byte/SHA binding')
    if _strong_binding(reference['path'])!=reference:
        raise ValueError('session reconciliation input differs from its exact binding')
    return Path(reference['path']).resolve(strict=True)


def _session_control(config):
    # Only the new session's output directory and the promoted checkpoint are
    # transition-specific. Every source, runtime, prompt, task, actor and trainer
    # control remains covered by this digest.
    return {key:value for key,value in config.items() if key not in {'out','checkpoint'}}


def reconcile_production_session(state_path, config_path, *, plan_ref,
        report_ref, training_receipt_ref, checkpoint_ref, successor_config_ref, out):
    """Prepare a new native production session from one exact earned checkpoint.

    This is a data-only transition. It never loads, edits or awards credit to a
    checkpoint; it copies the controller's histories into a fresh state file,
    archives the interrupted operation, and starts that state at the exact
    existing full checkpoint. A plan must bind the successor runtime, sources,
    CE policy, held set, protection and preflight receipts before the operation
    will create the paired config/state session.
    """
    from ..common.hashing import canonical_json_sha256_hex
    state_path=Path(state_path).resolve(strict=True)
    config_path=Path(config_path).resolve(strict=True)
    transaction_dir=Path(out).resolve()
    plan_path=_verify_strong_ref(plan_ref)
    plan=json.loads(plan_path.read_text())
    if plan.get('schema')!=SESSION_PLAN:
        raise ValueError('invalid production-session reconciliation plan')
    predecessor=plan.get('predecessor') or {};successor=plan.get('successor') or {}
    if set(predecessor)!={'state','config','checkpoint','report','training_receipt'}:
        raise ValueError('reconciliation plan must bind every predecessor observation')
    if set(successor)!={'config','out','control_sha256','artifacts','qualifications','immediate_parent','selected_next_mode'}:
        raise ValueError('reconciliation plan must bind the complete successor session')
    if successor.get('selected_next_mode') not in {'joint_sft_and_rl','sft_first'}:
        raise ValueError('successor must explicitly select joint SFT/RL or an SFT-first window')

    # Verify all external pins before taking the controller lock; compare state
    # again under lock before constructing the transaction.
    for name,reference in predecessor.items():_verify_strong_ref(reference)
    successor_config_path=_verify_strong_ref(successor_config_ref)
    if successor['config']!=successor_config_ref:
        raise ValueError('successor config differs from the reconciliation plan')
    if successor['config']!=_strong_binding(successor_config_path):
        raise ValueError('successor config changed after reconciliation planning')
    successor_production_out=Path(successor['out']).resolve()
    if successor_production_out!=state_path.parent:
        raise ValueError('successor must reuse the current logical learner output and rolling checkpoint slot')
    if transaction_dir in (state_path.parent,config_path.parent) or state_path.parent in transaction_dir.parents:
        raise ValueError('reconciliation staging must remain outside the live production output')

    stable_artifacts={'runtime_manifest','production_recipe','train_source','rl_cases','held_source',
                      'ce_policy','tokenizer_manifest','base_manifest','source_admission',
                      'protection_union','continuation_preflight'}
    artifacts=successor.get('artifacts')
    if not isinstance(artifacts,dict) or set(artifacts)!=stable_artifacts:
        raise ValueError('successor plan lacks exact runtime/source/CE/held/protection bindings')
    for reference in artifacts.values():_verify_strong_ref(reference)
    qualifications=successor.get('qualifications')
    if (not isinstance(qualifications,dict)
            or set(qualifications)!={'runtime','source_admission','protection_union','continuation_preflight'}):
        raise ValueError('successor plan lacks required external verification receipts')
    for name,item in qualifications.items():
        if (not isinstance(item,dict) or set(item)!={'status','reference'} or item['status']!='passed'
                or item['reference']!=artifacts[{'runtime':'runtime_manifest','source_admission':'source_admission',
                    'protection_union':'protection_union','continuation_preflight':'continuation_preflight'}[name]]):
            raise ValueError('successor plan has an unverified runtime/source/protection preflight')
    immediate_parent=successor.get('immediate_parent')
    if (not isinstance(immediate_parent,dict) or set(immediate_parent)!={'sha256','global_step'}
            or not isinstance(immediate_parent['sha256'],str) or len(immediate_parent['sha256'])!=64
            or type(immediate_parent['global_step']) is not int or immediate_parent['global_step']<0):
        raise ValueError('successor plan lacks the exact immediate checkpoint parent observation')
    preflight=json.loads(Path(artifacts['continuation_preflight']['path']).read_text())
    if (preflight.get('schema')!='natlang.native-continuation-cpu-preflight/1'
            or preflight.get('parent_sha256')!=immediate_parent['sha256']
            or preflight.get('global_step')!=immediate_parent['global_step']):
        raise ValueError('successor preflight does not prove its exact immediate checkpoint parent')

    with (state_path.parent/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=json.loads(state_path.read_text());old_config=json.loads(config_path.read_text())
        observed_state=_strong_binding(state_path);observed_config=_strong_binding(config_path)
        if predecessor['state']!=observed_state or predecessor['config']!=observed_config:
            raise ValueError('production state/config changed after the reconciliation observation')
        checkpoint=_verify_strong_ref(checkpoint_ref)
        if predecessor['checkpoint']!=checkpoint_ref:
            raise ValueError('earned checkpoint differs from its exact reconciliation plan')
        if predecessor['report']!=report_ref or predecessor['training_receipt']!=training_receipt_ref:
            raise ValueError('SFT report/receipt differ from the reconciliation plan')
        report_path=_verify_strong_ref(report_ref);receipt_path=_verify_strong_ref(training_receipt_ref)
        report=json.loads(report_path.read_text());receipt=json.loads(receipt_path.read_text())
        if _strong_binding(successor_config_path)!=successor_config_ref:
            raise ValueError('successor production config changed during reconciliation')
        for reference in artifacts.values():_verify_strong_ref(reference)
        new_config=json.loads(successor_config_path.read_text())

        if (state.get('schema')!='natlang.rl-production-state/1'
                or old_config.get('schema')!='natlang.rl-production-config/1'
                or old_config.get('native_state') is not True
                or new_config.get('schema')!='natlang.rl-production-config/1'
                or new_config.get('native_state') is not True):
            raise ValueError('reconciliation is only for native production controller state/config')
        if (state.get('phase') not in {'awaiting_semantic_refresh','requested_stop','mandatory_sft_pending'}
                or not state.get('pending_native_checkpoint')):
            raise ValueError('production controller has no pending earned SFT checkpoint to reconcile')
        current_checkpoint=Path(state['pending_native_checkpoint']).resolve(strict=True)
        if (current_checkpoint!=checkpoint or Path(state.get('checkpoint','')).resolve()!=checkpoint
                or report.get('schema')!='natlang.neuralese-native-sft-report/1'
                or report.get('checkpoint_sha256')!=checkpoint_ref['sha256']
                or Path(report.get('checkpoint','')).resolve()!=checkpoint
                or receipt.get('schema')!='natlang.native-training-readiness/1'
                or receipt.get('checkpoint',{}).get('sha256')!=checkpoint_ref['sha256']
                or Path(receipt.get('checkpoint',{}).get('path','')).resolve()!=checkpoint
                or receipt.get('training_receipt')!={'path':str(report_path),'sha256':report_ref['sha256']}
                or receipt.get('continuation_boundary')!=report.get('continuation_boundary')
                or receipt.get('native_identity',{}).get('checkpoint_sha256')!=checkpoint_ref['sha256']):
            raise ValueError('pending SFT report and receipt do not bind the exact earned full checkpoint')
        boundary=report.get('continuation_boundary') or {};finite=report.get('progress_numerics') or {}
        exposure=receipt.get('exposure') or {}
        if (boundary.get('kind')!='semantic-refresh' or boundary.get('checkpoint_sha256')!=checkpoint_ref['sha256']
                or type(boundary.get('new_optimizer_updates')) is not int or boundary['new_optimizer_updates']<1
                or finite.get('checkpoint_sha256')!=checkpoint_ref['sha256'] or finite.get('passed') is not True
                or finite.get('nonfinite')!=[] or type(exposure.get('consumed_updates')) is not int
                or exposure['consumed_updates']<1 or type(exposure.get('supervised_token_visits')) is not int
                or exposure['supervised_token_visits']<1
                or not isinstance(receipt.get('mandatory_sft'),dict)):
            raise ValueError('pending SFT evidence does not prove finite earned updates and exposure')
        if state.get('mandatory_sft')!=receipt.get('mandatory_sft'):
            raise ValueError('controller and checkpoint disagree on unfinished SFT obligations')
        progress=state.get('last_learner_progress') or {}
        if (progress.get('checkpoint_sha256')!=checkpoint_ref['sha256']
                or progress.get('continuation_boundary')!=boundary
                or progress.get('mandatory_sft')!=receipt.get('mandatory_sft')):
            raise ValueError('controller progress summary differs from the exact saved SFT checkpoint report')
        parent_sha=(receipt.get('continuation') or {}).get('checkpoint_sha256')
        if not isinstance(parent_sha,str) or len(parent_sha)!=64:
            raise ValueError('pending SFT receipt lacks its exact saved predecessor')
        known_ancestors={}
        for record in state.get('direct_sft_refresh_history',[]):
            if (not isinstance(record,dict) or set(record) not in ({'path','sha256'},{'path','sha256','bytes'})):
                raise ValueError('production SFT history contains an invalid publication reference')
            publication_path=Path(record['path']).resolve(strict=True);publication_binding=_strong_binding(publication_path)
            if (publication_binding['sha256']!=record['sha256']
                    or ('bytes' in record and publication_binding['bytes']!=record['bytes'])):
                raise ValueError('production SFT publication history changed after its recorded transition')
            publication=json.loads(publication_path.read_text())
            for item in (publication.get('parent'),publication.get('checkpoint')):
                if isinstance(item,dict) and isinstance(item.get('sha256'),str):known_ancestors[item['sha256']]=True
        for key in ('last_completed_update','supervised_checkpoint_transition'):
            item=state.get(key) or {}
            for name in ('checkpoint','earned_checkpoint','prior_checkpoint'):
                ref=item.get(name)
                if isinstance(ref,dict) and isinstance(ref.get('sha256'),str):known_ancestors[ref['sha256']]=True
        if parent_sha not in known_ancestors:
            raise ValueError('SFT continuation predecessor is not in this production state’s authenticated lineage')

        if (Path(new_config.get('out','')).resolve()!=successor_production_out
                or Path(new_config.get('checkpoint','')).resolve()!=checkpoint
                or new_config.get('id')==old_config.get('id')
                or new_config.get('native_state') is not True
                or new_config.get('mandatory_sft_only',False) is not (successor['selected_next_mode']=='sft_first')
                or not isinstance(new_config.get('production_stage'),str)
                or not isinstance(new_config.get('seed_command'),list)
                or not isinstance(new_config.get('trainer_parameters'),dict)):
            raise ValueError('successor config mode/checkpoint must match the selected production-session transition')
        control_digest=canonical_json_sha256_hex(_session_control(new_config))
        if control_digest!=successor.get('control_sha256'):
            raise ValueError('successor control/runtime/source/held configuration differs from its plan')
        artifact_paths={
            'production_recipe':new_config.get('production_recipe'),
            'rl_cases':new_config.get('cases'),
            'held_source':new_config.get('held_native_graphs'),
            'ce_policy':new_config.get('ce_supervision_policy'),
            'tokenizer_manifest':new_config.get('rl_tokenizer'),
        }
        for role,path in artifact_paths.items():
            if not isinstance(path,str) or Path(path).resolve()!=Path(artifacts[role]['path']).resolve():
                raise ValueError(f'successor config does not use its planned {role} artifact')
        if (not isinstance(new_config.get('runtime'),str)
                or json.loads(Path(artifacts['runtime_manifest']['path']).read_text()).get('runtime_root')
                    !=str(Path(new_config['runtime']).resolve())):
            raise ValueError('successor config runtime differs from its exact frozen runtime manifest')
        if new_config.get('foundation_native_graphs')!=artifacts['train_source']['path']:
            raise ValueError('successor config does not use its planned admitted SFT source')
        if new_config.get('seed_native_graphs',artifacts['train_source']['path'])!=artifacts['train_source']['path']:
            raise ValueError('successor seed source differs from its planned admitted SFT source')

        # Verify the new production configuration and its effective recipe now,
        # before any controller can consume it.
        from ..train.recipe import resolve_recipe_data
        resolved=resolve_recipe_data(successor_config_path)
        if resolved.get('native_state') is not True or resolved.get('out')!=str(successor_production_out):
            raise ValueError('successor production config failed native recipe resolution')

        prior_actor={key:state[key] for key in ('actor_policy_pending_load','latest_reward','readiness',
            'foundation_candidate','foundation_qualified','last_learner_progress') if key in state}
        pending={key:state.get(key) for key in ('phase','pending_update_kind','pending_native_checkpoint',
            'pending_native_round_dir','last_learner_progress','last_sft_failure') if key in state}
        transition_id=canonical_json_sha256_hex({'plan':plan_ref,'checkpoint':checkpoint_ref,
            'predecessor_state':observed_state,'successor_config':successor_config_ref})
        next_state=copy.deepcopy(state)
        next_state['checkpoint']=str(checkpoint)
        next_state.update(schema='natlang.rl-production-state/1',phase='collect',production_complete=False)
        for key in ('pending_native_checkpoint','pending_native_round_dir','pending_update_kind',
                    'actor_policy_pending_load','latest_reward','readiness','foundation_candidate','foundation_qualified',
                    'last_learner_progress','last_sft_failure'):
            next_state.pop(key,None)
        next_state.setdefault('production_session_reconciliation_history',[]).append({
            'schema':SESSION_RECONCILIATION,'transition_id':transition_id,'checkpoint_sha256':checkpoint_ref['sha256'],
            'optimizer_updates_credited':0,'sft_stage_completed':False,'behavioral_evaluation_passed':False})
        next_state['production_session_reconciliation']={'schema':SESSION_RECONCILIATION,
            'transition_id':transition_id,'checkpoint_sha256':checkpoint_ref['sha256'],
            'pending_operation_archived':True,'optimizer_updates_credited':0,
            'unfinished_sft_obligations_preserved':True,'capability_evaluation_pending':True,
            'successor_config':successor_config_ref,'immediate_parent':immediate_parent,
            'selected_next_mode':successor['selected_next_mode'],'applied':False}

        allowed_transaction_files={'session-reconciliation.intent.json','proposed-state.json',
            'successor-config.json','session-reconciliation.json'}
        if transaction_dir.exists():
            if not transaction_dir.is_dir() or any(p.name not in allowed_transaction_files
                    for p in transaction_dir.iterdir()):
                raise ValueError('reconciliation staging directory contains unrelated files')
            if (not (transaction_dir/'session-reconciliation.json').exists()
                    and not (transaction_dir/'session-reconciliation.intent.json').exists()
                    and any(transaction_dir.iterdir())):
                raise ValueError('uncommitted reconciliation outputs have no durable intent')
        transaction_dir.mkdir(parents=True,exist_ok=True)
        final_path=transaction_dir/'session-reconciliation.json';state_out=transaction_dir/'proposed-state.json'
        config_out=transaction_dir/'successor-config.json'
        if final_path.exists():
            existing=json.loads(final_path.read_text())
            if existing.get('transition_id')!=transition_id:
                raise ValueError('successor output is committed to a different reconciliation')
            if _strong_binding(state_out)!=existing.get('successor_state'):
                raise ValueError('successor state changed after reconciliation commit')
            if _strong_binding(config_out)!=existing.get('successor_config_snapshot'):
                raise ValueError('successor config snapshot changed after reconciliation commit')
            return existing
        intent_path=transaction_dir/'session-reconciliation.intent.json'
        intent={'schema':'natlang.rl-production-session-reconciliation-intent/1','transition_id':transition_id,
            'plan':plan_ref,'predecessor_state':observed_state,'predecessor_config':observed_config,
            'checkpoint':checkpoint_ref,'successor_config':successor_config_ref}
        if intent_path.exists():
            if json.loads(intent_path.read_text())!=intent:
                raise ValueError('incomplete successor output belongs to another reconciliation')
        else:write_json_durable(intent_path,intent)
        if state_out.exists():
            if json.loads(state_out.read_text())!=next_state:
                raise ValueError('incomplete successor state differs from its exact reconciliation intent')
        else:write_json_durable(state_out,next_state)
        if config_out.exists():
            if json.loads(config_out.read_text())!=new_config:
                raise ValueError('staged successor config differs from its exact plan')
        else:
            write_json_durable(config_out,new_config)
        # The staged config is byte-identical in meaning but intentionally has
        # its own path binding for the later guarded session launch.
        staged_config_ref=_strong_binding(config_out)
        next_state['production_session_reconciliation']['successor_config_snapshot']=staged_config_ref
        # Update staged state after inserting its exact config binding.
        write_json_durable(state_out,next_state)
        result={'schema':SESSION_RECONCILIATION,'transition_id':transition_id,
            'plan':plan_ref,'predecessor_state':observed_state,'predecessor_config':observed_config,
            'checkpoint':checkpoint_ref,'report':report_ref,'training_receipt':training_receipt_ref,
            'successor_config':successor_config_ref,'successor_config_snapshot':staged_config_ref,
            'successor_state':_strong_binding(state_out),'install_state_path':str(state_path),
            'archived_pending_operation':pending,'archived_checkpoint_specific_evidence':prior_actor,
            'stable_artifacts':artifacts,'optimizer_updates_credited':0,'sft_stage_completed':False,
            'behavioral_evaluation_passed':False,'capability_evaluation_pending':True,
            'checkpoint_bytes_rehashed':True,'checkpoint_optimizer_rng_decoded':False,
            'checkpoint_preservation':'exact full-file bytes carried forward without loading or modification',
            'prepared_only':True,'controller_state_mutated':False}
        write_json_durable(final_path,result)
        return result


def apply_production_session_reconciliation(state_path, transaction_dir):
    """Install a reviewed prepared session under the existing production lock."""
    state_path=Path(state_path).resolve(strict=True);transaction_dir=Path(transaction_dir).resolve(strict=True)
    final_path=transaction_dir/'session-reconciliation.json'
    result=json.loads(final_path.read_text())
    if result.get('schema')!=SESSION_RECONCILIATION or result.get('prepared_only') not in (True,False):
        raise ValueError('missing prepared production-session reconciliation')
    if Path(result['install_state_path']).resolve()!=state_path:
        raise ValueError('reconciliation targets another production state')
    proposed_path=_verify_strong_ref(result['successor_state'])
    config_path=_verify_strong_ref(result['successor_config_snapshot'])
    proposed=json.loads(proposed_path.read_text());config=json.loads(config_path.read_text())
    if (proposed.get('production_session_reconciliation',{}).get('transition_id')!=result.get('transition_id')
            or proposed.get('production_session_reconciliation',{}).get('successor_config')!=result['successor_config']
            or Path(config.get('out','')).resolve()!=state_path.parent
            or Path(config.get('checkpoint','')).resolve()!=Path(result['checkpoint']['path']).resolve()):
        raise ValueError('prepared successor state/config do not form the exact planned pair')
    successor_config_path=_verify_strong_ref(result['successor_config'])
    if json.loads(successor_config_path.read_text())!=config:
        raise ValueError('successor config changed after reconciliation preparation')
    with (state_path.parent/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        current=_strong_binding(state_path)
        installed=json.loads(state_path.read_text())
        installed_transition=installed.get('production_session_reconciliation') or {}
        if installed_transition.get('transition_id')==result['transition_id'] and installed_transition.get('applied') is True:
            pass
        elif current!=result['predecessor_state']:
            raise ValueError('production state changed after session reconciliation was prepared')
        else:
            proposed['production_session_reconciliation']['applied']=True
            write_json_durable(state_path,proposed)
        # The normal production entrypoint rejects an old config after this
        # point, preventing accidental reuse of stale actor/runtime/source pins.
        result={**result,'prepared_only':False,'controller_state_mutated':True,
            'applied_state':_strong_binding(state_path)}
        write_json_durable(final_path,result)
        return result


def recover(state_path, *, report_ref, receipt_ref, custody_ref, qualification_ref, evaluation_ref, out):
    """Authenticate a completed update and retain its unfinished stage obligations."""
    from .native_raw_rounds import _published_checkpoint_state
    state_path, out = Path(state_path).resolve(), Path(out).resolve()
    with (state_path.parent / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.loads(state_path.read_text())
        if state.get('phase') != 'mandatory_sft_pending' or state.get('pending_update_kind') != 'mandatory_sft':
            raise ValueError('direct SFT recovery requires its stopped pending controller')
        if state.get('pending_native_checkpoint') or state.get('pending_native_round_dir'):
            raise ValueError('direct SFT recovery cannot replace another pending publication')
        def read(ref): return json.loads(_bound(ref).read_text())
        report, receipt, custody, qualification, evaluation = map(read, (report_ref, receipt_ref, custody_ref, qualification_ref, evaluation_ref))
        parent, child = custody['parent'], custody['successor']
        parent_file, child_file = _bound(parent), _bound(child)
        if Path(state['checkpoint']).resolve() != parent_file or parent['sha256'] == child['sha256']:
            raise ValueError('direct SFT recovery predecessor changed')
        boundary = report.get('continuation_boundary') or {}
        finite = report.get('progress_numerics') or {}
        if (custody.get('schema') != 'natlang.earned-sft-checkpoint-custody/1'
                or custody.get('old_bindings_and_quotas_preserved') is not True
                or custody.get('old_exposure_not_decreased') is not True
                or child['global_step'] != parent['global_step'] + 1
                or not all(r.get('optimizer_present') and r.get('rng_present') for r in (parent,child))
                or boundary != {'kind':'semantic-refresh','reached':False,'new_optimizer_updates':1,'checkpoint_sha256':child['sha256']}
                or report.get('checkpoint_sha256') != child['sha256']
                or Path(report.get('checkpoint','')).resolve() != child_file
                or receipt.get('checkpoint',{}).get('sha256') != child['sha256']
                or receipt.get('continuation',{}).get('checkpoint_sha256') != parent['sha256']
                or finite.get('checkpoint_sha256') != child['sha256'] or finite.get('passed') is not True
                or finite.get('nonfinite') != []):
            raise ValueError('direct SFT recovery lost exact earned checkpoint custody')
        q = qualification.get('report') or {}
        if (qualification.get('checkpoint_sha256') != child['sha256'] or qualification.get('optimizer_updates') != 0
                or q.get('checkpoint_sha256') != child['sha256'] or q.get('passed') is not True
                or (q.get('complete_gradient_qualification') or {}).get('passed') is not True):
            raise ValueError('direct SFT recovery requires actual checkpoint-bound gradient qualification')
        score = read(evaluation.get('score'))
        evaluation_binding = read(evaluation.get('binding'))
        if (evaluation.get('schema') != 'natlang.direct-sft-evaluation/1'
                or evaluation.get('checkpoint_sha256') != child['sha256']
                or evaluation_binding.get('checkpoint_sha256') != child['sha256']
                or score.get('cases') != 4 or score.get('step') != child['global_step']):
            raise ValueError('direct SFT recovery requires the recorded four-case outcome check')
        if out.exists(): raise ValueError('direct SFT recovery receipt already exists')
        checkpoint = {'path':str(child_file),'sha256':child['sha256']}
        result = {'schema':'natlang.direct-sft-interrupted-recovery/1', 'parent':binding(parent_file),
                  'checkpoint':binding(child_file), 'report':report_ref,'receipt':receipt_ref,'custody':custody_ref,
                  'qualification':qualification_ref,'evaluation':evaluation_ref,'training_credit':0,
                  'stage_completion_granted':False,'behavioral_qualification_granted':False}
        # Record the intent before state mutation, so a crash cannot silently publish.
        write_json_durable(out, result)
        next_state = _published_checkpoint_state(state, checkpoint)
        next_state.pop('pending_update_kind', None)
        next_state.setdefault('interrupted_sft_recovery_history', []).append(binding(out))
        write_json_durable(state_path, next_state)
        return result


def publish_completed(state_path, *, report_ref, receipt_ref, custody_ref, out):
    """Publish a qualified finite SFT refresh without claiming stage completion."""
    from .native_raw_rounds import _published_checkpoint_state
    from ..train.native_operation_report import supervised_progress_numerics_ok
    state_path,out=Path(state_path).resolve(),Path(out).resolve()
    def read(ref): return json.loads(_bound(ref).read_text())
    with (state_path.parent/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=json.loads(state_path.read_text());report,receipt,custody=map(read,(report_ref,receipt_ref,custody_ref))
        parent,child=custody['parent'],custody['successor'];parent_path,child_path=_bound(parent),_bound(child)
        boundary=report.get('continuation_boundary') or {}
        state_parent=Path(state['checkpoint']).resolve()
        parent_matches=state_parent==parent_path
        receipt_parents={parent['sha256']}
        # A rolling slot is atomically replaced after its prior checkpoint was
        # independently pinned. Bind that prior identity to the last publication.
        if not parent_matches and state_parent==child_path:
            history=state.get('direct_sft_refresh_history') or []
            prior=read(history[-1]) if history else {}
            receipt_parents.update(read(ref).get('parent',{}).get('sha256') for ref in history)
            parent_matches=(prior.get('schema')=='natlang.direct-sft-refresh-publication/1'
                            and prior.get('checkpoint',{}).get('path')==str(child_path)
                            and prior.get('checkpoint',{}).get('sha256')==parent['sha256'])
        if (state.get('phase')!='awaiting_semantic_refresh' or not parent_matches
                or Path(state.get('pending_native_checkpoint','')).resolve()!=child_path
                or custody.get('schema')!='natlang.earned-sft-checkpoint-custody/1'
                or custody.get('old_bindings_and_quotas_preserved') is not True
                or custody.get('old_exposure_not_decreased') is not True
                or child['global_step']!=parent['global_step']+1
                or not all(x.get('optimizer_present') and x.get('rng_present') for x in (parent,child))
                or boundary.get('kind')!='semantic-refresh' or boundary.get('reached') is not True
                or boundary.get('new_optimizer_updates')!=1 or boundary.get('checkpoint_sha256')!=child['sha256']
                or report.get('checkpoint_sha256')!=child['sha256']
                or receipt.get('checkpoint',{}).get('sha256')!=child['sha256']
                or receipt.get('continuation',{}).get('checkpoint_sha256') not in receipt_parents
                or not supervised_progress_numerics_ok(report,{'sha256':child['sha256']})):
            raise ValueError('SFT refresh publication lacks exact qualified full-state custody')
        if out.exists():raise ValueError('SFT refresh publication intent already exists')
        result={'schema':'natlang.direct-sft-refresh-publication/1','parent':binding(parent_path),'checkpoint':binding(child_path),
                'report':report_ref,'receipt':receipt_ref,'custody':custody_ref,'stage_completion_granted':False,
                'behavioral_qualification_granted':False,'additional_training_credit':0}
        write_json_durable(out,result)
        next_state=_published_checkpoint_state(state,{'path':str(child_path),'sha256':child['sha256']})
        for key in ('pending_native_checkpoint','pending_native_round_dir','pending_update_kind'):next_state.pop(key,None)
        next_state.setdefault('direct_sft_refresh_history',[]).append(binding(out))
        write_json_durable(state_path,next_state)
        return result
