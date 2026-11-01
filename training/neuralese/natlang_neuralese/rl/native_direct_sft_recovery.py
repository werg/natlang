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


def _strong_path_binding(path):
    """Bind a file or a complete directory tree (used by tokenizer inputs)."""
    from ..common.hashing import canonical_json_sha256_hex
    path=Path(path).resolve(strict=True)
    if path.is_file():return _strong_binding(path)
    if not path.is_dir():raise ValueError('session artifacts must be files or directories')
    before=path.stat();rows=[]
    for child in sorted(path.rglob('*')):
        if child.is_symlink():raise ValueError('directory artifacts may not contain symlinks')
        if child.is_file():
            bound=_strong_binding(child)
            rows.append({'relative_path':child.relative_to(path).as_posix(),
                         'bytes':bound['bytes'],'sha256':bound['sha256']})
    if not rows:raise ValueError('directory artifact has no files')
    after=path.stat()
    if (before.st_dev,before.st_ino,before.st_mtime_ns,before.st_ctime_ns)!=(
            after.st_dev,after.st_ino,after.st_mtime_ns,after.st_ctime_ns):
        raise ValueError('directory artifact changed while hashing')
    return {'path':str(path),'tree_sha256':canonical_json_sha256_hex(rows),'files':rows}


def _verify_strong_ref(reference):
    if (not isinstance(reference,dict) or set(reference)!={'path','bytes','sha256'}
            or type(reference['bytes']) is not int or reference['bytes']<=0
            or not isinstance(reference['sha256'],str) or len(reference['sha256'])!=64):
        raise ValueError('session reconciliation needs an exact path/byte/SHA binding')
    if _strong_binding(reference['path'])!=reference:
        raise ValueError('session reconciliation input differs from its exact binding')
    return Path(reference['path']).resolve(strict=True)


def _verify_historical_file_ref(reference):
    """Freshly verify publication refs emitted by the direct-SFT writer."""
    if (not isinstance(reference,dict) or set(reference) not in
            ({'path','sha256'},{'path','sha256','bytes'})
            or not isinstance(reference.get('path'),str)
            or not isinstance(reference.get('sha256'),str) or len(reference['sha256'])!=64
            or ('bytes' in reference and (type(reference['bytes']) is not int or reference['bytes']<=0))):
        raise ValueError('historical SFT evidence needs an exact path/SHA binding')
    actual=_strong_binding(reference['path'])
    if (actual['sha256']!=reference['sha256']
            or ('bytes' in reference and actual['bytes']!=reference['bytes'])):
        raise ValueError('historical SFT evidence differs from its exact file binding')
    return Path(reference['path']).resolve(strict=True)


def _authenticated_refresh_publication(reference, seen=None):
    """Verify one publication and any historical continuation ancestor it cites."""
    seen=set() if seen is None else set(seen)
    path=_verify_historical_file_ref(reference)
    if str(path) in seen:raise ValueError('historical SFT publication chain contains a cycle')
    seen.add(str(path))
    publication_binding=_strong_binding(path)
    publication=json.loads(path.read_text())
    if publication.get('schema')!='natlang.direct-sft-refresh-publication/1':
        raise ValueError('historical SFT chain contains a non-publication artifact')
    parent=publication.get('parent') or {};checkpoint=publication.get('checkpoint') or {}
    if (not isinstance(parent.get('sha256'),str) or len(parent['sha256'])!=64
            or not isinstance(checkpoint.get('sha256'),str) or len(checkpoint['sha256'])!=64
            or not isinstance(parent.get('path'),str) or not isinstance(checkpoint.get('path'),str)):
        raise ValueError('historical SFT publication omits exact checkpoint identities')
    report_ref=publication.get('report') or {};receipt_ref=publication.get('receipt') or {}
    custody_ref=publication.get('custody') or {}
    report=json.loads(_verify_historical_file_ref(report_ref).read_text())
    receipt=json.loads(_verify_historical_file_ref(receipt_ref).read_text())
    custody=json.loads(_verify_historical_file_ref(custody_ref).read_text())
    boundary=report.get('continuation_boundary') or {}
    custody_parent=custody.get('parent') or {};successor=custody.get('successor') or {}
    continuation=(receipt.get('continuation') or {}).get('checkpoint_sha256')
    operation_parent=receipt.get('operation_parent')
    operation_parent_matches=(operation_parent is None or
        (isinstance(operation_parent,dict)
            and operation_parent.get('sha256')==custody_parent.get('sha256')
            and Path(operation_parent.get('path','')).resolve()==Path(custody_parent.get('path','')).resolve()
            and operation_parent.get('global_step')==custody_parent.get('global_step')))
    if (custody.get('schema')!='natlang.earned-sft-checkpoint-custody/1'
            or Path(custody_parent.get('path','')).resolve()!=Path(parent['path']).resolve()
            or custody_parent.get('sha256')!=parent['sha256']
            or Path(successor.get('path','')).resolve()!=Path(checkpoint['path']).resolve()
            or successor.get('sha256')!=checkpoint['sha256']
            or not successor.get('optimizer_present') or not successor.get('rng_present')
            or not operation_parent_matches
            or report.get('checkpoint_sha256')!=checkpoint['sha256']
            or boundary.get('checkpoint_sha256')!=checkpoint['sha256']
            or boundary.get('kind')!='semantic-refresh' or boundary.get('reached') is not True
            or type(boundary.get('new_optimizer_updates')) is not int or boundary['new_optimizer_updates']<1
            or receipt.get('checkpoint',{}).get('sha256')!=checkpoint['sha256']
            or not isinstance(continuation,str) or len(continuation)!=64):
        raise ValueError('historical SFT publication report/receipt/custody do not prove its checkpoint transition')
    authenticated_ancestors={parent['sha256']}
    if continuation!=parent['sha256']:
        proof=publication.get('historical_parent_proof') or {}
        ancestor=proof.get('prior_continuation_ancestor') or {}
        ancestor_ref=proof.get('publication') or ancestor.get('publication') or {}
        ancestor_publication=_authenticated_refresh_publication(ancestor_ref,seen)
        ancestor_pub=ancestor_publication['publication']
        proof_operation_parent=proof.get('operation_parent') or {}
        operation_parent_proves_link=(
            proof_operation_parent.get('sha256')==parent['sha256']
            and Path(proof_operation_parent.get('path','')).resolve()==Path(parent['path']).resolve()
            and proof_operation_parent.get('global_step')==custody_parent.get('global_step'))
        publication_parent_proves_link=(ancestor_pub.get('checkpoint',{}).get('sha256')==parent['sha256'])
        if (proof.get('schema')!='natlang.historical-sft-parent-proof/1'
                or proof.get('bytes_verified_currently') is not False
                or (proof.get('identity') or {}).get('path')!=parent['path']
                or (proof.get('identity') or {}).get('sha256')!=parent['sha256']
                or (proof.get('identity') or {}).get('global_step')!=custody_parent.get('global_step')
                or proof.get('reported_continuation_ancestor_sha256')!=continuation
                or ancestor.get('sha256')!=continuation
                or not (publication_parent_proves_link or operation_parent_proves_link)
                or continuation not in ancestor_publication['authenticated_ancestors']
                or proof.get('report')!=ancestor_pub.get('report')
                or proof.get('receipt')!=ancestor_pub.get('receipt')
                or proof.get('custody')!=ancestor_pub.get('custody')):
            raise ValueError('historical SFT continuation ancestor is not linked by the ordered publication chain')
        authenticated_ancestors.update(ancestor_publication['authenticated_ancestors'])
        authenticated_ancestors.add(continuation)
    else:
        authenticated_ancestors.add(continuation)
    return {'binding':publication_binding,'publication':publication,'report':report,
        'receipt':receipt,'custody':custody,'continuation_sha256':continuation,
        'authenticated_ancestors':authenticated_ancestors}


def _verify_artifact_ref(reference):
    if not isinstance(reference,dict) or not isinstance(reference.get('path'),str):
        raise ValueError('session artifact needs an exact file/tree binding')
    if _strong_path_binding(reference['path'])!=reference:
        raise ValueError('session artifact differs from its exact file/tree binding')
    return Path(reference['path']).resolve(strict=True)


def _verify_frozen_runtime_inventory(runtime_root, runtime):
    """Recheck the frozen Node build inputs and copied npm dependency tree."""
    import hashlib
    node_build=runtime.get('node_build') or {};node_dependencies=runtime.get('node_dependencies') or {}
    source_inputs=node_build.get('source_inputs') or {}
    if (not source_inputs or not isinstance(node_build.get('source_inputs_sha256'),str)
            or not isinstance(node_dependencies.get('files'),int)
            or node_dependencies['files']<=0 or not isinstance(node_dependencies.get('sha256'),str)
            or len(node_dependencies['sha256'])!=64):
        raise ValueError('runtime closure omits its built-source or dependency inventory')
    for relative,digest in source_inputs.items():
        source_file=(runtime_root/relative).resolve(strict=True)
        if not source_file.is_relative_to(runtime_root) or _strong_binding(source_file)['sha256']!=digest:
            raise ValueError('frozen Node runtime source/build input changed: '+str(relative))
    from ..common.hashing import canonical_json_sha256_hex
    if canonical_json_sha256_hex(source_inputs)!=node_build['source_inputs_sha256']:
        raise ValueError('frozen Node runtime source-input inventory digest differs')
    compiler=(runtime_root/node_build.get('compiler','')).resolve(strict=True)
    if not compiler.is_relative_to(runtime_root):
        raise ValueError('frozen Node compiler escapes its runtime directory')
    dependencies=runtime_root/'node_modules'
    if not dependencies.is_dir():raise ValueError('frozen runtime has no copied Node dependency tree')
    digest=hashlib.sha256();count=size=0
    for path in sorted(dependencies.rglob('*')):
        relative=path.relative_to(dependencies).as_posix()
        if path.is_symlink():
            digest.update(f'{relative}\0link\0{path.readlink()}\n'.encode())
        elif path.is_file():
            file_digest=_strong_binding(path)['sha256']
            digest.update(f'{relative}\0{file_digest}\n'.encode());count+=1;size+=path.stat().st_size
    bash_path=dependencies/'just-bash/package.json'
    bash_version=json.loads(bash_path.read_text()).get('version') if bash_path.is_file() else None
    if (digest.hexdigest()!=node_dependencies.get('sha256') or count!=node_dependencies.get('files')
            or size!=node_dependencies.get('bytes') or bash_version!=node_dependencies.get('just_bash_version')):
        raise ValueError('frozen Node dependency tree differs from its runtime closure')


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
    if set(successor)!={'config','out','control_sha256','artifacts','immediate_parent','selected_next_mode'}:
        raise ValueError('reconciliation plan must bind the complete successor session')
    if successor.get('selected_next_mode') not in {'joint_sft_and_rl','sft_first'}:
        raise ValueError('successor must explicitly select joint SFT/RL or an SFT-first window')

    # Verify all external pins before taking the controller lock; compare state
    # again under lock before constructing the transaction.
    for name,reference in predecessor.items():
        if name!='checkpoint':_verify_strong_ref(reference)
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
                      'protection_union','continuation_preflight','checkpoint_audit'}
    artifacts=successor.get('artifacts')
    if not isinstance(artifacts,dict) or set(artifacts)!=stable_artifacts:
        raise ValueError('successor plan lacks exact runtime/source/CE/held/protection bindings')
    for reference in artifacts.values():_verify_artifact_ref(reference)
    immediate_parent=successor.get('immediate_parent')
    if (not isinstance(immediate_parent,dict) or set(immediate_parent)!={'sha256','global_step'}
            or not isinstance(immediate_parent['sha256'],str) or len(immediate_parent['sha256'])!=64
            or type(immediate_parent['global_step']) is not int or immediate_parent['global_step']<0):
        raise ValueError('successor plan lacks the exact immediate checkpoint parent observation')
    parent_proof=json.loads(Path(artifacts['continuation_preflight']['path']).read_text())
    if parent_proof.get('schema')=='natlang.native-continuation-cpu-preflight/1':
        parent_ok=(parent_proof.get('passed') is True
            and parent_proof.get('parent_sha256')==immediate_parent['sha256']
            and parent_proof.get('global_step')==immediate_parent['global_step'])
    elif parent_proof.get('schema')=='natlang.live-runtime-adoption/1':
        # The original preflight file may be the learner's rolling slot and can
        # be overwritten after launch. The immutable launch receipt records its
        # hash and the exact parent from which the actual SFT continuation ran.
        manifest_ref=parent_proof.get('manifest') or {}
        manifest_path=_verify_strong_ref(manifest_ref)
        manifest=json.loads(manifest_path.read_text())
        launch_parent=parent_proof.get('parent') or {}
        source_ref=parent_proof.get('source') or {}
        argv=manifest.get('argv') or []
        def option_value(option):
            if not isinstance(argv,list) or argv.count(option)!=1:return None
            index=argv.index(option)
            return argv[index+1] if index+1<len(argv) else None
        launch_preflight=parent_proof.get('preflight') or {}
        manifest_parent=manifest.get('parent') or {}
        manifest_source=manifest.get('source') or {}
        parent_ok=(manifest.get('schema')=='natlang.native-live-source-runtime-continuation/1'
            and parent_proof.get('status','').startswith('checks passed')
            and launch_parent.get('sha256')==immediate_parent['sha256']
            and launch_parent.get('global_step')==immediate_parent['global_step']
            and launch_parent.get('path')==manifest_parent.get('path')
            and launch_parent.get('sha256')==manifest_parent.get('sha256')
            and launch_parent.get('global_step')==manifest_parent.get('global_step')
            and source_ref.get('path')==artifacts['train_source']['path']
            and source_ref.get('sha256')==artifacts['train_source'].get('sha256')
            and source_ref.get('bytes')==artifacts['train_source'].get('bytes')
            and source_ref.get('path')==manifest_source.get('path')
            and source_ref.get('sha256')==manifest_source.get('sha256')
            and source_ref.get('bytes')==manifest_source.get('bytes')
            and option_value('--continue-from')==launch_parent.get('path')
            and option_value('--continued-native-graphs')==source_ref.get('path')
            and isinstance(launch_preflight.get('path'),str)
            and isinstance(launch_preflight.get('sha256'),str) and len(launch_preflight['sha256'])==64
            and parent_proof.get('runtime_commit')==manifest.get('runtime_commit'))
    else:parent_ok=False
    if not parent_ok:
        raise ValueError('successor preflight or launch evidence does not prove its exact immediate checkpoint parent')

    # Recompute runtime and source custody from their actual receipts. Caller
    # booleans such as `status: passed` are declarations, not evidence.
    from ..common.custody import ProtectedWorlds, REGISTRY as PROTECTED_REGISTRY
    protection=ProtectedWorlds.load()
    protection_path=_verify_artifact_ref(artifacts['protection_union'])
    protection_json=json.loads(protection_path.read_text())
    protected_digest=(protection_json.get('protected_worlds_sha256')
        or protection_json.get('current_protection_digest') or protection_json.get('digest'))
    if protection_path.resolve()==Path(PROTECTED_REGISTRY).resolve():
        protected_digest=protection.digest
    if protected_digest!=protection.digest:
        raise ValueError('successor protection receipt does not prove the current ProtectedWorlds union')
    runtime_path=_verify_artifact_ref(artifacts['runtime_manifest'])
    runtime=json.loads(runtime_path.read_text())
    if (runtime.get('schema')!='natlang.protected-runtime-input-closure/1'
            or not isinstance(runtime.get('commit'),str) or not runtime.get('commit')
            or runtime.get('protected_worlds_sha256')!=protection.digest
            or not isinstance(runtime.get('destination'),str)):
        raise ValueError('successor runtime closure lacks its exact commit or current protection digest')
    runtime_root=Path(runtime['destination']).resolve(strict=True)
    ts_host=runtime_root/'ts-host'
    if (not (ts_host/'package.json').is_file()
            or not (ts_host/'dist/teacher/collector.js').is_file()):
        raise ValueError('frozen Node host lacks its package metadata or built collector entrypoint')
    _verify_frozen_runtime_inventory(runtime_root,runtime)
    bound_files=runtime.get('bound_files') or []
    if not bound_files:raise ValueError('runtime closure has no pinned prompt/control inputs')
    for bound in bound_files:
        if not isinstance(bound,dict) or not isinstance(bound.get('path'),str):
            raise ValueError('runtime closure has an invalid bound control')
        if _strong_binding(bound['path'])['sha256']!=bound.get('sha256'):
            raise ValueError('runtime closure control changed: '+bound['path'])
    closure_bindings={(str(Path(item['path']).resolve()),item['sha256']) for item in bound_files}
    for role in ('production_recipe','train_source','held_source','ce_policy'):
        ref=artifacts[role]
        if 'sha256' not in ref or (str(Path(ref['path']).resolve()),ref['sha256']) not in closure_bindings:
            raise ValueError(f'frozen runtime closure does not bind the exact {role} input')
    admission_path=_verify_artifact_ref(artifacts['source_admission'])
    admission=json.loads(admission_path.read_text())
    source_binding=admission.get('source') or admission.get('train_source')
    if (not isinstance(source_binding,dict)
            or Path(source_binding.get('path','')).resolve()!=Path(artifacts['train_source']['path']).resolve()
            or source_binding.get('sha256')!=artifacts['train_source'].get('sha256')
            or not isinstance(admission.get('schema'),str)
            or admission.get('current_protection_digest',admission.get('protected_worlds_sha256'))!=protection.digest):
        raise ValueError('source admission receipt does not bind the exact train source and current protection union')
    recipe_binding=admission.get('recipe') or admission.get('production_recipe')
    if (not isinstance(recipe_binding,dict)
            or Path(recipe_binding.get('path','')).resolve()!=Path(artifacts['production_recipe']['path']).resolve()
            or recipe_binding.get('sha256')!=artifacts['production_recipe'].get('sha256')):
        raise ValueError('source admission receipt names another production recipe')
    ce_binding=admission.get('ce_policy')
    if (not isinstance(ce_binding,dict)
            or Path(ce_binding.get('path','')).resolve()!=Path(artifacts['ce_policy']['path']).resolve()
            or ce_binding.get('sha256')!=artifacts['ce_policy'].get('sha256')):
        raise ValueError('source admission receipt does not bind the exact CE policy')

    with (state_path.parent/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=json.loads(state_path.read_text());old_config=json.loads(config_path.read_text())
        observed_state=_strong_binding(state_path);observed_config=_strong_binding(config_path)
        if predecessor['state']!=observed_state or predecessor['config']!=observed_config:
            raise ValueError('production state/config changed after the reconciliation observation')
        checkpoint_path=Path(checkpoint_ref['path']).resolve(strict=True)
        checkpoint_stat=checkpoint_path.stat()
        checkpoint=_verify_strong_ref(checkpoint_ref)
        checkpoint_stable_stat=checkpoint_path.stat()
        if (checkpoint_stat.st_dev,checkpoint_stat.st_ino,checkpoint_stat.st_size,checkpoint_stat.st_mtime_ns,
                checkpoint_stat.st_ctime_ns)!=(checkpoint_stable_stat.st_dev,checkpoint_stable_stat.st_ino,
                checkpoint_stable_stat.st_size,checkpoint_stable_stat.st_mtime_ns,checkpoint_stable_stat.st_ctime_ns):
            raise ValueError('earned checkpoint changed during reconciliation observation')
        if predecessor['checkpoint']!=checkpoint_ref:
            raise ValueError('earned checkpoint differs from its exact reconciliation plan')
        if predecessor['report']!=report_ref or predecessor['training_receipt']!=training_receipt_ref:
            raise ValueError('SFT report/receipt differ from the reconciliation plan')
        report_path=_verify_strong_ref(report_ref);receipt_path=_verify_strong_ref(training_receipt_ref)
        report=json.loads(report_path.read_text());receipt=json.loads(receipt_path.read_text())
        if _strong_binding(successor_config_path)!=successor_config_ref:
            raise ValueError('successor production config changed during reconciliation')
        for reference in artifacts.values():_verify_artifact_ref(reference)
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
                or receipt.get('mandatory_sft')!=report.get('mandatory_sft')
                or receipt.get('native_identity',{}).get('checkpoint_sha256')!=checkpoint_ref['sha256']):
            raise ValueError('pending SFT report and receipt do not bind the exact earned full checkpoint')
        boundary=report.get('continuation_boundary') or {}
        from ..train.native_operation_report import supervised_progress_numerics_ok
        finite_ok=supervised_progress_numerics_ok(report,{'path':str(checkpoint),'sha256':checkpoint_ref['sha256']})
        exposure=receipt.get('exposure') or {}
        if (boundary.get('kind')!='semantic-refresh' or boundary.get('reached') is not True
                or boundary.get('checkpoint_sha256')!=checkpoint_ref['sha256']
                or type(boundary.get('new_optimizer_updates')) is not int or boundary['new_optimizer_updates']<1
                or finite_ok is not True or type(exposure.get('consumed_updates')) is not int
                or exposure['consumed_updates']<1 or type(exposure.get('supervised_token_visits')) is not int
                or exposure['supervised_token_visits']<1
                or not isinstance(receipt.get('mandatory_sft'),dict)):
            raise ValueError('pending SFT evidence does not prove finite earned updates and exposure')
        if (not isinstance(receipt.get('mandatory_sft'),dict)
                or receipt['mandatory_sft'].get('schema')!='natlang.native-sft-progress/1'
                or not isinstance(receipt['mandatory_sft'].get('stages'),dict)):
            raise ValueError('saved checkpoint has no structured unfinished-SFT obligations')
        active_source_sha=artifacts['train_source']['sha256']
        matching_stages=[stage for stage in receipt['mandatory_sft']['stages'].values()
            if isinstance(stage,dict) and stage.get('train_source_sha256')==active_source_sha]
        if not matching_stages:
            raise ValueError('saved checkpoint SFT obligations do not include the planned train source')
        audit=json.loads(Path(artifacts['checkpoint_audit']['path']).read_text())
        audit_parent=audit.get('parent') or {};audit_exposure=audit.get('source_exposure') or {}
        stage_sources={stage.get('source_hash') for stage in matching_stages}
        observed_groups=audit_exposure
        if (audit.get('schema')!='natlang.earned-full-parent-cutover-audit/1'
                or audit.get('status')!='actual saved fullstate verified; not a capability qualification'
                or audit_parent.get('path')!=str(checkpoint)
                or audit_parent.get('sha256')!=checkpoint_ref['sha256']
                or audit_parent.get('bytes')!=checkpoint_ref['bytes']
                or type(audit_parent.get('global_step')) is not int
                or audit_parent['global_step']!=immediate_parent['global_step']+1
                or audit_parent.get('source_rows')!=exposure.get('train_rows')
                or audit_parent.get('source_hash') not in stage_sources
                or audit.get('optimizer_present') is not True or audit.get('rng_present') is not True
                or set(observed_groups)!={'226','534'}
                or any(not isinstance(item,dict) or item.get('rows')!=int(group)
                    or type(item.get('visits')) is not int or item['visits']<0 or item['visits']>item['rows']
                    or item.get('visited_rows')!=item['visits'] for group,item in observed_groups.items())
                or sum(item['visits'] for item in observed_groups.values())>exposure.get('consumed_rows',0)):
            raise ValueError('independent full-checkpoint audit does not bind the saved 73 parent and actual source visits')
        progress=state.get('last_learner_progress') or {}
        # A direct SFT owner may advance the shared checkpoint after the
        # controller's last state write. Preserve that older controller view as
        # history; the report/receipt and live full checkpoint are the new
        # source of progress truth.
        controller_progress_is_current=(progress.get('checkpoint_sha256')==checkpoint_ref['sha256']
            and progress.get('continuation_boundary')==boundary)
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
        configured_runtime=Path(new_config.get('runtime','')).resolve(strict=True)
        if configured_runtime!=ts_host:
            raise ValueError('successor runtime does not select the frozen Node host directory')
        actor_environment=new_config.get('actor_environment') or {}
        actor_repo=actor_environment.get('NATLANG_REPO')
        if actor_repo is not None and Path(actor_repo).resolve()!=runtime_root:
            raise ValueError('successor actor host repository differs from the frozen runtime closure')
        if new_config.get('foundation_native_graphs')!=artifacts['train_source']['path']:
            raise ValueError('successor config does not use its planned admitted SFT source')
        if new_config.get('seed_native_graphs',artifacts['train_source']['path'])!=artifacts['train_source']['path']:
            raise ValueError('successor seed source differs from its planned admitted SFT source')
        if Path(new_config.get('rl_tokenizer','')).resolve()!=Path(artifacts['tokenizer_manifest']['path']).resolve():
            raise ValueError('successor config does not use its exact tokenizer directory')
        if Path(new_config.get('base','')).resolve()!=Path(artifacts['base_manifest']['path']).resolve():
            raise ValueError('successor config does not use its exact base model artifact')

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
        old_obligations=state.get('mandatory_sft')
        if old_obligations is not None:
            next_state.setdefault('historical_mandatory_sft_obligations',[]).append({
                'observed_state':observed_state,'obligations':copy.deepcopy(old_obligations),
                'relation':'controller snapshot before the authenticated direct-SFT checkpoint'})
        next_state['mandatory_sft']=copy.deepcopy(receipt['mandatory_sft'])
        for key in ('pending_native_checkpoint','pending_native_round_dir','pending_update_kind',
                    'actor_policy_pending_load','latest_reward','readiness','foundation_candidate','foundation_qualified',
                    'last_learner_progress','last_sft_failure'):
            next_state.pop(key,None)
        from ..train.native_operation_report import control_progress
        next_state['last_learner_progress']=control_progress(report,report_path)
        next_state.pop('update_learner_parameters',None)
        next_state.setdefault('production_session_reconciliation_history',[]).append({
            'schema':SESSION_RECONCILIATION,'transition_id':transition_id,'checkpoint_sha256':checkpoint_ref['sha256'],
            'optimizer_updates_credited':0,'sft_stage_completed':False,'behavioral_evaluation_passed':False})
        next_state['production_session_reconciliation']={'schema':SESSION_RECONCILIATION,
            'transition_id':transition_id,'checkpoint_sha256':checkpoint_ref['sha256'],
            'pending_operation_archived':True,'optimizer_updates_credited':0,
            'unfinished_sft_obligations_preserved':True,'capability_evaluation_pending':True,
            'successor_config':successor_config_ref,'immediate_parent':immediate_parent,
            'selected_next_mode':successor['selected_next_mode'],'applied':False,
            'controller_progress_was_current':controller_progress_is_current,
            'active_mandatory_sft_checkpoint_sha256':checkpoint_ref['sha256']}

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
        final_checkpoint_stat=checkpoint_path.stat()
        if (checkpoint_stable_stat.st_dev,checkpoint_stable_stat.st_ino,checkpoint_stable_stat.st_size,
                checkpoint_stable_stat.st_mtime_ns,checkpoint_stable_stat.st_ctime_ns)!=(
                final_checkpoint_stat.st_dev,final_checkpoint_stat.st_ino,final_checkpoint_stat.st_size,
                final_checkpoint_stat.st_mtime_ns,final_checkpoint_stat.st_ctime_ns):
            raise ValueError('earned checkpoint changed after its full-byte verification')
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
            'archived_mandatory_sft_obligations':copy.deepcopy(old_obligations),
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


def _completed_refresh_obligations(custody, parent):
    """Accept preserved quotas or an explicitly verified zero-exposure replacement."""
    if custody.get('old_bindings_and_quotas_preserved') is True:return True
    record=custody.get('unearned_replacement') or {}
    if (custody.get('old_obligations_transition')!='unearned-sft-supersession-verified'
        or record.get('schema')!='natlang.unearned-sft-source-replacement/1'
        or record.get('parent_sha256')!=parent.get('sha256') or record.get('old_graph_visits')!=0
        or record.get('weights_optimizer_rng_rl_exposure')!='preserved'):
        return False
    try:
        from ..train.native_continuation import checked_policy_history
        checked_policy_history(record)
        proof=_bound({'path':record['proof_path'],'sha256':record['proof_sha256']})
        source=json.loads(proof.read_text())
        return (source.get('parent_sha256')==parent['sha256'] and source.get('old_stage')==record.get('old_stage')
            and source.get('old_source_sha256')==record.get('old_source_sha256')
            and source.get('new_source_sha256')==record.get('new_source_sha256'))
    except (KeyError,TypeError,ValueError,OSError):return False


def publish_completed(state_path, *, report_ref, receipt_ref, custody_ref, out):
    """Publish a qualified finite SFT refresh without claiming stage completion."""
    from .native_raw_rounds import _published_checkpoint_state
    from ..train.native_operation_report import supervised_progress_numerics_ok
    state_path,out=Path(state_path).resolve(),Path(out).resolve()
    def read(ref): return json.loads(_bound(ref).read_text())
    with (state_path.parent/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=json.loads(state_path.read_text());report,receipt,custody=map(read,(report_ref,receipt_ref,custody_ref))
        parent,child=custody['parent'],custody['successor'];child_path=_bound(child)
        boundary=report.get('continuation_boundary') or {}
        updates=boundary.get('new_optimizer_updates')
        state_parent=Path(state['checkpoint']).resolve()
        parent_path=Path(parent.get('path','')).resolve()
        same_slot=(state_parent==child_path and parent_path==child_path)
        parent_matches=state_parent==parent_path and not same_slot
        operation_parent=receipt.get('operation_parent')
        operation_parent_matches=(operation_parent is None or
            (isinstance(operation_parent,dict)
                and operation_parent.get('sha256')==parent.get('sha256')
                and Path(operation_parent.get('path','')).resolve()==parent_path
                and operation_parent.get('global_step')==parent.get('global_step')))
        historical_parent_proof=None
        if parent_matches:
            # Ordinary publication: the exact parent bytes still occupy their
            # pinned path, so freshly verify them as before.
            parent_path=_bound(parent)
        elif same_slot:
            # Same-slot publication: the child has atomically replaced the
            # parent's bytes. Authenticate the parent from the last committed
            # publication and its report/receipt/custody; never hash the child
            # and label those bytes as the historical parent.
            history=state.get('direct_sft_refresh_history') or []
            if not history or not isinstance(history[-1],dict):
                raise ValueError('same-slot SFT publication lacks its last authenticated parent publication')
            prior_ref=history[-1]
            if set(prior_ref) not in ({'path','sha256'},{'path','sha256','bytes'}):
                raise ValueError('same-slot SFT history has an invalid last publication reference')
            prior_path=Path(prior_ref['path']).resolve(strict=True)
            authenticated_prior=_authenticated_refresh_publication(prior_ref)
            prior_binding=authenticated_prior['binding'];prior=authenticated_prior['publication']
            prior_checkpoint=prior.get('checkpoint') or {}
            prior_parent=prior.get('parent') or {}
            prior_is_immediate=(Path(prior_checkpoint.get('path','')).resolve()==child_path
                and prior_checkpoint.get('sha256')==parent.get('sha256'))
            if not prior_is_immediate and (operation_parent is None or not operation_parent_matches):
                raise ValueError('last SFT publication does not prove the exact overwritten parent identity')
            prior_report_ref=prior.get('report') or {};prior_receipt_ref=prior.get('receipt') or {}
            prior_custody_ref=prior.get('custody') or {}
            prior_report=authenticated_prior['report'];prior_receipt=authenticated_prior['receipt']
            prior_custody=authenticated_prior['custody']
            prior_successor=(prior_custody.get('successor') or {})
            if (prior_is_immediate and (Path(prior_successor.get('path','')).resolve()!=child_path
                    or prior_successor.get('sha256')!=parent.get('sha256')
                    or prior_successor.get('global_step')!=parent.get('global_step'))):
                raise ValueError('last SFT publication custody does not prove the immediate parent step')
            # If RL ran between SFT windows, the last direct-SFT publication is
            # an objective ancestor rather than the immediate rolling parent.
            # The new trainer receipt authenticates that immediate parent; the
            # ordered SFT publication chain authenticates the preserved root.
            ancestors=set(authenticated_prior['authenticated_ancestors'])
            lineage_binding=prior_binding;lineage_publication=prior
            reported_ancestor=(receipt.get('continuation') or {}).get('checkpoint_sha256')
            if reported_ancestor not in ancestors | {parent['sha256']}:
                raise ValueError('SFT receipt continuation is not the exact parent or its authenticated publication ancestor')
            historical_parent_proof={
                'schema':'natlang.historical-sft-parent-proof/1',
                'identity':{'path':str(child_path),'sha256':parent['sha256'],
                            'global_step':parent['global_step']},
                'bytes_verified_currently':False,
                'publication':lineage_binding,
                'report':lineage_publication.get('report'),
                'receipt':lineage_publication.get('receipt'),
                'custody':lineage_publication.get('custody'),
                'prior_continuation_ancestor':{'sha256':reported_ancestor,
                    'publication':lineage_binding,'receipt':lineage_publication.get('receipt')},
                'reported_continuation_ancestor_sha256':reported_ancestor,
                'operation_parent':{'path':str(parent_path),'sha256':parent['sha256'],
                    'global_step':parent['global_step']},
            }
            parent_matches=True
        receipt_parents={parent['sha256']}
        if historical_parent_proof is not None:
            receipt_parents.update(authenticated_prior['authenticated_ancestors'])
        if (state.get('phase')!='awaiting_semantic_refresh' or not parent_matches
                or Path(state.get('pending_native_checkpoint','')).resolve()!=child_path
                or custody.get('schema')!='natlang.earned-sft-checkpoint-custody/1'
                or not _completed_refresh_obligations(custody,parent)
                or custody.get('old_exposure_not_decreased') is not True
                or type(updates) is not int or updates<1
                or child['global_step']!=parent['global_step']+updates
                or not all(x.get('optimizer_present') and x.get('rng_present') for x in (parent,child))
                or boundary.get('kind')!='semantic-refresh' or boundary.get('reached') is not True
                or boundary.get('checkpoint_sha256')!=child['sha256']
                or report.get('checkpoint_sha256')!=child['sha256']
                or receipt.get('checkpoint',{}).get('sha256')!=child['sha256']
                or not operation_parent_matches
                or receipt.get('continuation',{}).get('checkpoint_sha256') not in receipt_parents
                or not supervised_progress_numerics_ok(report,{'sha256':child['sha256']})):
            raise ValueError('SFT refresh publication lacks exact qualified full-state custody')
        if out.exists():raise ValueError('SFT refresh publication intent already exists')
        result={'schema':'natlang.direct-sft-refresh-publication/1',
                'parent':(binding(parent_path) if historical_parent_proof is None else historical_parent_proof['identity']),
                'historical_parent_proof':historical_parent_proof,
                'checkpoint':binding(child_path),
                'report':report_ref,'receipt':receipt_ref,'custody':custody_ref,'stage_completion_granted':False,
                'behavioral_qualification_granted':False,'additional_training_credit':0}
        write_json_durable(out,result)
        next_state=_published_checkpoint_state(state,{'path':str(child_path),'sha256':child['sha256']})
        for key in ('pending_native_checkpoint','pending_native_round_dir','pending_update_kind'):next_state.pop(key,None)
        next_state.setdefault('direct_sft_refresh_history',[]).append(binding(out))
        write_json_durable(state_path,next_state)
        return result
