"""Exact optimizer-owner drain evidence, shared by raw bootstrap and operators.

The supervisor chooses when to drain and sends the existing safe StopSignal.
This module only records/verifies actual process identities and immutable state;
it never signals a PID or invents a successful training/readiness outcome.
"""
from __future__ import annotations
import json
from pathlib import Path
import time
from ..common.jsonio import write_json_durable,write_bytes_durable
from ..train.native_readiness import _bound
from ..train.native_resident import binding

SCHEMA='natlang.native-raw-optimizer-handoff/1'
INTENT='natlang.native-raw-optimizer-drain/2'
LEGACY_INTENT='natlang.native-raw-optimizer-drain/1'
SAVED_UPDATE_RECOVERY='natlang.native-owned-saved-update-recovery/1'
SAVED_UPDATE_ACK='natlang.native-owned-saved-update-acknowledgement/1'


def _process_stat(pid):
    try:
        fields=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()
        return {'pid':pid,'start_ticks':fields[19],'state':fields[0],'ppid':int(fields[1])}
    except (FileNotFoundError,ProcessLookupError):return None


def observe(pid):
    try:
        before=_process_stat(pid)
        if before is None:return None
        value={**before,'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
               'argv_hex':Path('/proc',str(pid),'cmdline').read_bytes().hex()}
        after=_process_stat(pid)
        if after is None:return None
        if any(before[key]!=after[key] for key in ('pid','start_ticks','ppid')):
            raise ValueError('raw handoff process identity or parent changed during observation')
        return value
    except (FileNotFoundError,ProcessLookupError):return None


def same_process(expected,actual):
    return actual is not None and all(actual.get(key)==expected.get(key) for key in ('pid','boot_id','start_ticks'))


def process_argv(identity):
    """Decode exact /proc cmdline, preserving empty POSIX argument values."""
    raw=bytes.fromhex(identity['argv_hex'])
    if not raw or not raw.endswith(b'\0'):
        raise ValueError('captured live process argv needs its terminal NUL')
    return [part.decode() for part in raw[:-1].split(b'\0')]


def _thread_children(pid):
    try:tasks=list(Path('/proc',str(pid),'task').iterdir())
    except (FileNotFoundError,ProcessLookupError,PermissionError):return set()
    children=set()
    for task in tasks:
        try:children.update(int(value) for value in (task/'children').read_text().split())
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
    return children


def _process_table():
    table={}
    for path in Path('/proc').iterdir():
        if path.name.isdigit():
            try:value=_process_stat(int(path.name))
            except PermissionError:continue  # Unrelated processes can be private.
            if value is not None:table[value['pid']]=value
    return table


def _capture_process_tree(owner):
    table=_process_table();identities={owner['pid']:owner};links=[];todo=[owner['pid']]
    while todo:
        pid=todo.pop();parent=identities[pid];threads=_thread_children(pid)
        candidates=threads|{key for key,value in table.items() if value['ppid']==pid}
        for ident in sorted(candidates):
            child=observe(ident)
            if child is None or child['state'] in ('Z','X'):continue  # Already finished.
            scanned=table.get(ident)
            if (child['ppid']!=pid or child['boot_id']!=owner['boot_id']
                    or int(child['start_ticks'])<int(parent['start_ticks'])
                    or scanned is not None and any(child[key]!=scanned[key] for key in ('start_ticks','ppid'))):
                raise ValueError('raw handoff descendant identity or parent changed during census')
            if ident in identities:
                raise ValueError('raw handoff process census contains a repeated or cyclic descendant')
            if not same_process(parent,observe(pid)):
                raise ValueError('raw handoff parent identity changed during census')
            identities[ident]=child;todo.append(ident)
            links.append({'parent':parent,'child':child,
                          'observed_by':(['thread-children'] if ident in threads else [])+
                                        (['proc-ppid'] if scanned is not None and scanned['ppid']==pid else [])})
    # Do not publish a silently partial tree if a child was created, reparented
    # or recycled while the census was being assembled.
    final=_process_table()
    for ident,value in final.items():
        if value['ppid'] in identities and ident not in identities:
            if value['state'] in ('Z','X'):
                actual=observe(ident)
                if actual is None or actual['state'] in ('Z','X'):continue
            raise ValueError('raw handoff process tree grew during census; capture again')
    for ident,previous in identities.items():
        actual=observe(ident)
        if actual is None or actual['state'] in ('Z','X'):
            if ident==owner['pid']:raise ValueError('raw handoff owner exited during census')
            continue
        if (not same_process(previous,actual) or previous['argv_hex']!=actual['argv_hex']
                or previous['ppid']!=actual['ppid']):
            raise ValueError('raw handoff process identity or parent changed during census')
        for child in _thread_children(ident):
            if child not in identities:
                actual=observe(child)
                if actual is not None and actual['state'] not in ('Z','X'):
                    raise ValueError('raw handoff process tree grew during census; capture again')
    return [value for ident,value in identities.items() if ident!=owner['pid']],links


def capture_drain_intent(launch, out):
    """Call before the existing owner-safe signal, on the owning machine."""
    launch_path=_bound(launch);launch_bytes=launch_path.read_bytes();record=json.loads(launch_bytes)
    owner=observe(record['pid'])
    if owner is None or owner['state'] in ('Z','X'):
        raise ValueError('raw handoff needs its actual live original optimizer owner')
    argv=process_argv(owner)
    if argv!=record.get('argv') or record.get('identity') and not same_process(record['identity'],owner):
        raise ValueError('raw handoff launch argv or identity differs from its actual process')
    children,links=_capture_process_tree(owner)
    launch_snapshot=Path(out).with_name(Path(out).stem+'-launch.json')
    write_bytes_durable(launch_snapshot,launch_bytes)
    if binding(launch_snapshot)['sha256']!=launch['sha256']:
        raise ValueError('raw handoff launch changed while capturing its immutable snapshot')
    write_json_durable(out,{'schema':INTENT,'at':time.time(),'source_launch':launch,
        'launch':binding(launch_snapshot),'owner':owner,'children':children,
        'process_tree':{'schema':'natlang.linux-process-census/1','links':links}})
    return binding(out)


def _identity(value):
    return (isinstance(value,dict) and type(value.get('pid')) is int and value['pid']>0
        and isinstance(value.get('boot_id'),str) and bool(value['boot_id'])
        and isinstance(value.get('start_ticks'),str) and value['start_ticks'].isdigit()
        and isinstance(value.get('argv_hex'),str) and bool(value['argv_hex'])
        and value.get('state') not in (None,'Z','X'))


def _verify_process_tree(intent):
    census=intent.get('process_tree')
    if census is None:
        if intent.get('schema')==LEGACY_INTENT:return  # Preserve historical scope.
        raise ValueError('raw handoff process census is required for drain/2')
    if not isinstance(census,dict) or census.get('schema')!='natlang.linux-process-census/1' or not isinstance(census.get('links'),list):
        raise ValueError('raw handoff process census schema is invalid')
    known={intent['owner']['pid']:intent['owner']};children={item['pid']:item for item in intent['children']}
    for link in census['links']:
        if not isinstance(link,dict):raise ValueError('raw handoff process census link is invalid')
        parent,child=link.get('parent'),link.get('child');origins=link.get('observed_by')
        if (not isinstance(parent,dict) or not isinstance(child,dict)
                or known.get(parent.get('pid'))!=parent or children.get(child.get('pid'))!=child
                or child.get('pid') in known or type(child.get('ppid')) is not int or child['ppid']!=parent['pid']
                or child.get('boot_id')!=parent.get('boot_id')
                or not str(child.get('start_ticks','')).isdigit()
                or int(child['start_ticks'])<int(parent['start_ticks'])
                or not isinstance(origins,list) or not origins or any(not isinstance(value,str) for value in origins)
                or len(origins)!=len(set(origins))
                or any(value not in ('thread-children','proc-ppid') for value in origins)):
            raise ValueError('raw handoff process census lost exact parent/child identity evidence')
        known[child['pid']]=child
    if set(known)!={intent['owner']['pid'],*children}:
        raise ValueError('raw handoff process census omitted a captured descendant')


def complete_drain(intent, terminal, checkpoint, out, *, original_slot, resident_directory=None,
                   inprocess_completion=None, return_verified=False):
    """After owner/children drain and checkpoint pin, persist exact observations."""
    value=json.loads(_bound(intent).read_text())
    if value.get('schema') not in (INTENT,LEGACY_INTENT):raise ValueError('raw handoff requires its exact pre-signal intent')
    identities=[value['owner'],*value['children']]
    observations=[observe(item['pid']) for item in identities]
    if any(same_process(old,new) and new['state'] not in ('Z','X') for old,new in zip(identities,observations)):
        raise ValueError('raw optimizer owner or captured child is still alive')
    pinned=_bound(checkpoint);slot=_bound(original_slot)
    pin_stat,slot_stat=pinned.stat(),slot.stat()
    if (checkpoint['sha256']!=original_slot['sha256'] or pin_stat.st_size!=slot_stat.st_size
            or (pin_stat.st_dev,pin_stat.st_ino)==(slot_stat.st_dev,slot_stat.st_ino)):
        raise ValueError('raw handoff needs an independent immutable pin of the exact drained checkpoint')
    terminal_snapshot=Path(out).with_name(Path(out).stem+'-terminal.json')
    write_bytes_durable(terminal_snapshot,_bound(terminal).read_bytes())
    record={'schema':SCHEMA,'at':time.time(),'intent':intent,'source_terminal':terminal,
            'terminal':binding(terminal_snapshot),'checkpoint':checkpoint,
            'checkpoint_pin':{'original_slot':original_slot,'bytes':pin_stat.st_size,
                'original_inode':[slot_stat.st_dev,slot_stat.st_ino],'pin_inode':[pin_stat.st_dev,pin_stat.st_ino]},
            'observations':observations,'resident':None}
    if inprocess_completion is not None:
        from .native_owned_dispatch import verify_completion_contract
        terminal_value=json.loads(_bound(terminal).read_text())
        verify_completion_contract(inprocess_completion,value,terminal,terminal_value)
        if Path(terminal['path']).with_name('entry-failed.json').exists():
            raise ValueError('in-process owner has failure evidence, not a successful return')
        completion_snapshot=Path(out).with_name(Path(out).stem+'-completion.json')
        write_bytes_durable(completion_snapshot,_bound(inprocess_completion).read_bytes())
        record['inprocess_completion']=binding(completion_snapshot)
    if resident_directory is not None:
        from ..train.native_resident import current
        directory=Path(resident_directory);state=current(directory)
        snapshots=Path(out).parent/(Path(out).stem+'-resident');snapshots.mkdir(parents=True,exist_ok=True)
        def snapshot(name):
            source=directory/name
            if not source.exists():return None
            reference=binding(source);target=snapshots/name
            write_bytes_durable(target,_bound(reference).read_bytes())
            if binding(target)['sha256']!=reference['sha256']:raise ValueError('resident handoff snapshot changed')
            return binding(target)
        record['resident']={'directory':str(directory.resolve()),'generation':state['generation'] if state else None,
            'current':snapshot('current.json') if state else None,
            'initial':snapshot('initial.json'),'pending_uncredited':snapshot('pending.json')}
        if state is None:
            source=directory.parent/'recipe-plan.json'
            if source.exists():
                target=snapshots/'recipe-plan.json';write_bytes_durable(target,source.read_bytes())
                record['resident']['plan']=binding(target)
    write_json_durable(out,record)
    reference=binding(out)
    verified=verify_handoff(reference)
    return (reference,verified) if return_verified else reference


def _captured_owner_controls(reference):
    """One exact captured-launch/tree verifier for successful and failed owners."""
    intent=json.loads(_bound(reference).read_text())
    if intent.get('schema') not in (INTENT,LEGACY_INTENT) or not isinstance(intent.get('children'),list):
        raise ValueError('raw handoff needs its original pre-signal process tree')
    launch=json.loads(_bound(intent['launch']).read_text())
    if intent.get('source_launch',intent['launch'])['sha256']!=intent['launch']['sha256']:
        raise ValueError('raw handoff launch snapshot differs from original source')
    owner=intent.get('owner');identities=[owner,*intent['children']]
    if (not all(_identity(item) for item in identities) or launch.get('pid')!=owner['pid']
            or len({item['pid'] for item in identities})!=len(identities)
            or any(item['boot_id']!=owner['boot_id'] for item in identities)
            or process_argv(owner)!=launch.get('argv')):
        raise ValueError('raw handoff changed its exact boot/start/PID/argv owner binding')
    _verify_process_tree(intent)
    return intent,launch,identities


def _checkpoint_pin_controls(record):
    pin=record.get('checkpoint_pin') or {}
    checkpoint=record.get('checkpoint')or{}
    if (not isinstance(checkpoint.get('path'),str)or not Path(checkpoint['path']).is_absolute()
            or not isinstance(checkpoint.get('sha256'),str)or len(checkpoint['sha256'])!=64
            or any(char not in '0123456789abcdef'for char in checkpoint['sha256'])
            or pin.get('original_slot',{}).get('sha256')!=checkpoint['sha256']
            or not isinstance(pin.get('original_slot',{}).get('path'),str)
            or not Path(pin['original_slot']['path']).is_absolute()
            or type(pin.get('bytes')) is not int or pin['bytes']<=0
            or any(not isinstance(pin.get(key),list) or len(pin[key])!=2
                   or any(type(item) is not int or item<0 for item in pin[key]) for key in ('original_inode','pin_inode'))
            or pin['original_inode']==pin['pin_inode']):
        raise ValueError('raw handoff lost its exact independently pinned checkpoint evidence')


def verify_handoff_controls(reference, *, requirement=None):
    """Authenticate historical owner/drain controls without obsolete payloads."""
    record=json.loads(_bound(reference).read_text())
    if record.get('schema')!=SCHEMA:raise ValueError('raw bootstrap needs an exact original-owner handoff receipt')
    intent,launch,identities=_captured_owner_controls(record['intent'])
    terminal=json.loads(_bound(record['terminal']).read_text())
    if record.get('source_terminal',record['terminal'])['sha256']!=record['terminal']['sha256']:
        raise ValueError('raw handoff terminal snapshot differs from original source')
    if record.get('inprocess_completion') is not None:
        from .native_owned_dispatch import verify_completion_contract
        verify_completion_contract(record['inprocess_completion'],intent,
            record.get('source_terminal',record['terminal']),terminal)
        successful=True
    else:
        successful=type(terminal.get('returncode')) is int and terminal['returncode']==0
    if (not successful or any(type(item) not in (int,float) for item in
            (launch.get('at'),intent.get('at'),terminal.get('at'),record.get('at')))
            or not launch['at']<=intent['at']<=terminal['at']<=record['at']):
        raise ValueError('raw bootstrap requires successful exact post-drain terminal evidence')
    observations=record.get('observations')
    if not isinstance(observations,list) or len(observations)!=len(identities):
        raise ValueError('raw handoff omitted original process-tree drain observations')
    local_boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    for old,new in zip(identities,observations):
        if (new is not None and (new.get('pid')!=old['pid'] or same_process(old,new) and new.get('state') not in ('Z','X'))):
            raise ValueError('raw optimizer owner or captured child did not drain')
        if old['boot_id']==local_boot:
            from ..train.native_resident import owner_alive
            if owner_alive(old) and (observe(old['pid']) or {}).get('state') not in ('Z','X'):
                raise ValueError('raw optimizer owner or captured child is currently alive')
    if requirement:
        if requirement.get('trainer_launch')!=intent.get('source_launch',intent['launch']):
            raise ValueError('raw handoff belongs to another declared original trainer launch')
        if Path(requirement['trainer_exit']).resolve()!=Path(record.get('source_terminal',record['terminal'])['path']).resolve():
            raise ValueError('raw handoff terminal belongs to another original optimizer owner')
    _checkpoint_pin_controls(record)
    return record


def verify_handoff_metadata(reference, *, requirement=None):
    """Verify all owner evidence and the actual complete byte-bound checkpoint."""
    record=verify_handoff_controls(reference,requirement=requirement)
    checkpoint=_bound(record['checkpoint'])
    if record['checkpoint_pin']['bytes']!=checkpoint.stat().st_size:
        raise ValueError('raw handoff lost its exact independently pinned checkpoint evidence')
    return record


def _verify_reached_operational_boundary(recovery,boundary,updates):
    """A reached scheduling quantum can still have unfinished end evaluation."""
    if boundary['reached'] is False:
        return
    stop=json.loads(_bound(recovery['operational_stop']).read_text())
    argv=process_argv(stop['trainer'])
    if (boundary.get('kind')!='semantic-refresh' or argv.count('-m')!=1
            or argv[argv.index('-m')+1]!='natlang_neuralese.train.native_state_sft'):
        raise ValueError('reached operational recovery requires its actual SFT refresh boundary')
    from ..train.native_state_sft import build_parser
    try:
        args=build_parser().parse_args(argv[argv.index('-m')+2:])
    except (SystemExit,TypeError,ValueError) as error:
        raise ValueError('reached operational recovery lost its declared refresh quantum') from error
    quantum=args.refresh_after_updates
    if type(quantum)is not int or quantum<=0 or updates<quantum:
        raise ValueError('reached operational recovery did not earn its declared refresh quantum')


def _saved_update_recovery_controls(reference,receipt):
    """Authenticate earned historical work without treating failure as success."""
    from .native_bridge import verify_result_inventory
    from ..train.native_resident import code_inventory
    from ..common.hashing import canonical_json_sha256_hex as digest
    recovery=json.loads(_bound(reference).read_text())
    from . import native_interrupted_update as interrupted
    operational = recovery.get('schema') == interrupted.SCHEMA
    if (recovery.get('schema') not in (SAVED_UPDATE_RECOVERY, interrupted.SCHEMA)
            or recovery.get('original_operation_still_failed')is not True
            or recovery.get('complete_optimizer_rng_and_ordinary_exposure_verified')is not True):
        raise ValueError('saved-update recovery needs its explicit original failed-operation evidence')
    intent,launch,identities=_captured_owner_controls(recovery['captured_owner_intent'])
    terminal=json.loads(_bound(recovery['actual_failed_terminal']).read_text())
    exit_record=json.loads(_bound(terminal['actual_provider_exit']).read_text())
    spec=json.loads(_bound(recovery['original_request']).read_text());argv=spec.get('argv') or []
    def argument(flag):
        if argv.count(flag)!=1 or argv.index(flag)+1>=len(argv):
            raise ValueError('saved-update recovery lost its exact failed bridge invocation')
        return argv[argv.index(flag)+1]
    request_path=Path(argument('--request'));result_path=_bound(recovery['actual_failed_result'])
    request,result=verify_result_inventory(request_path,result_path)
    # The provider's mutable launch file can acquire later bookkeeping. The
    # captured observed launch already proves its actual PID/boot/start/argv;
    # retain the original source binding as provenance without reopening it.
    provider_launch=launch.get('original_provider_launch')or{}
    if (spec.get('kind')!='round' or request['kind']!='round' or result.get('status')!='failed'
            or spec.get('request_id')!=request['request_id'] or launch.get('request')!=request['request_id']
            or argv!=launch['argv'] or not (interrupted.bridge_invocation(spec,argv,recovery,intent) if operational
                else argv[1:4]==['-u','-m','natlang_neuralese.rl.native_bridge'])
            or Path(argument('--result')).resolve()!=result_path.resolve()
            or terminal.get('actual_operation_result')!=recovery['actual_failed_result']
            or type(terminal.get('returncode'))is not int or terminal['returncode']==0
            or exit_record.get('exit_code')!=terminal['returncode']
            or not isinstance(provider_launch.get('sha256'),str)or len(provider_launch['sha256'])!=64
            or Path(provider_launch.get('path','')).parent!=Path(terminal['actual_provider_exit']['path']).parent
            or not launch['at']<=intent['at']<=terminal['at']<=recovery['at']
            or exit_record.get('at')!=terminal['at']):
        raise ValueError('saved-update recovery changed its original failed request/owner/terminal')
    artifacts={item['path']:item for item in result['artifacts']}
    def artifact(reference,*,read=True):
        actual=artifacts.get(reference.get('path'))
        if not actual or actual['sha256']!=reference.get('sha256'):
            raise ValueError('saved-update recovery control is absent from the original failed inventory')
        if read:
            path=_bound(actual)
            if path.stat().st_size!=actual['bytes']:raise ValueError('saved-update recovery control size changed')
            return json.loads(path.read_text())
        return actual
    artifact({'path':str(request_path),'sha256':result['request_sha256']})
    artifact(request['config'])
    slot=artifact(recovery['original_slot'],read=False)
    report=artifact(recovery['report']);readiness=artifact(receipt)
    certificate=artifact(recovery['runtime_certificate'])
    if not operational or recovery.get('partial_quality'):
        artifact(recovery['partial_quality'],read=False)
    updates=recovery.get('actual_new_optimizer_updates');boundary=readiness.get('continuation_boundary')or{}
    checkpoint=recovery['checkpoint'];parent=request['parent_checkpoint']['sha256']
    if (type(updates)is not int or updates<=0 or type(recovery.get('saved_global_step'))is not int
            or type(recovery.get('saved_stage_step'))is not int
            or slot['bytes']!=recovery['checkpoint_bytes']
            or report.get('schema')!='natlang.neuralese-native-sft-report/1'
            or report.get('step')!=recovery['saved_stage_step']
            or report.get('checkpoint')!=slot['path'] or report.get('checkpoint_sha256')!=checkpoint['sha256']
            or (report.get('continuation')or{}).get('checkpoint_sha256')!=parent
            or boundary.get('new_optimizer_updates')!=updates
            or type(boundary.get('reached'))is not bool
            or (not operational and boundary['reached']is not True)
            or boundary.get('checkpoint_sha256')!=checkpoint['sha256']
            or (readiness.get('checkpoint')or{}).get('sha256')!=checkpoint['sha256']
            or (certificate.get('native_identity')or{}).get('checkpoint_sha256')!=checkpoint['sha256']):
        raise ValueError('saved-update recovery lost its actual saved parent/update/identity evidence')
    progress=artifact(binding(Path(slot['path']).with_name('progress.json')))
    if progress.get('global_step')!=recovery['saved_global_step'] or progress.get('step')!=recovery['saved_stage_step']:
        raise ValueError('saved-update recovery counters differ from the failed operation inventory')
    if operational:
        resident = interrupted.verify_controls(recovery,receipt=readiness,report=report,
            certificate=certificate,spec=spec,intent=intent,identities=identities,
            terminal=terminal,parent=parent)
        _verify_reached_operational_boundary(recovery,boundary,updates)
    else:
        resident = None
        _saved_update_memory_controls(recovery,artifact,slot,parent,spec,report)
    record={'schema':SAVED_UPDATE_RECOVERY,'at':recovery['at'],'intent':recovery['captured_owner_intent'],
        'checkpoint':checkpoint,'checkpoint_pin':{'original_slot':recovery['original_slot'],
            'bytes':recovery['checkpoint_bytes'],'original_inode':recovery['original_slot_inode'],
            'pin_inode':recovery['independent_pin_inode']},'resident':resident,'recovery':reference,
        'original_failed_result':recovery['actual_failed_result'],
        'outcome':{'status':'failed','returncode':terminal['returncode'],'recovered_optimizer_updates':updates,
                   'communication_runtime_qualified':certificate.get('passed')is True}}
    _checkpoint_pin_controls(record)
    return record,identities


def _saved_update_memory_controls(recovery,artifact,slot,parent,spec,report):
    from ..train.native_resident import code_inventory
    from ..common.hashing import canonical_json_sha256_hex as digest
    wrapper=artifact(binding(Path(slot['path']).with_name('memory-qualification.json')))
    proof=artifact(wrapper['proof']);prepared=artifact(wrapper['request'])
    observation=json.loads(_bound(recovery['saved_frame_observation']).read_text())
    remembered=observation.get('current_memory_qualification',observation.get('memory_qualification'))
    if (wrapper.get('passed')is not True or proof.get('passed')is not True
            or wrapper.get('parent_checkpoint_sha256')!=parent
            or any(proof.get(key)!=wrapper.get(key)for key in('parent_checkpoint_sha256','request'))
            or proof.get('runtime_code_sha256')!=prepared.get('runtime_code_sha256')
            or proof.get('identity')!=prepared.get('identity')
            or proof.get('runtime_code_sha256')!=digest(code_inventory(Path(spec['runtime'])/'training/neuralese/natlang_neuralese'))
            or proof.get('identity',{}).get('model_config')!=report.get('model_config')
            or remembered is not None and (remembered.get('passed')is not True
                or any(remembered.get(key)!=wrapper.get(key)for key in('parent_checkpoint_sha256','request')))):
        raise ValueError('saved-update recovery lost its actual physical memory qualification')


def prepare_saved_update_acknowledgement(recovery,receipt,out):
    """Record actual dead-owner observations on the owning host; no tensor load."""
    record,identities=_saved_update_recovery_controls(recovery,receipt)
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    if any(item['boot_id']!=boot for item in identities):
        raise ValueError('saved-update recovery drain observations must be captured on the original source host')
    observations=[observe(item['pid'])for item in identities]
    if any(same_process(old,new)and new['state']not in('Z','X')for old,new in zip(identities,observations)):
        raise ValueError('saved-update recovery owner or captured child remains alive')
    from . import native_interrupted_update as interrupted
    recovery_value = json.loads(_bound(recovery).read_text())
    payload_verification = None
    if recovery_value['schema'] == interrupted.SCHEMA:
        if Path(out).exists():
            payload_verification = json.loads(Path(out).read_text()).get('payload_verification')
            interrupted.verify_payload_evidence(payload_verification, record)
        else:
            payload_verification = interrupted.verify_checkpoint_pin(recovery_value)
    value={'schema':SAVED_UPDATE_ACK,'at':time.time(),'recovery':recovery,'receipt':receipt,
           'observations':observations,'source_boot_id':boot,'outcome':record['outcome']}
    if payload_verification is not None:value['payload_verification'] = payload_verification
    out=Path(out)
    if out.exists():
        existing=json.loads(out.read_text())
        if existing.get('recovery')!=recovery or existing.get('receipt')!=receipt:
            raise ValueError('saved-update acknowledgement belongs to another recovery')
    else:write_json_durable(out,value)
    verify_saved_update_acknowledgement(binding(out),receipt=receipt)
    return binding(out)


def verify_saved_update_acknowledgement(reference,*,receipt):
    value=json.loads(_bound(reference).read_text())
    if value.get('schema')!=SAVED_UPDATE_ACK or value.get('receipt')!=receipt:
        raise ValueError('saved-update acknowledgement differs from its original training receipt')
    record,identities=_saved_update_recovery_controls(value['recovery'],receipt)
    from . import native_interrupted_update as interrupted
    if json.loads(_bound(value['recovery']).read_text())['schema'] == interrupted.SCHEMA:
        interrupted.verify_payload_evidence(value.get('payload_verification'), record)
    observations=value.get('observations');boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    if (not isinstance(observations,list)or len(observations)!=len(identities)
            or type(value.get('at'))not in(int,float)or value['at']<record['at']
            or value.get('outcome')!=record['outcome']
            or any(old['boot_id']!=value.get('source_boot_id')for old in identities)):
        raise ValueError('saved-update acknowledgement lost its actual source-host drain evidence')
    for old,new in zip(identities,observations):
        if (new is not None and (new.get('pid')!=old['pid']or same_process(old,new)and new.get('state')not in('Z','X'))
                or old['boot_id']==boot and same_process(old,observe(old['pid']))and (observe(old['pid'])or{}).get('state')not in('Z','X')):
            raise ValueError('saved-update acknowledgement includes a live original owner')
    return record


def verify_handoff(reference, *, requirement=None):
    record=verify_handoff_metadata(reference,requirement=requirement)
    intent=json.loads(_bound(record['intent']).read_text())
    identities=[intent['owner'],*intent['children']]
    checkpoint=_bound(record['checkpoint'])
    # Optimizer/RNG/schema checks are shared with the real actor factory; do not
    # construct a tokenizer/model just to inspect a handoff checkpoint.
    from ..train.checkpoint_policy import CheckpointPolicy
    from .native_raw_policy import require_raw_training
    saved=CheckpointPolicy.load(checkpoint);training=require_raw_training(saved)
    if not isinstance(training.get('sft_progress'),dict):
        raise ValueError('raw handoff must preserve full actual model/optimizer/RNG/recipe exposure')
    resident=record.get('resident')
    if resident:
        if resident.get('current'):
            if not resident.get('initial'):raise ValueError('resident handoff omitted its actual optimizer owner')
            state=json.loads(_bound(resident['current']).read_text())
            if (state.get('generation')!=resident['generation'] or
                    (training.get('options',{}).get('resident_adoption') or {}).get('generation')!=resident['generation']
                    or (training.get('options',{}).get('resident_adoption') or {}).get('history')!=state.get('history')
                    or training.get('global_step',training.get('step',-1))<state['at_global_step']
                    or (training.get('source_receipts') or [{}])[0].get('sha256')!=state['payload']['train']['sha256']):
                raise ValueError('raw handoff checkpoint does not contain the exact current resident generation')
        if resident.get('initial'):
            initial=json.loads(_bound(resident['initial']).read_text())
            actual=initial.get('owner') or {}
            if not actual or not any(all(actual.get(key)==item[key] for key in ('pid','boot_id','start_ticks')) for item in identities):
                raise ValueError('raw handoff resident optimizer owner differs from the drained process tree')
    return record,saved


RECIPE_HANDOFF='natlang.native-raw-drained-recipe-handoff/2'
LEGACY_RECIPE_HANDOFF='natlang.native-raw-drained-recipe-handoff/1'
EXTERNAL_RECIPE_HANDOFF='natlang.native-raw-drained-recipe-handoff/3'
RECOVERED_RECIPE_HANDOFF='natlang.native-raw-drained-recipe-handoff/4'


def _recipe_plan(recipe_path, device):
    from ..train.recipe import load_recipe, resolve_stage_inputs, _binding_files, recipe_identity_sha256
    recipe=load_recipe(recipe_path)
    stages={stage['id']:({} if stage.get('admitted') is False else
        resolve_stage_inputs(recipe,stage,{'records':None,'pieces':None,'text_data':None}))
        for stage in recipe['stages']}
    inputs={item['path']:item['sha256'] for roles in stages.values()
        for value in roles.values() for item in _binding_files(value)}
    return {'schema':'natlang.neuralese-recipe-plan/1','recipe':recipe,
        'recipe_sha256':recipe_identity_sha256(recipe_path),'device':device,
        'stage_inputs':stages,'inputs':inputs}


def _stable_handoff_plan(plan):
    import copy
    value=copy.deepcopy(plan)
    # The existing verifier stamps the wall clock and its archive directory.
    # All identities, changes and prior inputs remain compared exactly.
    for entry in value.get('handoffs',[])[-1:]:
        entry.pop('time',None);entry.pop('superseded_runtime',None)
    return value


def _drained_plan_authority(record, saved):
    """Select actual adopted or initial-only authority, without manufacturing either."""
    from ..train.native_resident import digest
    evidence=record.get('resident') or {}
    if evidence.get('current'):
        state=json.loads(_bound(evidence['current']).read_text())
        return state,state['payload']['plan'],state['payload']['stage_id']
    if not evidence.get('initial') or not evidence.get('plan'):
        raise ValueError('drained recipe handoff requires its original resident or initial plan')
    initial=json.loads(_bound(evidence['initial']).read_text())
    plan=json.loads(_bound(evidence['plan']).read_text())
    training=saved['training'];options=training.get('options') or {}
    recipe=options.get('recipe_binding')
    if isinstance(recipe,str):recipe=json.loads(recipe)
    stage=initial.get('stage_id');inputs=plan.get('stage_inputs',{}).get(stage,{})
    receipts=training.get('source_receipts') or []
    if (digest(plan)!=initial.get('plan_sha256') or recipe!=initial.get('recipe')
            or not recipe or recipe.get('lineage_sha256',recipe.get('sha256'))!=plan.get('recipe_sha256')
            or options.get('stage_id')!=stage or len(receipts)<2
            or any(receipts[i].get('sha256')!=inputs.get(key,{}).get('sha256')
                   for i,key in enumerate(('native_graphs','held_native_graphs')))):
        raise ValueError('drained initial plan differs from the actual saved recipe/stage/sources')
    return None,plan,stage


def _external_frozen_package(record, package, old_plan, snapshots=None, *, executed_package=None, saved=None):
    """An external immutable runtime must be the actual captured launch runtime."""
    from .native_runtime_handoff import _owned_launch_chain, _read
    from ..train.native_resident import code_inventory
    _, _, references = _owned_launch_chain(record, snapshots)
    intent = _read((snapshots or references)['intent'])
    paths = (intent.get('env') or {}).get('PYTHONPATH', '').split(':')
    selected = next((Path(path)/'natlang_neuralese' for path in paths
                     if path and (Path(path)/'natlang_neuralese').is_dir()), None)
    executed = Path(executed_package or package).resolve()
    if (selected is None or selected.resolve() != executed
            or code_inventory(package) != old_plan['code']):
        raise ValueError('external resident runtime differs from its actual captured import path/code')
    if executed_package is not None:
        _verify_recovered_execution(record, executed, saved, snapshots=snapshots)
    return references



def _recovered_capacity_cases(request, proof, *, code, parent):
    """Validate the same numerical and whole-graph evidence in both formats."""
    if (request.get('schema') != 'natlang.native-memory-qualification-request/1'
            or request.get('runtime_code_sha256') != code
            or proof.get('schema') != 'natlang.native-memory-qualification/1'
            or proof.get('passed') is not True or type(proof.get('optimizer_updates')) is not int
            or proof['optimizer_updates'] != 0
            or proof.get('parent_checkpoint_sha256') != parent
            or proof.get('runtime_code_sha256') != code
            or proof.get('identity') != request.get('identity')
            or (proof.get('numerical_reference') or {}).get('passed') is not True
            or (proof.get('capacity') or {}).get('passed') is not True):
        raise ValueError('recovered runtime lacks an actual passed numerical/capacity case')
    rows = (proof.get('capacity') or {}).get('cases')
    if (not isinstance(rows, list) or not rows
            or any(not isinstance(row, dict) or row.get('status') != 'passed'
                   or any(type(row.get(k)) is not int or row[k] < 1
                          for k in ('forward_tokens', 'kv_slots')) for row in rows)):
        raise ValueError('recovered runtime requires complete successful whole-graph cases')
    return rows


def _verify_recovered_execution(record, package, saved, *, snapshots=None):
    """Bind a recovered runtime to its real launch and saved capacity evidence."""
    from .native_runtime_handoff import _owned_launch_chain
    from .native_raw_policy import require_raw_training
    from ..train.native_resident import code_inventory, digest
    if saved is None:
        raise ValueError('recovered runtime requires its verified full checkpoint')
    _, argv, _ = _owned_launch_chain(record, snapshots)
    options = require_raw_training(saved).get('options') or {}
    qualification = options.get('memory_qualification') or {}
    reference = qualification.get('request')
    parent = (options.get('continuation') or {}).get('checkpoint_sha256')
    def argument(flag):
        if argv.count(flag) != 1 or argv.index(flag)+1 >= len(argv):
            raise ValueError('recovered runtime requires one captured '+flag)
        return argv[argv.index(flag)+1]
    def same_binding(left, right):
        if not isinstance(left,dict) or not isinstance(right,dict):
            return False
        for ref in (left,right):
            path = _bound(ref)
            if 'bytes' in ref and (type(ref['bytes']) is not int or path.stat().st_size != ref['bytes']):
                raise ValueError('recovered runtime artifact byte count changed')
        return all(left.get(key)==right.get(key) for key in ('path','sha256'))
    requested = json.loads(argument('--memory-qualification'))
    requested = [requested] if isinstance(requested, dict) else requested
    if isinstance(reference,list):
        return _verify_recovered_cohort(package,options,qualification,argv,requested,same_binding)
    if (qualification.get('passed') is not True or not reference or not parent
            or qualification.get('parent_checkpoint_sha256') != parent
            or not isinstance(requested, list) or not any(same_binding(reference,ref) for ref in requested)):
        raise ValueError('recovered runtime lost its actual launch/parent qualification')
    request = json.loads(_bound(reference).read_text())
    code = digest(code_inventory(package))
    certificate = json.loads(Path(options.get('resident_geometry_qualification', '')).read_text())
    proof = json.loads(_bound(certificate['evidence']).read_text())
    if (request.get('schema') != 'natlang.native-memory-qualification-request/1'
            or request.get('runtime_code_sha256') != code
            or certificate.get('schema') != 'natlang.native-resident-geometry-qualification/1'
            or certificate.get('passed') is not True or certificate.get('runtime_code_sha256') != code
            or proof.get('schema') != 'natlang.native-memory-qualification/1'
            or proof.get('passed') is not True or proof.get('runtime_code_sha256') != code
            or not same_binding(proof.get('request'),reference) or proof.get('parent_checkpoint_sha256') != parent
            or certificate.get('identity') != request.get('identity')
            or proof.get('identity') != request.get('identity')):
        raise ValueError('recovered runtime code/capacity proof differs from actual execution')
    rows = _recovered_capacity_cases(request, proof, code=code, parent=parent)
    limits = {key: max(row[key] for row in rows) for key in ('forward_tokens', 'kv_slots')}
    if certificate.get('limits') != limits:
        raise ValueError('recovered capacity certificate differs from its measured cases')
    return {'package': str(Path(package).resolve()), 'code': code,
            'qualification': qualification, 'capacity_certificate': binding(Path(options['resident_geometry_qualification']))}



def _verify_recovered_cohort(package, options, qualification, argv, requested, same_binding):
    """Verify the real trainer's complete request/alias/reference/capacity cohort."""
    from ..train.native_resident import code_inventory, digest
    parent=(options.get('continuation') or {}).get('checkpoint_sha256')
    references=qualification['request']
    def equal_bindings(left,right):
        return (isinstance(left,list) and isinstance(right,list) and len(left)==len(right)
                and all(same_binding(a,b) for a,b in zip(left,right)))
    if (qualification.get('passed') is not True or not parent
            or qualification.get('parent_checkpoint_sha256')!=parent
            or not references or not equal_bindings(references,requested)
            or argv.count('--out')!=1 or argv.index('--out')+1>=len(argv)):
        raise ValueError('recovered cohort lost its actual launch/parent/request inventory')
    output=Path(argv[argv.index('--out')+1])
    summary_path=output/'memory-qualification.json'
    summary=json.loads(summary_path.read_text())
    checks=summary.get('checks');aliases=summary.get('request_aliases')
    if (summary.get('schema')!='natlang.native-memory-cohort-qualification/1'
            or summary.get('passed') is not True or summary.get('optimizer_updates')!=0
            or summary.get('parent_checkpoint_sha256')!=parent
            or not equal_bindings(summary.get('request'),references)
            or not isinstance(checks,list) or len(checks)!=len(references)
            or not isinstance(aliases,list) or len(aliases)!=len(references)
            or summary.get('retained_capacity')!=options.get('resident_geometry_qualification')):
        raise ValueError('recovered cohort summary differs from its saved complete qualification')
    code=digest(code_inventory(package));proofs={};cases=[]
    for reference,alias,check in zip(references,aliases,checks):
        resolved=alias.get('resolved_binding');canonical=alias.get('canonical_binding')
        if (not same_binding(alias.get('requested_binding'),reference)
                or not same_binding(check.get('request'),canonical)
                or _bound(resolved).read_bytes()!=_bound(canonical).read_bytes()):
            raise ValueError('recovered cohort lost its exact resolved/canonical request aliases')
        request=json.loads(_bound(canonical).read_text())
        rows=_recovered_capacity_cases(request,check,code=code,parent=parent)
        proof_path=output/'memory-qualifications'/canonical['sha256']/parent/'proof.json'
        if json.loads(proof_path.read_text())!=check:
            raise ValueError('recovered cohort summary changed its original per-request proof')
        if canonical['sha256'] not in proofs:
            proofs[canonical['sha256']]=binding(proof_path);cases.extend(rows)
    certificate_path=Path(options['resident_geometry_qualification'])
    certificate=json.loads(certificate_path.read_text());evidence=json.loads(_bound(certificate['evidence']).read_text())
    limits={k:max(row[k]for row in cases)for k in ('forward_tokens','kv_slots')}
    if (certificate.get('schema')!='natlang.native-resident-geometry-qualification/1'
            or certificate.get('passed') is not True or certificate.get('runtime_code_sha256')!=code
            or certificate.get('identity')!=checks[0]['identity'] or certificate.get('limits')!=limits):
        raise ValueError('recovered capacity certificate differs from its complete measured cohort')
    if len(proofs)==1:
        if not same_binding(certificate['evidence'],next(iter(proofs.values()))):
            raise ValueError('recovered single-cohort capacity lost its actual proof')
    elif (evidence.get('schema')!='natlang.native-memory-combined-capacity/1'
            or evidence.get('passed')is not True or evidence.get('optimizer_updates')!=0
            or evidence.get('runtime_code_sha256')!=code or evidence.get('parent_checkpoint_sha256')!=parent
            or evidence.get('identity')!=checks[0]['identity']
            or not equal_bindings(evidence.get('proofs'),list(proofs.values()))
            or (evidence.get('capacity')or{}).get('cases')!=cases):
        raise ValueError('recovered combined capacity lost its complete original proof union')
    return {'package':str(Path(package).resolve()),'code':code,'qualification':qualification,
            'capacity_certificate':binding(certificate_path),'cohort_summary':binding(summary_path)}


def _external_recipe_plans(record, old_plan, recipes):
    """Reconstruct metadata with the actual captured historical implementation.

    A changed classifier reader can legitimately enumerate different code
    dependencies. Never reinterpret its historical plan with today's reader or
    treat different reader bytes as a path relocation. No checkpoint is loaded.
    """
    import os
    import subprocess
    import sys
    from .native_runtime_handoff import _owned_launch_chain, _read
    from ..train.native_resident import code_inventory
    continuation, _, references = _owned_launch_chain(record)
    intent = _read(references['intent'])
    paths = (intent.get('env') or {}).get('PYTHONPATH', '').split(':')
    package = next((Path(path)/'natlang_neuralese' for path in paths
                    if path and (Path(path)/'natlang_neuralese').is_dir()), None)
    if package is None:
        raise ValueError('historical recipe reconstruction lost its captured runtime')
    if continuation.get('schema') == 'natlang.native-observed-bridge-dispatch/1':
        package = Path(continuation['historical_package'])
        if code_inventory(package) != old_plan['code']:
            raise ValueError('observed recovery changed its historical plan reader')
    else:
        _external_frozen_package(record, package, old_plan)
    return _package_recipe_plans(package, old_plan['code'], recipes, old_plan['device'])


def _package_recipe_plans(package, code, recipes, device):
    from ..common.native_metadata_replay import package_metadata_replay
    recipes=[{'path':ref['path'],'sha256':ref.get('file_sha256',ref['sha256']),
        **({'bytes':ref['bytes']}if'bytes'in ref else{})}for ref in recipes]
    plans = package_metadata_replay(package, code, 'recipe-plans',
        {'device': device, 'recipes': recipes}, recipes)
    if (not isinstance(plans, list) or len(plans) != len(recipes)
            or any(not isinstance(plan,dict)or plan.get('schema')!='natlang.neuralese-recipe-plan/1'
                   or plan.get('device')!=device for plan in plans)):
        raise ValueError('historical recipe reconstruction lost its exact inventory')
    return plans



def _external_started_stage_layout(record, saved, old_plan, stage_id, *, snapshots=None, receipt_snapshot=None):
    """Authenticate a real external owner's output without inventing recipe-runner markers."""
    from .native_runtime_handoff import _owned_launch_chain
    from ..train.native_continuation import collect_sft_progress, progress_obligations
    from ..train.native_operation_report import compact_obligations
    stages=[stage for stage in old_plan['recipe']['stages']if stage['id']==stage_id]
    _,argv,_=_owned_launch_chain(record,snapshots)
    def argument(flag):
        if argv.count(flag)!=1 or argv.index(flag)+1>=len(argv):
            raise ValueError('external started stage needs its exact captured '+flag)
        return argv[argv.index(flag)+1]
    root=Path(record['resident']['directory']).resolve().parent
    output=Path(argument('--out')).resolve(strict=True)
    training=saved['training'];options=training.get('options')or{}
    if (len(stages)!=1 or stages[0]['kind']!='native_state_sft'
            or argument('--stage-id')!=stage_id or options.get('stage_id')!=stage_id
            or not output.is_dir()or not output.is_relative_to(root)
            or Path(record['checkpoint_pin']['original_slot']['path']).resolve()!=output/'checkpoint.pt'
            or type(training.get('global_step'))is not int or training['global_step']<=0
            or not any(value>0 for value in training.get('exposure_counts',{}).values())):
        raise ValueError('external started-stage output differs from its real saved/captured work')
    source=output/'native-training-receipt.json'
    reference=receipt_snapshot or binding(source)
    receipt=json.loads(_bound(reference).read_text());checkpoint=record['checkpoint']
    actual_recipe=options.get('recipe_binding')
    if isinstance(actual_recipe,str):actual_recipe=json.loads(actual_recipe)
    mandatory=compact_obligations(progress_obligations(collect_sft_progress(training,borrow_frames=True)),
        {'path':str(output/'checkpoint.pt'),'sha256':checkpoint['sha256'],
         'bytes':record['checkpoint_pin']['bytes']})
    model=saved['model_config'];model_id=model['architecture']+':'+model['base_inventory_sha256']
    if (receipt.get('schema')!='natlang.native-training-readiness/1'
            or receipt.get('stage_id')!=stage_id or receipt.get('lineage')!=str(output)
            or receipt.get('model_id')!=model_id or receipt.get('recipe')!=actual_recipe
            or receipt.get('checkpoint',{}).get('sha256')!=checkpoint['sha256']
            or Path(receipt.get('checkpoint',{}).get('path','')).resolve()!=output/'checkpoint.pt'
            or receipt.get('native_identity',{}).get('checkpoint_sha256')!=checkpoint['sha256']
            or receipt.get('mandatory_sft')!=mandatory):
        raise ValueError('external started-stage receipt differs from its actual checkpoint/stage/exposure')
    return {'stage_id':stage_id,'output_directory':str(output),'receipt':reference,
            'global_step':training['global_step'],'started_stage_ids':[stage_id]}

def prepare_drained_recipe_handoff(handoff, recipe, out, *, package=None, external_package=None, executed_package=None, verified_handoff=None):
    """Persist an existing verified lineage handoff, never a resident rewrite."""
    from ..train import native_resident as resident
    from ..train.recipe import lineage_handoff
    from .native_runtime_handoff import _loaded_handoff
    record,saved=(verify_handoff(handoff) if verified_handoff is None else
                  _loaded_handoff(handoff,verified_handoff))
    loaded=(record,saved)
    if executed_package is not None and external_package is None:
        raise ValueError('recovered execution needs its separate historical external package')
    state,old_plan,stage_id=_drained_plan_authority(record,saved);del saved
    directory=Path(record['resident']['directory'])
    package=Path(package or Path(__file__).resolve().parents[1]).resolve()
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():
        reference=binding(out);value=verify_drained_recipe_handoff(reference,handoff=handoff,verified_handoff=loaded)
        if binding(recipe)!=value['recipe']:raise ValueError('drained recipe handoff recipe changed')
        if (bool(value.get('external_runtime')) != bool(external_package)
                or external_package and Path(external_package).resolve()!=Path(value['old_package']).resolve()
                or (str(Path(executed_package).resolve()) if executed_package else None) !=
                    ((value.get('external_runtime') or {}).get('executed_runtime') or {}).get('package')):
            raise ValueError('drained recipe handoff changed its external runtime layout')
        if resident.code_inventory(package)!=value['new_code']:
            raise ValueError('drained recipe handoff requested runtime changed')
        return reference
    with resident.locked(directory):
        if resident.current(directory) is not None:
            raise ValueError('retire the saved resident generation before recipe handoff')
        initial=json.loads((directory/'initial.json').read_text())
        if resident.owner_alive(initial.get('owner') or {}):
            raise ValueError('drained recipe handoff has a live optimizer owner')
        retired=None
        if state is not None:
            retired=directory/'retired'/(state['generation']+'.json')
            retirement=json.loads(retired.read_text())
            if ({k:v for k,v in retirement.items() if k not in ('checkpoint','terminal')}!=state
                    or retirement.get('checkpoint')!=record['checkpoint']
                    or retirement.get('terminal',{}).get('sha256')!=record['terminal']['sha256']):
                raise ValueError('drained recipe handoff retirement differs from original owner/checkpoint')
        elif initial!=json.loads(_bound(record['resident']['initial']).read_text()):
            raise ValueError('drained recipe handoff initial owner changed before transition')
        from ..train.recipe import stage_started
        layout=None
        if not stage_started(directory.parent,stage_id):
            if not external_package:raise ValueError('drained recipe handoff lost the actual started-stage boundary')
            layout=_external_started_stage_layout(record,loaded[1],old_plan,stage_id)
        started=tuple(layout['started_stage_ids'])if layout else ()
        plan_path=directory.parent/'recipe-plan.json'
        proposed=_recipe_plan(_bound(binding(recipe)),old_plan['device'])
        frozen=(Path(external_package).resolve() if external_package else
                directory.parent/'runtime/natlang_neuralese')
        external = None
        if external_package:
            refs = _external_frozen_package(record, frozen, old_plan, executed_package=executed_package, saved=loaded[1])
            from .native_runtime_handoff import _snap
            snapshots = out.parent/(out.stem+'-external-launch'); snapshots.mkdir(exist_ok=True)
            external = {'package': str(frozen), 'launch_control': {
                name: _snap(ref, snapshots, name+'.json') for name,ref in refs.items()}}
            if executed_package is not None:
                external['executed_runtime'] = _verify_recovered_execution(record, executed_package, loaded[1])
            if layout:
                layout['receipt']=_snap(layout['receipt'],snapshots,'started-stage-receipt.json')
                external['started_stage_layout']=layout
        # Validate first; the existing implementation owns runtime archival and
        # the durable plan commit, including priority/source continuation rules.
        old_copy=out.with_name(out.stem+'-old-plan.json')
        installed=json.loads(plan_path.read_text())
        if installed==old_plan:
            lineage_handoff(old_plan,proposed,frozen,package,plan_path,directory.parent,dry_run=True,started_stage_ids=started)
            write_json_durable(old_copy,old_plan)
            if external_package:
                # Validate through the same policy, then publish only its plan.
                # The actual qualified external package is never moved or copied.
                updated=lineage_handoff(old_plan,proposed,frozen,package,plan_path,directory.parent,dry_run=True,started_stage_ids=started)
                updated['handoffs'][-1]['superseded_runtime']=None
                write_json_durable(plan_path,updated)
            else:
                updated=lineage_handoff(old_plan,proposed,frozen,package,plan_path,directory.parent,started_stage_ids=started)
        else:
            # Recover the exact already committed handoff after interruption
            # between the existing plan commit and this receipt publication.
            if not old_copy.exists() or json.loads(old_copy.read_text())!=old_plan:
                raise ValueError('drained recipe handoff requires the final adopted original plan')
            previous=installed.get('handoffs',[{}])[-1].get('superseded_runtime')
            old_runtime=frozen if external_package else directory.parent/previous/'natlang_neuralese' if previous else frozen
            expected=lineage_handoff(old_plan,proposed,old_runtime,package,plan_path,directory.parent,dry_run=True,started_stage_ids=started)
            if _stable_handoff_plan(expected)!=_stable_handoff_plan(installed):
                raise ValueError('drained recipe handoff found another installed plan')
            updated=installed
        new_copy=out.with_name(out.stem+'-new-plan.json');write_json_durable(new_copy,updated)
        previous=updated['handoffs'][-1].get('superseded_runtime')
        old_package=frozen if external_package else (directory.parent/previous/'natlang_neuralese') if previous else package
        if resident.code_inventory(old_package)!=old_plan['code']:
            raise ValueError('drained recipe handoff lost immutable original runtime code')
        value={'schema':RECOVERED_RECIPE_HANDOFF if executed_package else EXTERNAL_RECIPE_HANDOFF if external_package else RECIPE_HANDOFF,'handoff':handoff,'checkpoint':record['checkpoint'],
            'retirement':binding(retired) if retired else None,'old_plan':binding(old_copy),'recipe':binding(recipe),
            'new_plan':binding(new_copy),'old_package':str(old_package.resolve()),
            'new_package':str(package),'old_code':resident.code_inventory(old_package),
            'new_code':resident.code_inventory(package),**({'external_runtime':external} if external else {})}
        write_json_durable(out,value)
    verify_drained_recipe_handoff(binding(out),handoff=handoff,verified_handoff=loaded)
    return binding(out)


def verify_drained_recipe_handoff(reference, *, handoff, verified_handoff=None):
    """Recompute the existing policy verifier; a self-consistent receipt is insufficient."""
    from ..train import native_resident as resident
    from ..train.recipe import lineage_handoff
    value=json.loads(_bound(reference).read_text())
    if value.get('schema') not in (RECIPE_HANDOFF,LEGACY_RECIPE_HANDOFF,EXTERNAL_RECIPE_HANDOFF,RECOVERED_RECIPE_HANDOFF) or value.get('handoff')!=handoff:
        raise ValueError('drained recipe handoff belongs to another original owner')
    record,saved=(verify_handoff(handoff) if verified_handoff is None else verified_handoff)
    if record!=json.loads(_bound(handoff).read_text()):
        raise ValueError('reused handoff verification belongs to another exact receipt')
    state,original_plan,actual_stage_id=_drained_plan_authority(record,saved)
    if value.get('checkpoint')!=record['checkpoint']:
        raise ValueError('drained recipe handoff changed its exact parent')
    if state is not None:
        retirement=json.loads(_bound(value['retirement']).read_text())
        if ({k:v for k,v in retirement.items() if k not in ('checkpoint','terminal')}!=state
                or retirement.get('checkpoint')!=record['checkpoint']
                or retirement.get('terminal',{}).get('sha256')!=record['terminal']['sha256']):
            raise ValueError('drained recipe handoff retirement changed')
    elif value.get('schema')not in(RECIPE_HANDOFF,EXTERNAL_RECIPE_HANDOFF,RECOVERED_RECIPE_HANDOFF) or value.get('retirement') is not None:
        raise ValueError('initial-only recipe transition requires explicit version2 proof')
    prior=json.loads(_bound(value['old_plan']).read_text())
    if prior!=original_plan:
        raise ValueError('drained recipe handoff lost its final original plan')
    old_package,new_package=Path(value['old_package']),Path(value['new_package'])
    external=value.get('external_runtime')
    if bool(external)!=(value.get('schema') in (EXTERNAL_RECIPE_HANDOFF,RECOVERED_RECIPE_HANDOFF)):
        raise ValueError('external resident runtime layout needs its explicit authority schema')
    if external:
        if external.get('package')!=str(old_package.resolve()):
            raise ValueError('external resident package binding changed')
        executed = external.get('executed_runtime')
        if bool(executed) != (value.get('schema') == RECOVERED_RECIPE_HANDOFF):
            raise ValueError('recovered runtime requires its explicit execution authority schema')
        _external_frozen_package(record,old_package,prior,external.get('launch_control'),
            executed_package=executed['package'] if executed else None,saved=saved)
        if executed and _verify_recovered_execution(record,executed['package'],saved,
                snapshots=external.get('launch_control')) != executed:
            raise ValueError('recovered runtime execution authority changed')
    layout=(external or {}).get('started_stage_layout');started=()
    if layout:
        if layout.get('stage_id')!=actual_stage_id:raise ValueError('external started-stage identity changed')
        actual=_external_started_stage_layout(record,saved,prior,layout['stage_id'],
            snapshots=external.get('launch_control'),receipt_snapshot=layout['receipt'])
        if actual!=layout:raise ValueError('external started-stage layout changed')
        started=tuple(actual['started_stage_ids'])
    if (resident.code_inventory(old_package)!=value['old_code'] or value['old_code']!=prior['code']
            or resident.code_inventory(new_package)!=value['new_code']):
        raise ValueError('drained recipe handoff runtime changed')
    next_plan=json.loads(_bound(value['new_plan']).read_text())
    proposed=_recipe_plan(_bound(value['recipe']),prior['device'])
    out=Path(record['resident']['directory']).parent
    try:
        expected=lineage_handoff(prior,proposed,old_package,new_package,out/'recipe-plan.json',out,dry_run=True,started_stage_ids=started)
    except ValueError:
        # Changed current readers can reject a historical input before the
        # final plan comparison. The exact frozen replay still must pass every
        # lineage check; no exception or dependency is waived.
        expected=None
    if (expected is None or _stable_handoff_plan(expected)!=_stable_handoff_plan(next_plan)
            or next_plan['code']!=value['new_code']):
        proposed=_package_recipe_plans(new_package,value['new_code'],[value['recipe']],prior['device'])[0]
        expected=lineage_handoff(prior,proposed,old_package,new_package,out/'recipe-plan.json',out,dry_run=True,started_stage_ids=started)
        if (_stable_handoff_plan(expected)!=_stable_handoff_plan(next_plan)
                or next_plan['code']!=value['new_code']):
            raise ValueError('drained recipe handoff plan differs from verified continuation')
    return value


def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser(description='Verify and persist a drained recipe/runtime handoff without model updates.')
    parser.add_argument('--handoff',required=True,help='JSON binding of the original immutable optimizer handoff')
    parser.add_argument('--recipe',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--package',type=Path)
    parser.add_argument('--external-package',type=Path,help='Exact captured external immutable old package; never installed or moved')
    parser.add_argument('--executed-package',type=Path,help='Actual qualified recovery runtime, separately bound from the historical plan package')
    args=parser.parse_args(argv)
    print(json.dumps(prepare_drained_recipe_handoff(json.loads(args.handoff),args.recipe,args.output,package=args.package,external_package=args.external_package,executed_package=args.executed_package)))


if __name__=='__main__':main()
