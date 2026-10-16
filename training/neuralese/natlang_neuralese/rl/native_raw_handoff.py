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
            if child not in identities and observe(child) is not None:
                raise ValueError('raw handoff process tree grew during census; capture again')
    return [value for ident,value in identities.items() if ident!=owner['pid']],links


def capture_drain_intent(launch, out):
    """Call before the existing owner-safe signal, on the owning machine."""
    launch_path=_bound(launch);launch_bytes=launch_path.read_bytes();record=json.loads(launch_bytes)
    owner=observe(record['pid'])
    if owner is None or owner['state'] in ('Z','X'):
        raise ValueError('raw handoff needs its actual live original optimizer owner')
    argv=[part.decode() for part in bytes.fromhex(owner['argv_hex']).split(b'\0') if part]
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


def complete_drain(intent, terminal, checkpoint, out, *, original_slot, resident_directory=None):
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
    verify_handoff(binding(out))
    return binding(out)


def verify_handoff(reference, *, requirement=None):
    """Verify bound trusted supervisor evidence, not a generated model assertion."""
    record=json.loads(_bound(reference).read_text())
    if record.get('schema')!=SCHEMA:raise ValueError('raw bootstrap needs an exact original-owner handoff receipt')
    intent=json.loads(_bound(record['intent']).read_text())
    if intent.get('schema') not in (INTENT,LEGACY_INTENT) or not isinstance(intent.get('children'),list):
        raise ValueError('raw handoff needs its original pre-signal process tree')
    launch=json.loads(_bound(intent['launch']).read_text())
    if intent.get('source_launch',intent['launch'])['sha256']!=intent['launch']['sha256']:
        raise ValueError('raw handoff launch snapshot differs from original source')
    owner=intent.get('owner');identities=[owner,*intent['children']]
    if (not all(_identity(item) for item in identities) or launch.get('pid')!=owner['pid']
            or len({item['pid'] for item in identities})!=len(identities)
            or any(item['boot_id']!=owner['boot_id'] for item in identities)
            or [part.decode() for part in bytes.fromhex(owner['argv_hex']).split(b'\0') if part]!=launch.get('argv')):
        raise ValueError('raw handoff changed its exact boot/start/PID/argv owner binding')
    _verify_process_tree(intent)
    terminal=json.loads(_bound(record['terminal']).read_text())
    if record.get('source_terminal',record['terminal'])['sha256']!=record['terminal']['sha256']:
        raise ValueError('raw handoff terminal snapshot differs from original source')
    if (terminal.get('returncode')!=0 or any(type(item) not in (int,float) for item in
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
    checkpoint=_bound(record['checkpoint'])
    pin=record.get('checkpoint_pin') or {}
    if (pin.get('original_slot',{}).get('sha256')!=record['checkpoint']['sha256']
            or type(pin.get('bytes')) is not int or pin['bytes']!=checkpoint.stat().st_size
            or any(not isinstance(pin.get(key),list) or len(pin[key])!=2
                   or any(type(item) is not int or item<0 for item in pin[key]) for key in ('original_inode','pin_inode'))
            or pin['original_inode']==pin['pin_inode']):
        raise ValueError('raw handoff lost its exact independently pinned checkpoint evidence')
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


def prepare_drained_recipe_handoff(handoff, recipe, out, *, package=None):
    """Persist an existing verified lineage handoff, never a resident rewrite."""
    from ..train import native_resident as resident
    from ..train.recipe import lineage_handoff
    record,saved=verify_handoff(handoff)
    state,old_plan,stage_id=_drained_plan_authority(record,saved);del saved
    directory=Path(record['resident']['directory'])
    package=Path(package or Path(__file__).resolve().parents[1]).resolve()
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():
        reference=binding(out);value=verify_drained_recipe_handoff(reference,handoff=handoff)
        if binding(recipe)!=value['recipe']:raise ValueError('drained recipe handoff recipe changed')
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
        if not stage_started(directory.parent,stage_id):
            raise ValueError('drained recipe handoff lost the actual started-stage boundary')
        plan_path=directory.parent/'recipe-plan.json'
        proposed=_recipe_plan(_bound(binding(recipe)),old_plan['device'])
        frozen=directory.parent/'runtime/natlang_neuralese'
        # Validate first; the existing implementation owns runtime archival and
        # the durable plan commit, including priority/source continuation rules.
        old_copy=out.with_name(out.stem+'-old-plan.json')
        installed=json.loads(plan_path.read_text())
        if installed==old_plan:
            lineage_handoff(old_plan,proposed,frozen,package,plan_path,directory.parent,dry_run=True)
            write_json_durable(old_copy,old_plan)
            updated=lineage_handoff(old_plan,proposed,frozen,package,plan_path,directory.parent)
        else:
            # Recover the exact already committed handoff after interruption
            # between the existing plan commit and this receipt publication.
            if not old_copy.exists() or json.loads(old_copy.read_text())!=old_plan:
                raise ValueError('drained recipe handoff requires the final adopted original plan')
            previous=installed.get('handoffs',[{}])[-1].get('superseded_runtime')
            old_runtime=directory.parent/previous/'natlang_neuralese' if previous else frozen
            expected=lineage_handoff(old_plan,proposed,old_runtime,package,plan_path,directory.parent,dry_run=True)
            if _stable_handoff_plan(expected)!=_stable_handoff_plan(installed):
                raise ValueError('drained recipe handoff found another installed plan')
            updated=installed
        new_copy=out.with_name(out.stem+'-new-plan.json');write_json_durable(new_copy,updated)
        previous=updated['handoffs'][-1].get('superseded_runtime')
        old_package=(directory.parent/previous/'natlang_neuralese') if previous else package
        if resident.code_inventory(old_package)!=old_plan['code']:
            raise ValueError('drained recipe handoff lost immutable original runtime code')
        value={'schema':RECIPE_HANDOFF,'handoff':handoff,'checkpoint':record['checkpoint'],
            'retirement':binding(retired) if retired else None,'old_plan':binding(old_copy),'recipe':binding(recipe),
            'new_plan':binding(new_copy),'old_package':str(old_package.resolve()),
            'new_package':str(package),'old_code':resident.code_inventory(old_package),
            'new_code':resident.code_inventory(package)}
        write_json_durable(out,value)
    verify_drained_recipe_handoff(binding(out),handoff=handoff)
    return binding(out)


def verify_drained_recipe_handoff(reference, *, handoff):
    """Recompute the existing policy verifier; a self-consistent receipt is insufficient."""
    from ..train import native_resident as resident
    from ..train.recipe import lineage_handoff
    value=json.loads(_bound(reference).read_text())
    if value.get('schema') not in (RECIPE_HANDOFF,LEGACY_RECIPE_HANDOFF) or value.get('handoff')!=handoff:
        raise ValueError('drained recipe handoff belongs to another original owner')
    record,saved=verify_handoff(handoff)
    state,original_plan,_=_drained_plan_authority(record,saved);del saved
    if value.get('checkpoint')!=record['checkpoint']:
        raise ValueError('drained recipe handoff changed its exact parent')
    if state is not None:
        retirement=json.loads(_bound(value['retirement']).read_text())
        if ({k:v for k,v in retirement.items() if k not in ('checkpoint','terminal')}!=state
                or retirement.get('checkpoint')!=record['checkpoint']
                or retirement.get('terminal',{}).get('sha256')!=record['terminal']['sha256']):
            raise ValueError('drained recipe handoff retirement changed')
    elif value.get('schema')!=RECIPE_HANDOFF or value.get('retirement') is not None:
        raise ValueError('initial-only recipe transition requires explicit version2 proof')
    prior=json.loads(_bound(value['old_plan']).read_text())
    if prior!=original_plan:
        raise ValueError('drained recipe handoff lost its final original plan')
    old_package,new_package=Path(value['old_package']),Path(value['new_package'])
    if (resident.code_inventory(old_package)!=value['old_code'] or value['old_code']!=prior['code']
            or resident.code_inventory(new_package)!=value['new_code']):
        raise ValueError('drained recipe handoff runtime changed')
    next_plan=json.loads(_bound(value['new_plan']).read_text())
    proposed=_recipe_plan(_bound(value['recipe']),prior['device'])
    out=Path(record['resident']['directory']).parent
    expected=lineage_handoff(prior,proposed,old_package,new_package,out/'recipe-plan.json',out,dry_run=True)
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
    args=parser.parse_args(argv)
    print(json.dumps(prepare_drained_recipe_handoff(json.loads(args.handoff),args.recipe,args.output,package=args.package)))


if __name__=='__main__':main()
