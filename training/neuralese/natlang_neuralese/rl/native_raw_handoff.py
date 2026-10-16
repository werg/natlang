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
INTENT='natlang.native-raw-optimizer-drain/1'


def observe(pid):
    try:
        proc=Path('/proc',str(pid));fields=(proc/'stat').read_text().rsplit(')',1)[1].split()
        return {'pid':pid,'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                'start_ticks':fields[19],'state':fields[0],'argv_hex':(proc/'cmdline').read_bytes().hex()}
    except (FileNotFoundError,ProcessLookupError):return None


def same_process(expected,actual):
    return actual is not None and all(actual.get(key)==expected.get(key) for key in ('pid','boot_id','start_ticks'))


def capture_drain_intent(launch, out):
    """Call before the existing owner-safe signal, on the owning machine."""
    record=json.loads(_bound(launch).read_text());owner=observe(record['pid'])
    if owner is None or owner['state'] in ('Z','X'):
        raise ValueError('raw handoff needs its actual live original optimizer owner')
    argv=[part.decode() for part in bytes.fromhex(owner['argv_hex']).split(b'\0') if part]
    if argv!=record.get('argv'):
        raise ValueError('raw handoff launch argv differs from its actual process')
    children=[];todo=[owner['pid']];seen={owner['pid']}
    while todo:
        pid=todo.pop()
        # Linux assigns a child to the spawning thread, not necessarily
        # the process leader. Inspect every live TID and deduplicate PIDs.
        try:tasks=list(Path('/proc',str(pid),'task').iterdir())
        except (FileNotFoundError,ProcessLookupError):continue
        ids=set()
        for task in tasks:
            try:ids.update(int(value) for value in (task/'children').read_text().split())
            except (FileNotFoundError,ProcessLookupError):continue
        for ident in sorted(ids-seen):
            seen.add(ident);child=observe(ident)
            if child is not None:children.append(child);todo.append(child['pid'])
    launch_snapshot=Path(out).with_name(Path(out).stem+'-launch.json')
    write_bytes_durable(launch_snapshot,_bound(launch).read_bytes())
    write_json_durable(out,{'schema':INTENT,'at':time.time(),'source_launch':launch,
        'launch':binding(launch_snapshot),'owner':owner,'children':children})
    return binding(out)


def _identity(value):
    return (isinstance(value,dict) and type(value.get('pid')) is int and value['pid']>0
        and isinstance(value.get('boot_id'),str) and bool(value['boot_id'])
        and isinstance(value.get('start_ticks'),str) and value['start_ticks'].isdigit()
        and isinstance(value.get('argv_hex'),str) and bool(value['argv_hex'])
        and value.get('state') not in (None,'Z','X'))


def complete_drain(intent, terminal, checkpoint, out, *, original_slot, resident_directory=None):
    """After owner/children drain and checkpoint pin, persist exact observations."""
    value=json.loads(_bound(intent).read_text())
    if value.get('schema')!=INTENT:raise ValueError('raw handoff requires its exact pre-signal intent')
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
    write_json_durable(out,record)
    verify_handoff(binding(out))
    return binding(out)


def verify_handoff(reference, *, requirement=None):
    """Verify bound trusted supervisor evidence, not a generated model assertion."""
    record=json.loads(_bound(reference).read_text())
    if record.get('schema')!=SCHEMA:raise ValueError('raw bootstrap needs an exact original-owner handoff receipt')
    intent=json.loads(_bound(record['intent']).read_text())
    if intent.get('schema')!=INTENT or not isinstance(intent.get('children'),list):
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


RECIPE_HANDOFF='natlang.native-raw-drained-recipe-handoff/1'


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


def prepare_drained_recipe_handoff(handoff, recipe, out, *, package=None):
    """Persist an existing verified lineage handoff, never a resident rewrite."""
    from ..train import native_resident as resident
    from ..train.recipe import lineage_handoff
    record,saved=verify_handoff(handoff);del saved
    state_ref=(record.get('resident') or {}).get('current')
    if state_ref is None:raise ValueError('drained recipe handoff requires its original resident generation')
    state=json.loads(_bound(state_ref).read_text());directory=Path(record['resident']['directory'])
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
        retired=directory/'retired'/(state['generation']+'.json')
        retirement=json.loads(retired.read_text())
        if ({k:v for k,v in retirement.items() if k not in ('checkpoint','terminal')}!=state
                or retirement.get('checkpoint')!=record['checkpoint']
                or retirement.get('terminal',{}).get('sha256')!=record['terminal']['sha256']):
            raise ValueError('drained recipe handoff retirement differs from original owner/checkpoint')
        from ..train.recipe import stage_started
        if not stage_started(directory.parent,state['payload']['stage_id']):
            raise ValueError('drained recipe handoff lost the actual started-stage boundary')
        old_plan=state['payload']['plan'];plan_path=directory.parent/'recipe-plan.json'
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
        old_package=(directory.parent/previous/'natlang_neuralese') if previous else frozen
        value={'schema':RECIPE_HANDOFF,'handoff':handoff,'checkpoint':record['checkpoint'],
            'retirement':binding(retired),'old_plan':binding(old_copy),'recipe':binding(recipe),
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
    if value.get('schema')!=RECIPE_HANDOFF or value.get('handoff')!=handoff:
        raise ValueError('drained recipe handoff belongs to another original owner')
    record,saved=verify_handoff(handoff);del saved
    if value.get('checkpoint')!=record['checkpoint']:
        raise ValueError('drained recipe handoff changed its exact parent')
    state=json.loads(_bound(record['resident']['current']).read_text())
    retirement=json.loads(_bound(value['retirement']).read_text())
    if ({k:v for k,v in retirement.items() if k not in ('checkpoint','terminal')}!=state
            or retirement.get('checkpoint')!=record['checkpoint']
            or retirement.get('terminal',{}).get('sha256')!=record['terminal']['sha256']):
        raise ValueError('drained recipe handoff retirement changed')
    prior=json.loads(_bound(value['old_plan']).read_text())
    if prior!=state['payload']['plan']:
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
