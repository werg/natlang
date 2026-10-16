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
    children=[];todo=[owner['pid']]
    while todo:
        pid=todo.pop()
        try:ids=Path('/proc',str(pid),'task',str(pid),'children').read_text().split()
        except FileNotFoundError:continue
        for ident in ids:
            child=observe(int(ident))
            if child is not None:children.append(child);todo.append(child['pid'])
    write_json_durable(out,{'schema':INTENT,'at':time.time(),'launch':launch,'owner':owner,'children':children})
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
    record={'schema':SCHEMA,'at':time.time(),'intent':intent,'terminal':terminal,'checkpoint':checkpoint,
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
    owner=intent.get('owner');identities=[owner,*intent['children']]
    if (not all(_identity(item) for item in identities) or launch.get('pid')!=owner['pid']
            or len({item['pid'] for item in identities})!=len(identities)
            or any(item['boot_id']!=owner['boot_id'] for item in identities)
            or [part.decode() for part in bytes.fromhex(owner['argv_hex']).split(b'\0') if part]!=launch.get('argv')):
        raise ValueError('raw handoff changed its exact boot/start/PID/argv owner binding')
    terminal=json.loads(_bound(record['terminal']).read_text())
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
        if requirement.get('trainer_launch')!=intent['launch']:
            raise ValueError('raw handoff belongs to another declared original trainer launch')
        if Path(requirement['trainer_exit']).resolve()!=_bound(record['terminal']):
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
