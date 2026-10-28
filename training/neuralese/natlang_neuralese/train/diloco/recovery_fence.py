"""Bounded checkpoint fence for independently recoverable DiLoCo cohorts."""
from __future__ import annotations
import json
import os
from pathlib import Path
import re
import time
from ...common.jsonio import write_json_durable

SCHEMA='natlang.diloco-recovery-fence/1'


def request(out, worker, fingerprint, *, now=time.time):
    directory=Path(out)/'diloco/recovery-fence';path=directory/'request.json'
    if not path.is_file():return None
    value=json.loads(path.read_text());ident=value.get('id','')
    if (value.get('schema')!=SCHEMA or not re.fullmatch('[0-9a-f]{32}',ident)
            or value.get('worker')!=worker or value.get('fingerprint')!=fingerprint
            or not isinstance(value.get('deadline'),(int,float))):
        raise ValueError('invalid DiLoCo recovery fence request')
    if ((directory/(ident+'.complete.json')).exists() or (directory/(ident+'.failed.json')).exists()
            or value['deadline']<=now()):return None
    return value


def checkpoint_ready(out, value, checkpoint, *, global_step, local_step, worker_state):
    directory=Path(out)/'diloco/recovery-fence';checkpoint=Path(checkpoint)
    pin=directory/'pins'/value['id']/'checkpoint.pt';pin.parent.mkdir(parents=True,exist_ok=True)
    if pin.exists():raise ValueError('recovery fence pin already exists')
    os.link(checkpoint,pin)
    stat=pin.stat()
    ready={'schema':SCHEMA,'id':value['id'],'worker':value['worker'],'fingerprint':value['fingerprint'],
           'checkpoint':str(pin),'bytes':stat.st_size,'global_step':global_step,'local_step':local_step,
           'versions':worker_state['versions'],'sequences':worker_state['sequences'],
           'deadline':value['deadline']}
    write_json_durable(directory/(value['id']+'.ready.json'),ready)
    return ready


def wait_release(out, value, *, stopping=lambda:False, now=time.time, sleep=time.sleep):
    directory=Path(out)/'diloco/recovery-fence';release=directory/(value['id']+'.release.json')
    reason='deadline'
    while now()<value['deadline'] and not stopping():
        if release.is_file():
            entry=json.loads(release.read_text())
            if entry.get('schema')!=SCHEMA or entry.get('id')!=value['id']:
                raise ValueError('invalid DiLoCo recovery fence release')
            reason='released';break
        sleep(.2)
    if stopping():reason='trainer-stop'
    write_json_durable(directory/(value['id']+'.complete.json'),{'schema':SCHEMA,'id':value['id'],'reason':reason})
    return reason
