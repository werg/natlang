"""Shared retryable control hook for externally supervised native learners."""
from __future__ import annotations
import hashlib, json, subprocess
from pathlib import Path
try:
    from .native_remote_budget import bounded_remote
except ImportError:
    from native_remote_budget import bounded_remote


def attempt_control(pin, sandbox, entry, state, journal, selection, write_durable, *, presence, launch):
    """Derive, journal and request an independently verified compact control.

    Callers catch errors, durably record them and continue the full-checkpoint
    fallback. A prior control can use source-free GET after its sandbox is gone.
    """
    helper=Path(selection['helper']['path']);raw=helper.read_bytes();observed=hashlib.sha256(raw).hexdigest()
    if observed!=selection['helper']['sha256']:
        raise ValueError('control helper SHA expected '+selection['helper']['sha256']+' observed '+observed)
    compile(raw.decode(),'<remote-control>','exec')
    control=entry.setdefault('control',{'attempts':0})
    selected=selection['target_selection']['sha256']
    if control.get('selection_sha256') not in (None, selected):
        raise ValueError('control selection epoch differs from this checkpoint')
    if control.get('selection_sha256') is None:
        if control.get('attempts',0) or control.get('archive'):
            raise ValueError('historical control lacks an authenticated selection epoch')
        control['selection_sha256']=selected;write_durable(journal,state)
    if control.get('phase_path'):

        phase_path=Path(control['phase_path']);phase=json.loads(phase_path.read_text()).get('phase') if phase_path.exists() else None
        if phase in ('independently_verified','reused_independently_verified'):
            control['status']='independently_verified';write_durable(journal,state);return
        if subprocess.run(['systemctl','--user','is-active','--quiet',control['unit']]).returncode==0:return
    if not control.get('archive'):
        control['attempts']+=1;target=selection['control_directory']+'/step%d-attempt%d'%(pin['step'],control['attempts'])
        control['status']='derivation_requested';write_durable(journal,state)
        if sandbox is None:raise ValueError('historical source sandbox unavailable for control derivation')
        args=['python3','-c',bounded_remote(raw.decode()),'--checkpoint',pin['path'],'--checkpoint-sha256',pin['sha256'],'--step',str(pin['step']),'--out',target,'--selection',selection['target_selection']['path'],'--selection-sha256',selection['target_selection']['sha256'],'--helper-path',selection['target_helper']['path'],'--helper-sha256',selection['helper']['sha256']]
        result=sandbox.exec(args,timeout_seconds=selection.get('timeout_seconds',600)).result(timeout=selection.get('timeout_seconds',600)+30)
        control['collector_invoked']='NATLANG_CONTROL_COLLECTOR_INVOKED' in result.stdout;write_durable(journal,state)
        if result.returncode:raise RuntimeError('control derivation failed: '+result.stderr[-500:])
        if not control['collector_invoked']:raise ValueError('control collector was not invoked')
        value=json.loads(result.stdout.strip().splitlines()[-1])
        if value['checkpoint_sha256']!=pin['sha256']:raise ValueError('control checkpoint differs')
        control.update(archive=value['archive'],control=value['control'],status='derived');write_durable(journal,state)
    archive=control['archive'];control['attempts']+=1
    role=selection['role_prefix']+'-step%d-publication-control'%pin['step'];unit=role+'-export-'+str(control['attempts'])
    existing=presence(role,archive)
    if sandbox is None and not existing:raise ValueError('private control object absent and historical sandbox unavailable')
    control.update(role=role,unit=unit,phase_path=selection['phase_directory']+'/coreweave-'+role+'-'+Path(archive['path']).name+'-export-progress-20261014.json',status='export_requested');write_durable(journal,state)
    launch(role,unit,pin['sandbox_id'],archive,existing)
