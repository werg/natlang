"""Durable consumer registrations serialized with artifact retirement.

Managed consumers register the exact role before resolving or fetching it, and
release their same identity after the last use. Registrations do not expire;
recovery must explicitly release an abandoned consumer after verifying its end.
Existing owner GETs are separately protected by their lock-bound pending state.
"""
from __future__ import annotations
import fcntl,json
from pathlib import Path


def leased_roles(path):
    path=Path(path)
    if not path.exists():return set()
    state=json.loads(path.read_text())
    if state.get('schema')!='natlang.retention-consumer-leases/1' or not isinstance(state.get('leases'),list):raise ValueError('invalid consumer lease ledger')
    return {x['role'] for x in state['leases']}


def change_lease(*,lock_path,state_path,retirement_state_path,role,consumer_id,identity,release,write_durable):
    """Register before use; release explicitly after use, never guess expiry."""
    if not role or not consumer_id or not isinstance(identity,dict) or not identity:raise ValueError('incomplete consumer identity')
    parent=Path(retirement_state_path).parent.resolve()
    if Path(state_path).resolve()!=parent/'standalone-retention-consumer-leases.json' or Path(lock_path).resolve()!=parent/'standalone-cloud-retention.lock':raise ValueError('consumer lease must use the shared retention namespace')
    with Path(lock_path).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        state_path=Path(state_path);retirement_state_path=Path(retirement_state_path)
        state=json.loads(state_path.read_text()) if state_path.exists() else {'schema':'natlang.retention-consumer-leases/1','leases':[]}
        if state.get('schema')!='natlang.retention-consumer-leases/1' or not isinstance(state.get('leases'),list):raise ValueError('invalid consumer lease ledger')
        matches=[x for x in state['leases'] if x.get('role')==role and x.get('consumer_id')==consumer_id]
        if len(matches)>1:raise ValueError('duplicate consumer lease')
        if release:
            if len(matches)!=1 or matches[0].get('identity')!=identity:raise ValueError('consumer lease release identity differs')
            state['leases'].remove(matches[0]);write_durable(state_path,state);return
        retired=json.loads(retirement_state_path.read_text()) if retirement_state_path.exists() else {}
        if role in retired.get('retired_roles',[]) or any((op.get('role')==role or op.get('control_role')==role) and op.get('phase') in ('intent','control_deleted','deleted') for op in retired.get('operations',[])):
            raise ValueError('artifact retirement already fenced')
        value={'role':role,'consumer_id':consumer_id,'identity':identity}
        if matches:
            if matches[0]!=value:raise ValueError('consumer lease identity changed')
            return
        state['leases'].append(value);write_durable(state_path,state)


def main(argv=None):
    import argparse
    from natlang_neuralese.common.jsonio import write_json_durable
    parser=argparse.ArgumentParser(description='Register or release a managed artifact consumer before or after use')
    parser.add_argument('action',choices=('register','release'));parser.add_argument('--lock',type=Path,required=True);parser.add_argument('--leases',type=Path,required=True);parser.add_argument('--retirement-state',type=Path,required=True);parser.add_argument('--role',required=True);parser.add_argument('--consumer-id',required=True);parser.add_argument('--identity-json',required=True)
    args=parser.parse_args(argv);identity=json.loads(args.identity_json)
    change_lease(lock_path=args.lock,state_path=args.leases,retirement_state_path=args.retirement_state,role=args.role,consumer_id=args.consumer_id,identity=identity,release=args.action=='release',write_durable=write_json_durable)


if __name__=='__main__':main()
