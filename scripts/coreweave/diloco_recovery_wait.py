#!/usr/bin/env python3
"""Wait for the exact qualified local parent and all owned learners before recovery supervision."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time


def ready(record_path,pin_path,parent_path,copy_receipt,qualification):
    """Read-only check; the recovery operator independently rechecks full identities."""
    try:
        record=json.loads(Path(record_path).read_text());pin=json.loads(Path(pin_path).read_text())
        copied=json.loads(Path(copy_receipt).read_text());qualified=json.loads(Path(qualification).read_text())
        workers=record['learners'];local=copied['local_copy'];parent=Path(parent_path)
        if (record.get('state')!='running' or len(workers)!=4 or any(not x for x in workers)
                or len(set(workers))!=4 or record['coordinator']!=workers[0]):return False
        ownership=record.get('ownership') or {}
        if ('coordinator' not in ownership.get(workers[0],[]) or
                any('learner-'+str(i) not in ownership.get(worker,[]) for i,worker in enumerate(workers))):return False
        if (not parent.is_file() or parent.stat().st_size!=pin['bytes'] or copied.get('sha256')!=pin['sha256']
                or copied.get('bytes')!=pin['bytes'] or local.get('committed') is not True
                or Path(local.get('path','')).resolve()!=parent.resolve()):return False
        if (qualified.get('schema')!='natlang.lfm3-seed-independent-full-state/1' or qualified.get('passed') is not True
                or qualified.get('sha256')!=pin['sha256'] or qualified.get('bytes')!=pin['bytes']
                or qualified.get('global_step')!=pin['global_step']
                or Path(qualified.get('checkpoint','')).resolve()!=parent.resolve()):return False
        return True
    except (OSError,ValueError,KeyError,TypeError):return False


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run_id');parser.add_argument('--state',required=True)
    parser.add_argument('--storage',required=True);parser.add_argument('--parent-pin',required=True);parser.add_argument('--parent-local',required=True)
    parser.add_argument('--parent-copy-receipt',required=True);parser.add_argument('--parent-qualification',required=True)
    parser.add_argument('--interval',type=float,default=15);parser.add_argument('--backup-interval',type=float,default=1800)
    parser.add_argument('--ownership-journal',type=Path,help='actual worker process and runtime custody journal')
    args=parser.parse_args()
    if args.interval<=0 or args.backup_interval<=0:parser.error('intervals must be positive')
    from scripts.coreweave import cwctl
    record=cwctl.STATE/'diloco'/(args.run_id+'.json')
    while not ready(record,args.parent_pin,args.parent_local,args.parent_copy_receipt,args.parent_qualification):
        print(json.dumps({'event':'recovery_prerequisites_wait','run_id':args.run_id}),flush=True);time.sleep(args.interval)
    command=[sys.executable,str(Path(__file__).with_name('diloco_periodic_recovery.py')),args.run_id,
             '--state',args.state,'--storage',args.storage,'--parent-local',args.parent_local,
             '--parent-copy-receipt',args.parent_copy_receipt,'--parent-qualification',args.parent_qualification,
             '--interval',str(args.backup_interval)]
    if args.ownership_journal:
        command.extend(['--ownership-journal',str(args.ownership_journal)])
    os.execv(sys.executable,command)


if __name__=='__main__':main()
