#!/usr/bin/env python3
"""Read-only proof for reconciling legacy DiLoCo sandbox assignments."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cwctl
import diloco

PROBE = r'''
import json,os,pathlib,sys
job,worker,instance=sys.argv[1:]
root=pathlib.Path('/proc'); found=[]
boot=(root/'sys/kernel/random/boot_id').read_text().strip()
for proc in root.iterdir():
    if not proc.name.isdigit(): continue
    try:
        env=dict(x.split(b'=',1) for x in (proc/'environ').read_bytes().split(b'\0') if b'=' in x)
        cmd=(proc/'cmdline').read_bytes().split(b'\0')
        if (env.get(b'NATLANG_CW_JOB')!=job.encode() or
            env.get(b'NATLANG_DILOCO_WORKER_ID')!=worker.encode() or
            b'natlang_neuralese.train.native_state_sft' not in cmd): continue
        if instance and env.get(b'NATLANG_CW_INSTANCE')!=instance.encode():continue
        ticks=(proc/'stat').read_text().split(') ',1)[1].split()[19]
        found.append({'pid':int(proc.name),'start_ticks':ticks})
    except (OSError,ValueError,IndexError): continue
if len(found)!=1: raise SystemExit('expected exactly one matching live trainer')
marker=pathlib.Path('/work/.coreweave')
if instance: marker=marker/'instances'/instance
if not (marker/'inputs-ready').is_file(): raise SystemExit('input completion marker absent')
print(json.dumps({'boot_id':boot,'trainer':found[0],'job':job,'worker':int(worker),'instance':instance}))
'''


def audit(record, assignments, *, read_sandbox=cwctl.read_sandbox):
    run=record['run']; url=record['url']; evidence={}; seen=set()
    for worker, ident in sorted(assignments.items()):
        if ident in seen:raise ValueError('one sandbox cannot own two worker IDs')
        seen.add(ident)
        existing=record['learners'][worker]
        if existing is not None and existing!=ident:raise ValueError('assignment conflicts with run journal')
        expected=diloco.learner_job(run,worker,url)
        instance=expected['env'].get('NATLANG_CW_INSTANCE','')
        box=read_sandbox(ident);box.get_status()
        if str(getattr(box.status,'value',box.status)).lower()!='running':raise ValueError('sandbox is not running')
        result=box.exec(['python3','-c',PROBE,expected['id'],str(worker),instance],timeout_seconds=30).result()
        if result.returncode:raise ValueError(f'worker {worker} lacks exact live boot/role/input evidence')
        proof=json.loads(result.stdout)
        if proof.get('job')!=expected['id'] or proof.get('worker')!=worker or proof.get('instance')!=instance:
            raise ValueError('sandbox returned mismatched ownership')
        evidence[str(worker)]={'sandbox_id':ident,**proof}
    return {'schema':'natlang.diloco-ownership-audit/1','run_id':run['id'],'workers':evidence,
            'note':'Read-only point-in-time proof; verify identities again before any adoption.'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('run_id');parser.add_argument('assignments',nargs='+',help='WORKER=SANDBOX_ID');args=parser.parse_args()
    record=json.loads((diloco.RUNS/(args.run_id+'.json')).read_text());assignments={}
    for item in args.assignments:
        worker,ident=item.split('=',1);worker=int(worker)
        if worker in assignments or not 0<=worker<record['run']['settings']['workers']:raise ValueError('invalid worker assignment')
        assignments[worker]=ident
    print(json.dumps(audit(record,assignments),indent=2))
if __name__=='__main__':main()
