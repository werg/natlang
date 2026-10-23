"""Continuously admit exact closed Qwen36 tranches from the internal spool.

The worker polls publisher-produced tranche directories only; it does not
produce teacher data. Each selected boundary passes the frozen full-source
verifier, both-model native converter, current protection manifest and exact
registered-content dedup before normal registry publication.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time, traceback
from pathlib import Path
from scripts.coreweave.teacher_tranche_admission import Tranche, file_binding, read_bound

ROOT=Path('/home/werg/natlang')
CODE_ROOT=Path(__file__).resolve().parents[2]
SPOOL=Path('/home/werg/data/natlang-internal-spool/teacher-qwen36-continuation-20261023')
ORIGINAL_SETTINGS=Path('/home/werg/data/dgxops/qwen36-ready-native-admission-20261022/source-admission-settings.json')
PUBLISHER_CONFIG=Path('/home/werg/data/dgxops/qwen36-internal-spool-continuation-root-20261023/publisher.json')
DEPLOYMENT=Path('/home/werg/data/dgxops/qwen36-internal-spool-continuation-root-20261023/deployment.json')
TOKENIZERS=SPOOL/'assets/tokenizers/tokenizer-relocation.json'
PROTECTION=Path('/home/werg/data/dgxops/acquired-connected-source-protection-20261016-v2/protection-manifest.json')
RELOCATIONS=SPOOL/'assets/qwen36-predecessor-relocations.json'
OPERATOR_ROOT=Path('/home/werg/natlang-teacher-drain-export-20261022')
ORIGINAL_SETTINGS_SHA='7572d88ad494c03448c27081abeba95c1bb2430859a846ca17fdccf7154e838b'
PUBLISHER_SHA='121a590664c484325ad4bf34087e25cb9614f74b78ffb892d4c7785c817d9308'
DEPLOYMENT_SHA='6772cabd89be7b6d5cbdcc27e5be49c52ea3a2a900f8e82beeed2500fedffe8a'
BAD_82='qwen36-verified-positive-native-graphs-20261023-internal-batch0082-tranche0000'


def _run(args:list[str],log:Path,env:dict|None=None)->None:
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('ab',buffering=0) as stream:
        if env is None:
            env={**os.environ,'PYTHONPATH':os.pathsep.join((str(CODE_ROOT),str(CODE_ROOT/'training/neuralese')))}
        proc=subprocess.run(args,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=False)
    if proc.returncode: raise RuntimeError(f'command failed ({proc.returncode}): {log}')

def _exists_valid_source(ctx:Tranche)->bool:
    source_path=ctx.admission(SPOOL)/'canonical-source-verification.json'
    result_path=SPOOL/'source-executions'/ctx.corpus_id/'source-result.json'
    if not source_path.is_file() or not result_path.is_file(): return False
    proof=read_bound(file_binding(source_path)); result=read_bound(file_binding(result_path))
    return (proof.get('status')=='passed' and proof.get('training_admission') is False
        and result.get('training_admission') is False and result.get('training_exposure') is False
        and result.get('source_proof')==file_binding(source_path))

def _ensure_source(ctx:Tranche,target:str)->None:
    if _exists_valid_source(ctx): return
    fetch=[sys.executable,str(CODE_ROOT/'scripts/coreweave/fetch_teacher_internal_ready_assets.py'),
           '--batch',str(ctx.batch),'--source-cases-sha256',target]
    _run(fetch,SPOOL/'worker-logs'/ctx.corpus_id/'fetch-ready-assets.log')
    prepare=[sys.executable,str(CODE_ROOT/'scripts/coreweave/prepare_teacher_internal_ready_relocation.py'),
             '--batch',str(ctx.batch),'--source-cases-sha256',target]
    _run(prepare,SPOOL/'worker-logs'/ctx.corpus_id/'prepare-relocation.log')
    ordinal_data=read_bound(file_binding(SPOOL/'assets/ready-packet/candidate-scan.json'))
    matches=[i for i,row in enumerate(ordinal_data['scan']['candidates'])
             if row.get('cases_binding',{}).get('sha256')==target]
    if len(matches)!=1: raise ValueError('selected source cases do not bind exactly one ready packet')
    relocation=SPOOL/'assets/ready-packet'/f'packet-{matches[0]:05d}'/'relocation.json'
    cmd=[sys.executable,str(CODE_ROOT/'scripts/coreweave/teacher_internal_source.py'),
        '--original-settings',str(ORIGINAL_SETTINGS),'--original-settings-sha256',ORIGINAL_SETTINGS_SHA,
        '--publisher-config',str(PUBLISHER_CONFIG),'--publisher-sha256',PUBLISHER_SHA,
        '--deployment',str(DEPLOYMENT),'--deployment-sha256',DEPLOYMENT_SHA,
        '--ready-packet-relocation',str(relocation),'--spool',str(SPOOL),
        '--operator-root',str(OPERATOR_ROOT),'--batch',str(ctx.batch),'--number',str(ctx.number)]
    _run(cmd,SPOOL/'worker-logs'/ctx.corpus_id/'source.log')

def _conversion_is_current(path:Path,ctx:Tranche)->bool:
    result=read_bound(file_binding(path/'both-port-conversion.json'))
    current_source_proof=ctx.admission(SPOOL)/'canonical-source-verification.json'
    current_source=ctx.admission(SPOOL)/'native-trajectories.jsonl'
    if (result.get('batch')!=ctx.batch or result.get('tranche')!=ctx.number
            or result.get('training_registration') is not False or result.get('training_exposure') is not False):
        return False
    if (result.get('source_proof')!=file_binding(current_source_proof)
            or result.get('source',{}).get('sha256')!=file_binding(current_source)['sha256']
            or result.get('protection_manifest',{}).get('sha256')!=file_binding(PROTECTION)['sha256']):
        return False
    if any(row.get('corpus_id')==BAD_82 for row in result.get('same_actor_source_boundary_inputs',[])):
        return False
    policies={row['model']:read_bound(row['receipt']) for row in result.get('models',[])}
    if set(policies)!={'lfm25','mellum21'}: return False
    return (policies['lfm25'].get('explicit_history_reasoning_policy')=='keep'
        and policies['mellum21'].get('explicit_history_reasoning_policy') is None
        and all(row.get('failed_assistant_history_policy')=='mask-runtime-failures'
                for row in policies.values()))

def _next_conversion(ctx:Tranche)->Path|None:
    root=SPOOL/'conversions'/f'batch-{ctx.batch:04d}'
    root.mkdir(parents=True,exist_ok=True)
    prefix=f'tranche-{ctx.number:04d}-attempt-'
    for candidate in sorted(root.glob(prefix+'*')):
        result=candidate/'both-port-conversion.json'
        if result.is_file():
            if _conversion_is_current(candidate,ctx):
                return candidate
    used=[int(p.name.rsplit('-',1)[-1]) for p in root.glob(prefix+'*') if p.name.rsplit('-',1)[-1].isdigit()]
    number=max(used,default=-1)+1
    output=root/f'{prefix}{number:04d}'
    settings=SPOOL/'operator-settings'/f'{ctx.corpus_id}.json'
    if not settings.is_file(): raise ValueError('source stage did not write its exact bound settings')
    _run([sys.executable,str(CODE_ROOT/'scripts/coreweave/teacher_predecessor_relocations.py'),
          '--repository',str(ROOT),'--candidate-root',str(SPOOL/'conversions'),
          '--namespace',ctx.teacher_namespace,'--output',str(RELOCATIONS)],
         SPOOL/'worker-logs'/ctx.corpus_id/'refresh-predecessor-relocations.log')
    cmd=[sys.executable,str(CODE_ROOT/'scripts/coreweave/teacher_internal_conversion.py'),
        '--repository',str(ROOT),'--settings',str(settings),'--settings-sha256',file_binding(settings)['sha256'],
        '--tokenizer-manifest',str(TOKENIZERS),'--protection-manifest',str(PROTECTION),
        '--output',str(output),'--relocation-manifest',str(RELOCATIONS),
        '--batch',str(ctx.batch),'--number',str(ctx.number)]
    _run(cmd,SPOOL/'worker-logs'/ctx.corpus_id/'conversion.log')
    return output

def _registered_ids()->set[str]:
    result=subprocess.run(['git','show','origin/main:training/neuralese_corpora.json'],cwd=ROOT,
                          capture_output=True,text=True,check=True)
    return {row['id'] for row in json.loads(result.stdout)['corpora']}

def _publish_if_new_delta(ctx:Tranche,conversion:Path)->dict|None:
    proof=read_bound(file_binding(conversion/'both-port-conversion.json'))
    row_counts={}
    for row in proof['models']:
        receipt=read_bound(row['receipt']); row_counts[row['model']]=receipt['outputs']['train']['rows']
    if not any(row_counts.values()): return None
    known=_registered_ids()
    base=f'qwen36-verified-positive-native-graphs-20261023-internal-batch{ctx.batch:04d}-tranche{ctx.number:04d}'
    if ctx.batch==82 and ctx.number==0:
        repair_id=base+'-history-policy-keep-repair-v1'
        if repair_id in known: return None
        if BAD_82 not in known:
            raise ValueError('bad unexposed batch82 view is no longer registered; repair publication needs its prior exact entry')
        # Repair the already registered but unexposed incomplete view under a
        # versioned id. Its bad LFM receipt remains preserved as transition proof.
        env={**os.environ,'PYTHONPATH':os.pathsep.join((str(CODE_ROOT),str(CODE_ROOT/'training/neuralese')))}
        return subprocess.run([sys.executable,str(CODE_ROOT/'scripts/coreweave/publish_teacher_internal_tranche.py'),
            '--batch',str(ctx.batch),'--number',str(ctx.number),'--conversion',str(conversion),
            '--suffix','history-policy-keep-repair-v1','--supersedes',BAD_82],cwd=ROOT,
            env=env,capture_output=True,text=True,check=True).stdout
    suffix='content-delta-v1' if base in known else ''
    env={**os.environ,'PYTHONPATH':os.pathsep.join((str(CODE_ROOT),str(CODE_ROOT/'training/neuralese')))}
    return subprocess.run([sys.executable,str(CODE_ROOT/'scripts/coreweave/publish_teacher_internal_tranche.py'),
        '--batch',str(ctx.batch),'--number',str(ctx.number),'--conversion',str(conversion),
        '--suffix',suffix],cwd=ROOT,env=env,capture_output=True,text=True,check=True).stdout

def _boundaries(first_batch:int,namespace:str,batches:set[int]|None=None,tranches:set[int]|None=None):
    for batch_dir in sorted(SPOOL.glob('batch-*')):
        try: batch=int(batch_dir.name.split('-')[1])
        except (IndexError,ValueError): continue
        if batch<first_batch or (batches is not None and batch not in batches): continue
        for tranche_path in sorted(batch_dir.glob('tranche-*/tranche-*.json')):
            raw=read_bound(file_binding(tranche_path))
            number=raw.get('number')
            if tranches is not None and number not in tranches: continue
            if (raw.get('schema')!='natlang.teacher-incremental-export-tranche/1'
                    or raw.get('batch')!=batch or type(number) is not int
                    or raw.get('training_admission') is not False):
                continue
            private=SPOOL/f'batch-{batch:04d}'/'private-receipts'/f'tranche-{number:04d}.json'
            if not private.is_file(): continue
            yield Tranche(batch,number,teacher_namespace=namespace),raw

def run(*,once:bool,poll_seconds:int,batches:set[int]|None=None,tranches:set[int]|None=None)->None:
    publisher=read_bound(file_binding(PUBLISHER_CONFIG))
    if (publisher.get('schema')!='natlang.teacher-incremental-export-publisher/1'
            or publisher.get('directory_root')!=str(SPOOL)):
        raise ValueError('worker spool is not bound to the selected internal publisher')
    original=read_bound(file_binding(ORIGINAL_SETTINGS))
    if file_binding(ORIGINAL_SETTINGS)['sha256']!=ORIGINAL_SETTINGS_SHA:
        raise ValueError('selected original source settings changed')
    namespace=original['teacher_namespace']
    first=publisher['first_export_batch']
    run_id=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())
    worker_log=SPOOL/'worker-logs'/f'worker-{run_id}.jsonl'
    seen=set()
    while True:
        changed=False
        for ctx,tranche in _boundaries(first,namespace,batches,tranches):
            key=ctx.corpus_id
            if key in seen: continue
            seen.add(key); changed=True
            target=tranche['context']['source-cases.jsonl']['sha256']
            record={'at':time.time(),'tranche':key,'source_cases_sha256':target,
                    'private_receipt':file_binding(SPOOL/f'batch-{ctx.batch:04d}'/'private-receipts'/f'tranche-{ctx.number:04d}.json'),
                    'training_admission':False}
            try:
                _ensure_source(ctx,target)
                conversion=_next_conversion(ctx)
                if conversion is None: raise RuntimeError('conversion did not produce a durable both-port receipt')
                result=_publish_if_new_delta(ctx,conversion)
                record.update({'source_proof':file_binding(ctx.admission(SPOOL)/'canonical-source-verification.json'),
                    'conversion':file_binding(conversion/'both-port-conversion.json'),
                    'row_counts':{x['model']:read_bound(x['receipt'])['outputs']['train']['rows']
                                  for x in read_bound(file_binding(conversion/'both-port-conversion.json'))['models']},
                    'publication':result,'status':'registered' if result else 'no-new-numeric-rows',
                    'training_admission':bool(result),'training_exposure':False})
            except Exception as error:
                record.update({'status':'error','error':f'{type(error).__name__}: {error}',
                    'traceback':traceback.format_exc(),'training_admission':False,'training_exposure':False})
            worker_log.parent.mkdir(parents=True,exist_ok=True)
            with worker_log.open('a') as stream: stream.write(json.dumps(record,sort_keys=True)+'\n')
            print(json.dumps(record,sort_keys=True),flush=True)
        if once: return
        time.sleep(poll_seconds)
        # New poll sees new publisher exports; prior work remains auditable in
        # the append-only log and is independently checked before every rerun.
        seen=set()

def main():
    p=argparse.ArgumentParser();p.add_argument('--once',action='store_true');p.add_argument('--poll-seconds',type=int,default=60);p.add_argument('--batch',type=int,action='append');p.add_argument('--tranche',type=int,action='append');a=p.parse_args()
    run(once=a.once,poll_seconds=a.poll_seconds,batches=set(a.batch) if a.batch else None,
        tranches=set(a.tranche) if a.tranche else None)
if __name__=='__main__':main()
