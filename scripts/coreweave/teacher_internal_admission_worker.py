"""Continuously admit exact closed Qwen36 tranches from the internal spool.

The worker polls publisher-produced tranche directories only; it does not
produce teacher data. Each selected boundary passes the frozen full-source
verifier, both-model native converter, current protection manifest and exact
registered-content dedup before normal registry publication.
"""
from __future__ import annotations
import hashlib
import argparse, json, os, subprocess, sys, time, traceback
from pathlib import Path
from natlang_neuralese.common.paths import resolve
from scripts.coreweave.teacher_tranche_admission import Tranche, file_binding, read_bound
from scripts.coreweave.teacher_internal_conversion import _verified_zero_conversion

ROOT=resolve('repo')
CODE_ROOT=Path(__file__).resolve().parents[2]
SPOOL=resolve('data_nvme','natlang-internal-spool','teacher-qwen36-continuation-20261023')
ORIGINAL_SETTINGS=resolve('data_nvme','dgxops','qwen36-ready-native-admission-20261022','source-admission-settings.json')
PUBLISHER_CONFIG=resolve('data_nvme','dgxops','qwen36-internal-spool-continuation-root-20261023','publisher.json')
DEPLOYMENT=resolve('data_nvme','dgxops','qwen36-internal-spool-continuation-root-20261023','deployment.json')
TOKENIZERS=SPOOL/'assets/tokenizers/tokenizer-relocation.json'
PROTECTION=resolve('data_nvme','dgxops','acquired-connected-source-protection-20261016-v2','protection-manifest.json')
RELOCATIONS=SPOOL/'assets/qwen36-predecessor-relocations.json'
OPERATOR_ROOT=resolve('data_nvme','natlang-teacher-drain-export-20261022')
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

def _existing_conversion(corpus_id:str,conversion_ref:dict)->dict|None:
    """Return a prior exact publication; reject immutable-ID content conflicts."""
    registry_ref=file_binding(ROOT/'training/neuralese_corpora.json')
    registry=read_bound(registry_ref)
    matches=[entry for entry in registry.get('corpora',[]) if entry.get('id')==corpus_id]
    if len(matches)!=1 or matches[0].get('training_admission') is not True:
        raise ValueError(f'existing publication is not one admitted registry entry: {corpus_id}')
    entry=matches[0]
    manifest_ref=file_binding(ROOT/'training/corpus-manifests'/f'{corpus_id}.json')
    manifest=read_bound(manifest_ref)
    if manifest.get('id')!=corpus_id or manifest.get('path')!=entry.get('path'):
        raise ValueError(f'existing publication manifest differs from registry: {corpus_id}')
    proof_rel='evidence/registration-proof.json'
    proof_members=[row for row in manifest.get('files',[]) if row.get('path')==proof_rel]
    registry_proof=(entry.get('source_binding') or {}).get('registration_proof') or {}
    if (len(proof_members)!=1 or registry_proof.get('path')!=f"{entry['path']}/{proof_rel}"
            or any(registry_proof.get(key)!=proof_members[0].get(key) for key in ('sha256','bytes'))):
        raise ValueError(f'existing publication does not bind its exact registration proof: {corpus_id}')
    entry_path=ROOT/entry['path']/proof_rel
    proof_ref=file_binding(entry_path)
    proof=read_bound(proof_ref)
    if (proof.get('schema')!='natlang.teacher-internal-tranche-registration-proof/1'
            or proof.get('corpus_id')!=corpus_id or proof.get('training_admission') is not True):
        raise ValueError(f'existing publication has an invalid registration proof: {corpus_id}')
    if any(proof_ref.get(key)!=proof_members[0].get(key) for key in ('sha256','bytes')):
        raise ValueError(f'existing registration proof differs from its manifest: {corpus_id}')
    registered_ref=proof.get('conversion') or {}
    if (registered_ref.get('sha256')!=conversion_ref.get('sha256')
            or registered_ref.get('bytes')!=conversion_ref.get('bytes')):
        raise ValueError(f'immutable corpus ID is already bound to a different conversion: {corpus_id}')
    if (file_binding(ROOT/'training/neuralese_corpora.json')!=registry_ref
            or file_binding(ROOT/'training/corpus-manifests'/f'{corpus_id}.json')!=manifest_ref
            or file_binding(entry_path)!=proof_ref):
        raise ValueError(f'existing publication custody changed while checking idempotency: {corpus_id}')
    return {'corpus_id':corpus_id,'proof':proof_ref,'idempotent':True,
            'registered':True,'training_exposure':False}

def _publish_if_new_delta(ctx:Tranche,conversion:Path)->dict|None:
    conversion_ref=file_binding(conversion/'both-port-conversion.json')
    proof=read_bound(conversion_ref)
    row_counts={}
    for row in proof['models']:
        receipt=read_bound(row['receipt']); row_counts[row['model']]=receipt['outputs']['train']['rows']
    if not any(row_counts.values()):
        if set(row_counts)!={'lfm25','mellum21'}:
            raise ValueError('zero-row disposition requires both exact native model receipts')
        source_proof=read_bound(proof['source_proof'])
        contracts={'lfm25':{'tokenizer_fingerprint':'be949ec48f44e17b2b17e1131aeb1dab4262175ba20f9783ecd8e59c27a58105'},
                   'mellum21':{'tokenizer_fingerprint':'fbf22d8773d09be802cb9977895067db36cdfba747fcb24b02ddbf1312d3778f'}}
        verified=_verified_zero_conversion(ROOT,conversion/'both-port-conversion.json',ctx,
                                           source_proof,contracts)
        return {'status':'verified-zero-delta','conversion':conversion_ref,
                'training_admission':False,'training_exposure':False,
                'accounted_boundary':verified}
    known=_registered_ids()
    base=f'qwen36-verified-positive-native-graphs-20261023-internal-batch{ctx.batch:04d}-tranche{ctx.number:04d}'
    if ctx.batch==82 and ctx.number==0:
        repair_id=base+'-history-policy-keep-repair-v1'
        if repair_id in known:
            existing=_existing_conversion(repair_id,conversion_ref)
            if existing: return existing
        if BAD_82 not in known:
            raise ValueError('bad unexposed batch82 view is no longer registered; repair publication needs its prior exact entry')
        # Repair the already registered but unexposed incomplete view under a
        # versioned id. Its bad LFM receipt remains preserved as transition proof.
        env={**os.environ,'PYTHONPATH':os.pathsep.join((str(CODE_ROOT),str(CODE_ROOT/'training/neuralese')))}
        return subprocess.run([sys.executable,str(CODE_ROOT/'scripts/coreweave/publish_teacher_internal_tranche.py'),
            '--batch',str(ctx.batch),'--number',str(ctx.number),'--conversion',str(conversion),
            '--suffix','history-policy-keep-repair-v1','--supersedes',BAD_82],cwd=ROOT,
            env=env,capture_output=True,text=True,check=True).stdout
    if base in known:
        try:
            existing=_existing_conversion(base,conversion_ref)
        except ValueError as error:
            if 'different conversion' not in str(error):
                raise
            existing=None
        if existing: return existing
        # Recognize the one immutable suffix used by earlier worker versions
        # before switching to conversion-addressed deltas.
        legacy_delta=base+'-content-delta-v1'
        if legacy_delta in known:
            try:
                existing=_existing_conversion(legacy_delta,conversion_ref)
            except ValueError as error:
                if 'different conversion' not in str(error):
                    raise
            if existing: return existing
        # The source boundary may legitimately yield an additional numeric
        # view after a separately versioned repair. Name that immutable delta
        # from its exact conversion binding so polling is idempotent.
        suffix='content-delta-'+hashlib.sha256(
            f"{conversion_ref['sha256']}:{conversion_ref['bytes']}".encode()).hexdigest()[:16]
        delta_id=base+'-'+suffix
        if delta_id in known:
            existing=_existing_conversion(delta_id,conversion_ref)
            if existing: return existing
            raise ValueError(f'exact conversion-derived corpus ID conflicts: {delta_id}')
    else:
        suffix=''
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


def _poll_policy_binding()->list[dict]:
    """Small policy inputs invalidate scheduling reuse when they change.

    This is not a new custody proof. A skipped completed operation grants no
    admission, exposure or fresh verification; consumers keep their normal
    exact-content checks. A new worker always performs a fresh first pass.
    """
    return [file_binding(path) for path in (
        ORIGINAL_SETTINGS, PUBLISHER_CONFIG, DEPLOYMENT, TOKENIZERS, PROTECTION)]


def _poll_stage_bindings(ctx:Tranche)->list[dict]:
    """Observe small stage declarations, without claiming their verification.

    Proof repair, a changed model receipt or a newly completed conversion must
    return through the normal verifier even when the publisher is unchanged.
    Large graph payloads remain checked by that verifier and actual consumers.
    """
    paths=[ctx.admission(SPOOL)/'canonical-source-verification.json',
           SPOOL/'source-executions'/ctx.corpus_id/'source-result.json']
    root=SPOOL/'conversions'/f'batch-{ctx.batch:04d}'
    paths.extend(sorted(root.glob(f'tranche-{ctx.number:04d}-attempt-*/both-port-conversion.json')))
    bindings=[]
    for path in paths:
        if not path.is_file():
            bindings.append({'path':str(path),'missing':True})
            continue
        ref=file_binding(path);bindings.append(ref)
        if path.name=='both-port-conversion.json':
            for model in read_bound(ref).get('models',[]):
                receipt=Path(model['receipt']['path'])
                bindings.append(file_binding(receipt) if receipt.is_file() else
                                {'path':str(receipt),'missing':True})
    return bindings


def _poll_boundary_signature(tranche:dict,private:dict,policy:list[dict],stages:list[dict])->str:
    return hashlib.sha256(json.dumps({'tranche':tranche,'private':private,'policy':policy,'stages':stages},
                                    sort_keys=True,separators=(',',':')).encode()).hexdigest()


def _process_verified_boundary(ctx:Tranche)->dict:
    source_ref=file_binding(ctx.admission(SPOOL)/'canonical-source-verification.json')
    source_proof=read_bound(source_ref)
    from scripts.coreweave.teacher_internal_source import verified_zero_positive_source
    empty=verified_zero_positive_source(source_proof)
    if empty is not None:
        result_ref=file_binding(SPOOL/'source-executions'/ctx.corpus_id/'source-result.json')
        result=read_bound(result_ref)
        if (result.get('source_disposition')!='verified-zero-positive-source'
                or result.get('source_proof')!=source_ref
                or result.get('source')!=source_proof['source']
                or result.get('training_admission')is not False
                or result.get('training_exposure')is not False):
            raise ValueError('zero-positive boundary lacks its exact completed source execution')
        return {**empty,'source_proof':source_ref,'source_result':result_ref,
                'new_registration':False,'row_counts':{'lfm25':0,'mellum21':0}}
    conversion=_next_conversion(ctx)
    if conversion is None:raise RuntimeError('conversion did not produce a durable both-port receipt')
    result=_publish_if_new_delta(ctx,conversion)
    idempotent=isinstance(result,dict)and result.get('idempotent')is True
    zero_delta=isinstance(result,dict)and result.get('status')=='verified-zero-delta'
    return {'source_proof':source_ref,'conversion':file_binding(conversion/'both-port-conversion.json'),
            'row_counts':{x['model']:read_bound(x['receipt'])['outputs']['train']['rows']
                          for x in read_bound(file_binding(conversion/'both-port-conversion.json'))['models']},
            'publication':result,
            'status':('verified-zero-delta'if zero_delta else
                      'idempotent-existing-publication'if idempotent else
                      'registered'if result else'no-new-numeric-rows'),
            'new_registration':bool(result)and not idempotent and not zero_delta,
            'training_admission':bool(result)and not zero_delta,'training_exposure':False}


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
    completed={}
    while True:
        policy=_poll_policy_binding();skipped=0
        for ctx,tranche in _boundaries(first,namespace,batches,tranches):
            key=ctx.corpus_id
            private=file_binding(SPOOL/f'batch-{ctx.batch:04d}'/'private-receipts'/f'tranche-{ctx.number:04d}.json')
            target=tranche['context']['source-cases.jsonl']['sha256']
            record={'at':time.time(),'tranche':key,'source_cases_sha256':target,
                    'private_receipt':private,
                    'training_admission':False}
            try:
                signature=_poll_boundary_signature(tranche,private,policy,_poll_stage_bindings(ctx))
                if completed.get(key)==signature:
                    skipped+=1
                    continue
                _ensure_source(ctx,target)
                record.update(_process_verified_boundary(ctx))
            except Exception as error:
                record.update({'status':'error','error':f'{type(error).__name__}: {error}',
                    'traceback':traceback.format_exc(),'training_admission':False,'training_exposure':False})
            worker_log.parent.mkdir(parents=True,exist_ok=True)
            with worker_log.open('a') as stream: stream.write(json.dumps(record,sort_keys=True)+'\n')
            print(json.dumps(record,sort_keys=True),flush=True)
            if record['status'] in ('registered','idempotent-existing-publication','verified-zero-delta',
                                     'verified-zero-positive-source'):
                # Normal work may have created or repaired the observed stages.
                # Store the pre-work identity: a subsequent poll rechecks any
                # transition instead of assuming newly observed bytes passed.
                completed[key]=signature
            else:
                completed.pop(key,None)  # Failed or incomplete work retries next poll.
        if once: return
        if skipped:
            print(json.dumps({'at':time.time(),'status':'unchanged-completed-boundaries',
                              'skipped':skipped,'new_admission':False,'fresh_verification':False}),flush=True)
        time.sleep(poll_seconds)

def main():
    p=argparse.ArgumentParser();p.add_argument('--once',action='store_true');p.add_argument('--poll-seconds',type=int,default=60);p.add_argument('--batch',type=int,action='append');p.add_argument('--tranche',type=int,action='append');a=p.parse_args()
    run(once=a.once,poll_seconds=a.poll_seconds,batches=set(a.batch) if a.batch else None,
        tranches=set(a.tranche) if a.tranche else None)
if __name__=='__main__':main()
