"""Package and normally register one fully verified internal teacher tranche for both students."""
from __future__ import annotations
import argparse, hashlib, json, shutil, subprocess, sys
from pathlib import Path
from natlang_neuralese.common import line_backbones as lb
from natlang_neuralese.common.paths import resolve
from scripts.coreweave.teacher_tranche_admission import file_binding, read_bound

ROOT=resolve('repo')
SPOOL=resolve('data_nvme','natlang-internal-spool','teacher-qwen36-continuation-20261023')
PROTECTION_SHA='1d5b29b2e099229bee9bcc2562aee967ecc4666d211b2ade9873c597a0542704'

def _copy(source:Path,target:Path):
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if file_binding(source)['sha256'] != file_binding(target)['sha256']:
            raise ValueError(f'immutable package member differs: {target}')
    else: shutil.copyfile(source,target)

def publish(batch:int,number:int,conversion:Path,attempt:int=0,*,suffix:str='',supersedes:str|None=None)->dict:
    corpus_id=f'qwen36-verified-positive-native-graphs-20261023-internal-batch{batch:04d}-tranche{number:04d}'
    if suffix:
        corpus_id += '-' + suffix
    source_dir=SPOOL/'admission'/f'batch-{batch:04d}'/f'tranche-{number:04d}'
    source_proof_path=source_dir/'canonical-source-verification.json'
    source_path=source_dir/'native-trajectories.jsonl'
    source_proof=read_bound(file_binding(source_proof_path))
    conversion_path=conversion/'both-port-conversion.json'
    conversion_ref=file_binding(conversion_path); converted=read_bound(conversion_ref)
    if (source_proof.get('status')!='passed' or source_proof.get('training_admission') is not False
        or source_proof.get('rows',0)<1 or converted.get('training_registration') is not False
        or converted.get('training_exposure') is not False or converted.get('batch')!=batch
        or converted.get('tranche')!=number or converted.get('source_proof')!=file_binding(source_proof_path)
        or converted.get('source',{}).get('sha256')!=file_binding(source_path)['sha256']
        or source_proof.get('current_protection',{}).get('sha256')!=PROTECTION_SHA):
        raise ValueError('source or conversion proofs do not agree on exact non-admission inputs')
    declared=frozenset(row.get('model')for row in converted.get('models',[]))
    if declared not in lb.model_key_pairs(): raise ValueError('both native model views are required')
    # The conversion's own receipts name its (lfm backbone, mellum) pair; lfm line first as in earlier packages.
    models={m:None for m in sorted(declared,key=lambda key:lb.LINES.index(lb.line_of(key)))}
    for row in converted.get('models',[]):
        model=row['model']; receipt=read_bound(row['receipt']); output_root=conversion/model
        if (model not in models or receipt.get('status')!='passed' or receipt.get('model_key')!=model
            or receipt.get('source',{}).get('sha256')!=file_binding(source_path)['sha256']):
            raise ValueError(f'{model} conversion receipt does not pass exact source binding')
        train=output_root/'train/native-graphs.jsonl.zst'; test=output_root/'test/native-graphs.jsonl.zst'
        if (file_binding(train)['sha256']!=receipt['outputs']['train']['sha256']
            or file_binding(test)['sha256']!=receipt['outputs']['test']['sha256']
            or receipt['outputs']['train']['rows']<0 or receipt['outputs']['test']['rows']!=0):
            raise ValueError(f'{model} output files differ from exact conversion receipt')
        models[model]={'receipt':row['receipt'],'train':file_binding(train),'test':file_binding(test),
            'rows':receipt['outputs']['train']['rows'],'tokens':receipt['outputs']['train']['tokens'],
            'dedup_omissions':receipt['deduplication']['omitted_rows'],'model_id':receipt['model_id'],
            'tokenizer_fingerprint':receipt['tokenizer']['expanded_fingerprint']}
    if any(value is None for value in models.values()): raise ValueError('both native model views are required')
    package=SPOOL/'registered-packages'/corpus_id
    package.mkdir(parents=True,exist_ok=True)
    for model in models:
        for rel in ('receipt.json','train/native-graphs.jsonl.zst','test/native-graphs.jsonl.zst'):
            _copy(conversion/model/rel,package/model/rel)
        models[model]['conversion_receipt']=models[model]['receipt']
        models[model]['receipt']=file_binding(package/model/'receipt.json')
        models[model]['train']=file_binding(package/model/'train/native-graphs.jsonl.zst')
        models[model]['test']=file_binding(package/model/'test/native-graphs.jsonl.zst')
    evidence={
        'native-trajectories.jsonl':source_path,
        'canonical-source-verification.json':source_proof_path,
        'source-result.json':SPOOL/'source-executions'/f'qwen36-verified-positive-native-graphs-20261022-recovery7853-batch{batch:04d}-tranche{number:04d}'/'source-result.json',
        'both-port-conversion.json':conversion_path,
        'tokenizer-relocation.json':SPOOL/'assets/tokenizers/tokenizer-relocation.json',
        'predecessor-relocations.json':SPOOL/'assets/qwen36-predecessor-relocations.json',
        **{f'base-inventory-{m}.json':conversion/f'base-inventory-{m}.json' for m in models}}
    for rel,path in evidence.items():
        if not path.is_file(): raise ValueError(f'missing required admission evidence: {path}')
        _copy(path,package/'evidence'/rel)
    result_path=evidence['source-result.json']; result=read_bound(file_binding(result_path))
    if (result.get('training_admission') is not False or result.get('training_exposure') is not False
        or result.get('source_proof')!=file_binding(source_proof_path)):
        raise ValueError('source execution result does not bind exact canonical source proof')
    proof={'schema':'natlang.teacher-internal-tranche-registration-proof/1','status':'passed',
        'corpus_id':corpus_id,'source_namespace':'qwen36-verified-positive-native-graphs-20261022-recovery7853',
        'batch':batch,'tranche':number,'source_verification':file_binding(package/'evidence/canonical-source-verification.json'),
        'conversion':file_binding(package/'evidence/both-port-conversion.json'),
        'source_result':file_binding(package/'evidence/source-result.json'),
        'source':file_binding(package/'evidence/native-trajectories.jsonl'),
        'current_protection_manifest_sha256':PROTECTION_SHA,
        'eligible_teacher_action_turns':source_proof['teacher_action_turns'],
        'models':models,'training_admission':True,'training_exposure':False,
        'prefix_debt':'Earlier retained source boundaries remain deferred custody evidence; this entry only admits its exact verified batch/tranche.'}
    if supersedes:
        superseded_entry=next((item for item in json.loads((ROOT/'training/neuralese_corpora.json').read_text())['corpora']
                               if item.get('id')==supersedes),None)
        if not superseded_entry or superseded_entry.get('training_admission') is not True:
            raise ValueError('policy repair does not name one currently admitted superseded view')
        superseded_manifest=read_bound(file_binding(ROOT/'training/corpus-manifests'/f'{supersedes}.json'))
        if superseded_manifest.get('id')!=supersedes:
            raise ValueError('superseded source view manifest is not exact')
        old_receipt_path=ROOT/superseded_entry['path']/'lfm25/receipt.json'
        old_receipt_ref=file_binding(old_receipt_path);old_receipt=read_bound(old_receipt_ref)
        if (old_receipt.get('source',{}).get('sha256')!=file_binding(source_path)['sha256']
                or old_receipt.get('explicit_history_reasoning_policy') is not None
                or old_receipt.get('failed_assistant_history_policy')!='mask-runtime-failures'):
            raise ValueError('superseded LFM view does not have the exact source and missing keep-policy defect')
        _copy(old_receipt_path,package/'evidence/superseded-lfm-receipt.json')
        proof['history_policy_transition']={'kind':'explicit-LFM-history-reasoning-keep-repair/1',
            'supersedes_corpus_id':supersedes,'superseded_lfm_receipt':file_binding(package/'evidence/superseded-lfm-receipt.json'),
            'superseded_lfm_receipt_sha256':old_receipt_ref['sha256'],
            'old_explicit_history_reasoning_policy':None,'new_explicit_history_reasoning_policy':'keep',
            'failed_assistant_history_policy':'mask-runtime-failures',
            'source_boundary_sha256':file_binding(source_path)['sha256'],
            'shared_graph_content_dedup':converted.get('same_actor_source_boundary_inputs',[]),
            'training_exposure':False}
    proof_path=package/'evidence/registration-proof.json'; proof_bytes=(json.dumps(proof,sort_keys=True,indent=2)+'\n').encode()
    if proof_path.exists() and proof_path.read_bytes()!=proof_bytes:
        raise ValueError('registration proof already exists with different bytes')
    if not proof_path.exists(): proof_path.write_bytes(proof_bytes)
    include=[]
    for model in models: include += [f'{model}/receipt.json',f'{model}/train/native-graphs.jsonl.zst',f'{model}/test/native-graphs.jsonl.zst']
    include += [f'evidence/{name}' for name in [*evidence,'registration-proof.json']]
    if supersedes: include.append('evidence/superseded-lfm-receipt.json')
    entry={'id':corpus_id,'path':f'runs/{corpus_id}','owner':'dgx','training_admission':True,
        'source_class':'verified-self-hosted-teacher-positive-native-whole-trajectory-native-graphs',
        'split':'Train-only; empty test views; protected source groups remain excluded.','include':include,
        'derived_from':([supersedes,'qwen-verified-positive-native-graphs-20261015-v21']
                         if supersedes else ['qwen-verified-positive-native-graphs-20261015-v21']),
        'license_provenance':'Project-collected trajectories from the pinned self-hosted Qwen3.6-35B-A3B teacher; exact source actor/runtime bindings retained.',
        'admission':f"{source_proof['rows']} full verified teacher trajectories; {source_proof['teacher_action_turns']} action turns source-verified; per-model exact dedup omissions and emitted rows are in both native conversion receipts.",
        'note':('Versioned LFM history-policy repair; zero-row Mellum view is exact boundary deduplication. '
                'Registration does not establish exposure.' if supersedes else
                'Registration establishes availability only; learner exposure is recorded by each model continuation.'),
        'source_binding':{'registration_proof':{'path':f'runs/{corpus_id}/evidence/registration-proof.json','sha256':file_binding(proof_path)['sha256'],
             'bytes':file_binding(proof_path)['bytes']},'source_sha256':file_binding(package/'evidence/native-trajectories.jsonl')['sha256'],
             'canonical_source_verification_sha256':file_binding(package/'evidence/canonical-source-verification.json')['sha256'],
             'current_protection_manifest_sha256':PROTECTION_SHA,'teacher_action_turns':source_proof['teacher_action_turns'],
             'models':models,'training_exposure':False}}
    entry_path=SPOOL/'registered-packages'/f'{corpus_id}.entry.json'; entry_bytes=(json.dumps(entry,sort_keys=True,indent=2)+'\n').encode()
    if entry_path.exists() and entry_path.read_bytes()!=entry_bytes:
        raise ValueError('registration entry already exists with different bytes')
    if not entry_path.exists(): entry_path.write_bytes(entry_bytes)
    link=f'runs/{corpus_id}={package}'
    command=[sys.executable,str(ROOT/'scripts/neuralese_data/register_corpora.py'),'--entries',str(entry_path),
        '-m',f'Register verified Qwen36 internal batch{batch:04d} tranche{number:04d} native graphs','--link',link]
    proc=subprocess.run(command,capture_output=True,text=True)
    if proc.returncode: raise RuntimeError(f'normal registry publication failed: {proc.stdout}\n{proc.stderr}')
    return {'corpus_id':corpus_id,'package':str(package),'entry':str(entry_path),'proof':file_binding(proof_path),
        'registered':proc.stdout.strip(),'registration_lock':str(Path.home()/'.local/state/natlang/register-corpora.lock'),
        'training_exposure':False}

def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',type=int,required=True);p.add_argument('--number',type=int,required=True);p.add_argument('--conversion',type=Path,required=True);p.add_argument('--suffix',default='');p.add_argument('--supersedes');a=p.parse_args()
    print(json.dumps(publish(a.batch,a.number,a.conversion,suffix=a.suffix,supersedes=a.supersedes),sort_keys=True),flush=True)
if __name__=='__main__':main()
