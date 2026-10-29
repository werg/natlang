#!/usr/bin/env python3
"""Prepare LFM3 20261029 first-outer controls from actual pin/readback metadata.

This never launches a role or invents a parent. The four destinations must stage the
same seed at --seed-logical, each own pinned parent in its declared path, and the
same receipt/recipe paths before running. Host-specific preflight remains required.
"""
import argparse, copy, json
from pathlib import Path
from natlang_neuralese.common.hashing import sha256_file_hex, canonical_json_sha256_hex
from natlang_neuralese.common.jsonio import write_json_durable
from natlang_neuralese.train import recipe as R,native_resident as resident
from natlang_neuralese.train.native_diloco import trainer_argv, first_outer_rng_seed, IDENTITY_SCHEMA
from natlang_neuralese.train.diloco.worker import DilocoSettings
from natlang_neuralese.train.native_state_sft import build_parser

def bound(ref):
    p=Path(ref['path']).resolve(strict=True)
    if sha256_file_hex(p)!=ref['sha256'] or ('bytes' in ref and p.stat().st_size!=ref['bytes']):raise ValueError('input receipt differs: '+str(p))
    return p

def prepare(a):
    inventory=json.loads(bound(a.inventory).read_text()); prior=R.load_recipe(bound(a.recipe)); peers=inventory.get('peers')
    if inventory.get('schema')!='natlang.fresh-first-outer-inventory/1' or not isinstance(peers,list) or len(peers)!=4:raise ValueError('actual four-peer inventory required')
    seed=inventory['seed'];sha=seed['sha256'];size=seed['bytes']
    if sha!='51bacc47b54e937684e8e757dde12e1b01719c8f24e1d52d8f185b236753648f' or size!=12482163595:raise ValueError('common production seed changed')
    if not a.seed_logical.startswith('/work/') or not a.worker_out.startswith('/work/') or not a.coordinator_root.startswith('/work/'):raise ValueError('fresh work namespaces required')
    s=prior['stages'][0];old=R.resolve_stage_inputs(prior,s,{});train=old['native_graphs']['sha256'];held=old['held_native_graphs']['sha256']
    if train!='9d93141df15287e147afb7fa32b499ef4d59d54cdfe5de1755d24b879a821362' or held!='5053ab2b6754f2998294215e7892b5c18443d973f55e6c04239803ffac7b5e7f' or old['foundation_native_graphs']!=old['native_graphs']:raise ValueError('fresh source/held/foundation binding differs')
    source_hashes=set(); records=[]
    for rank,peer in enumerate(peers):
        if peer.get('worker')!=rank:raise ValueError('peer ordering differs')
        parent=peer['parent'];pin=json.loads(bound(peer['pin_receipt']).read_text());private=json.loads(bound(peer['private_receipt']).read_text());meta=json.loads(bound(peer['full_state_receipt']).read_text());presence=json.loads(bound(peer['owner_presence_receipt']).read_text())
        if (pin.get('sha256')!=parent['sha256'] or pin.get('bytes')!=parent['bytes'] or pin.get('step')!=parent['step'] or pin.get('source_sha256')!=train):raise ValueError('parent pin receipt differs')
        verified=private.get('independent_verification') or private.get('cloud_full_get') or {}
        if (not private.get('private') or private.get('identity',{}).get('sha256')!=parent['sha256'] or verified.get('sha256')!=parent['sha256'] or verified.get('bytes')!=parent['bytes'] or not private.get('verified_at')):raise ValueError('private full readback missing')
        if (meta.get('checkpoint_sha256')!=parent['sha256'] or meta.get('step')!=parent['step'] or meta.get('train_sha256')!=train or meta.get('held_sha256')!=held or any(meta.get(k) is not True for k in ('full_state_verified','optimizer_verified','rng_verified','exposure_verified')) or meta.get('prior_outer_history')):raise ValueError('fresh parent full state not verified')
        if presence.get('sha256')!=parent['sha256'] or presence.get('bytes')!=parent['bytes'] or presence.get('owner_access') is not True or presence.get('private_object_present') is not True or presence.get('observed_at',0)<private['verified_at']:raise ValueError('fresh owner private presence missing')
        if rank and (parent['sha256']!=sha or parent['bytes']!=size or parent['step']!=2):raise ValueError('seed clone differs from common step2 baseline')
        if rank==0 and parent['step']<2:raise ValueError('peer1 earned parent predates common seed')
        source_hashes.add(meta.get('source_hash'));records.append({'parent':parent,'pin_receipt':peer['pin_receipt'],'private_receipt':peer['private_receipt'],'full_state_receipt':peer['full_state_receipt'],'owner_presence_receipt':peer['owner_presence_receipt'],'owner_get':'verified' if verified.get('readback_location')=='owner' else 'pending'})
    witness=json.loads(bound(inventory['peer1_lineage_witness']).read_text())
    owner_ref=witness.get('owner') or {};owner=json.loads(bound(owner_ref).read_text());accepted=owner.get('accepted') or {}
    baseline=json.loads(bound(witness['baseline_pin_receipt']).read_text());latest=json.loads(bound(peers[0]['pin_receipt']).read_text())
    if (witness.get('schema')!='natlang.fresh-writer-lineage/1' or witness.get('continuity_verified') is not True
            or witness.get('baseline_sha256')!=sha or witness.get('latest_sha256')!=peers[0]['parent']['sha256']
            or baseline.get('sha256')!=sha or baseline.get('step')!=2
            or baseline.get('owner_receipt')!=owner_ref.get('path') or latest.get('owner_receipt')!=owner_ref.get('path')
            or witness.get('initial_checkpoint_sha256')!=owner.get('fresh_checkpoint_sha256')
            or witness.get('pid')!=accepted.get('pid') or witness.get('boot_id')!=accepted.get('boot_id')
            or witness.get('process_start_ticks')!=accepted.get('process_start_ticks') or accepted.get('accepted') is not True):
        raise ValueError('peer1 newer parent lacks exact continuous fresh-writer custody')
    if len(source_hashes)!=1 or any(not isinstance(x,str) or len(x)!=64 for x in source_hashes):raise ValueError('peer source hash differs')
    source_hash=source_hashes.pop();settings={'run_id':a.run_id,'workers':4,'coordinator_url':a.coordinator_url,'local_steps':8,'wire_dtype':'bf16','start_grace_seconds':30};DilocoSettings.from_dict(settings)
    a.out.mkdir(parents=True,exist_ok=False);declarations=[]
    for rank,record in enumerate(records):
        parent=record['parent'];fork=first_outer_rng_seed(sha,rank) if parent['sha256']==sha else None
        proof={'schema':'natlang.diloco-first-outer/1','worker':rank,'parent':{k:parent[k] for k in ('path','sha256','bytes')},'historical_outer_checkpoint':None,'seed_sha256':sha,'source_hash':source_hash,'held_sha256':held,'rng_fork_seed':fork};path=a.out/f'peer-{rank}.json';write_json_durable(path,proof);declarations.append({**resident.binding(path),'bytes':path.stat().st_size})
    manifest=a.out/'manifest.json';write_json_durable(manifest,{'schema':'natlang.diloco-new-outer-manifest/1','workers':4,'peers':declarations})
    recipe=copy.deepcopy(prior);stage=recipe['stages'][0];stage['inputs']['diloco_new_outer']='first_outer_manifest';recipe['input_bindings']['first_outer_manifest']=resident.binding(manifest);stage['parameters']['diloco']=settings
    path=a.out/'recipe.json';write_json_durable(path,recipe);recipe=R.load_recipe(path);stage=recipe['stages'][0];resolved=R.resolve_stage_inputs(recipe,stage,{})
    if not R._new_outer_continuation(old,resolved):raise ValueError('fresh recipe reanchor handoff rejected')
    binding=R.native_recipe_binding(path,R.recipe_identity_sha256(bound(a.recipe)));argv=R.stage_input_args(resolved,'native_state_sft',target='portable')+['--device','cuda']+R.stage_parameter_args(R.effective_stage_parameters(stage,[],recipe))+['--recipe-binding',json.dumps(binding,sort_keys=True,separators=(',',':')),'--stage-id',stage['id'],'--prerequisite-stages','[]','--continue-from',a.seed_logical,'--out',a.worker_out];build_parser().parse_args(argv)
    identities=[{'schema':IDENTITY_SCHEMA,'parent_checkpoint_sha256':sha,'trainer_argv':trainer_argv(argv),'diloco':DilocoSettings.from_dict({**settings,'worker_id':i}).identity()} for i in range(4)]
    if any(x!=identities[0] for x in identities[1:]):raise ValueError('worker coordinator identities differ')
    argfile=a.out/'worker-args.json';write_json_durable(argfile,argv)
    dependencies={dep['path']:dep['sha256'] for value in resolved.values() for dep in R._binding_files(value)}
    dependencies[str(path)]=sha256_file_hex(path)
    role={'kind':'sft','identity':identities[0],'dependencies':[{'path':p,'sha256':h} for p,h in sorted(dependencies.items())]}
    authority={'schema':'natlang.diloco-heterogeneous-identity/1','baseline_sha256':sha,'recipe_lineage_sha256':R.recipe_identity_sha256(bound(a.recipe)),'diloco':DilocoSettings.from_dict(settings).identity(),'roles':{str(i):[role] for i in range(4)}}
    from natlang_neuralese.train.diloco.role_contract import validate
    validate(authority,4,verify_files=True);authority_file=a.out/'role-contract.json';write_json_durable(authority_file,authority)
    init=['init','--root',a.coordinator_root,'--parent',a.seed_logical,'--base',stage['parameters']['base'],'--trainer-argv',json.dumps(argv,separators=(',',':')),'--settings',json.dumps(settings,separators=(',',':')),'--role-contract',str(authority_file),'--quorum','3','--ema',json.dumps({'decay':0.88,'warmup':0},separators=(',',':')),'--outer-schedule',json.dumps([{'from_round':0,'lr':1.,'momentum':0.},{'from_round':10,'lr':0.7,'momentum':0.9}],separators=(',',':')),'--max-staleness','2'];initfile=a.out/'coordinator-init-args.json';write_json_durable(initfile,init)
    result={'schema':'natlang.fresh-first-outer-prepared/1','inventory':a.inventory,'prior_recipe':a.recipe,'recipe':resident.binding(path),'manifest':resident.binding(manifest),'peer_declarations':declarations,'worker_args':resident.binding(argfile),'role_contract':resident.binding(authority_file),'coordinator_init_args':resident.binding(initfile),'source_sha256':train,'held_sha256':held,'source_hash':source_hash,'seed':seed,'peers':records,'settings':settings,'quorum':3,'common_identity_sha256':canonical_json_sha256_hex(identities[0]),'peer1_lineage_witness':inventory['peer1_lineage_witness'],'scope':'LFM3 20261029 task-specific inert controls; verify actual runtime, target staged paths, owner process and physical capacity before launch. Owner full GET status is separate.'};write_json_durable(a.out/'receipt.json',result);return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--inventory',type=Path,required=True);p.add_argument('--inventory-sha256',required=True);p.add_argument('--recipe',type=Path,required=True);p.add_argument('--recipe-sha256',required=True);p.add_argument('--out',type=Path,required=True)
    for name in ('seed-logical','worker-out','coordinator-root','coordinator-url','run-id'):p.add_argument('--'+name,required=True)
    a=p.parse_args();a.inventory={'path':str(a.inventory),'sha256':a.inventory_sha256};a.recipe={'path':str(a.recipe),'sha256':a.recipe_sha256};print(json.dumps(prepare(a)))
if __name__=='__main__':main()
