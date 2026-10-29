"""Publish actual admitted whole-graph RL evidence to a shared DiLoCo coordinator."""
from __future__ import annotations
import json,tarfile,tempfile,os
from pathlib import Path
import torch
from ..common.hashing import sha256_file_hex_cached,canonical_json_sha256_hex
from ..train.diloco.worker import DilocoSettings,transport_from_env
from ..train.diloco.role_contract import SCHEMA,validate
from ..train.native_diloco import identity

def _flag(argv,name):
    values=[argv[i+1] for i,x in enumerate(argv[:-1]) if x==name]
    if len(values)!=1:raise ValueError('on-policy stage needs exactly one '+name)
    return values[0]

def _logical_source_paths(path, parents=()):
    """Verify the immutable DAG without replacing its declared sandbox paths."""
    path=Path(os.path.abspath(path))
    if path in parents:raise ValueError('cyclic source dependency')
    if path.suffix!='.json':return []
    value=json.loads(path.read_text())
    if value.get('schema')!='natlang.native-graph-sources/1' or not value.get('sources'):raise ValueError('invalid native source DAG')
    result=[]
    for item in value['sources']:
        if item.get('kind') not in ('native-graphs','semantic-pairs'):raise ValueError('invalid source kind')
        child=Path(item['path']);child=Path(os.path.abspath(child if child.is_absolute() else path.parent/child))
        if sha256_file_hex_cached(child)!=item.get('sha256'):raise ValueError('native source dependency changed')
        result.append(child);result.extend(_logical_source_paths(child,(*parents,path)))
    return result


def register_admitted_stage(args,trainer_args):
    """Register once, only after real RL admission and an exact drained checkpoint."""
    if '--diloco' not in trainer_args:return None
    settings=DilocoSettings.from_dict(json.loads(_flag(trainer_args,'--diloco')));worker=settings.resolved_worker_id
    checkpoint=Path(args.continue_from).resolve(strict=True);parent_sha=sha256_file_hex_cached(checkpoint)
    state=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True);training=state.get('training') or {};saved=training.get('diloco') or {}
    if (saved.get('worker')!=worker or not saved.get('session') or not saved.get('contract_sha256') or saved.get('pending') or saved.get('awaiting_global') or any(saved.get('steps_since') or []) or any(saved.get('tokens_since') or [])):
        raise ValueError('on-policy stage requires its exact drained worker state without uncommitted windows')
    recipe=json.loads(_flag(trainer_args,'--recipe-binding'));paths=[Path(recipe['path']),Path(_flag(trainer_args,'--native-graphs')),Path(_flag(trainer_args,'--held-native-graphs')),Path(_flag(trainer_args,'--objective-spec')),Path(args.out)/'rl-data-receipt.json',*[Path(x) for x in args.rollouts]]
    for source in (Path(_flag(trainer_args,'--native-graphs')),Path(_flag(trainer_args,'--held-native-graphs'))):paths.extend(_logical_source_paths(source))
    refs=[];seen=set()
    for path in paths:
        path=Path(os.path.abspath(path))
        if not path.is_file():raise ValueError('stage dependency missing')
        if str(path) not in seen:refs.append({'path':str(path),'sha256':sha256_file_hex_cached(path)});seen.add(str(path))
    receipt=Path(os.path.abspath(Path(args.out)/'rl-data-receipt.json'))
    item={'kind':'jolt','identity':identity(checkpoint,trainer_args,settings),'dependencies':refs,
          'on_policy_parent_sha256':parent_sha,'on_policy_parent':{'path':str(checkpoint),'sha256':parent_sha},
          'collector_receipt':next(x for x in refs if x['path']==str(receipt))}
    return _publish(item,saved,settings,args.out)


def _publish(item,saved,settings,out):
    refs=item['dependencies'];worker=settings.resolved_worker_id
    transport=transport_from_env(settings);authority=transport.json('GET',f'/v1/role-contracts?worker={worker}')
    if authority.get('schema')!=SCHEMA:raise ValueError('RL stage coordinator lacks heterogeneous authority')
    initial=transport.json('GET','/v1/manifest')['identity']
    static={(ref['path'],ref['sha256']) for roles in initial['roles'].values() for stage in roles for ref in stage.get('dependencies',[])}
    digest=canonical_json_sha256_hex(item);existing=authority['roles'][str(worker)]
    if not any(canonical_json_sha256_hex(x)==digest for x in existing):existing.append(item)
    validate(authority,settings.workers,verify_files=True,only_digest=digest,verify_parent=True)
    # Local evidence contains only source/objective/receipt bytes; the full policy parent stays with its owner.
    fd,temp=tempfile.mkstemp(prefix='.jolt-contract-',suffix='.tar',dir=out);os.close(fd);temp=Path(temp)
    try:
        with tarfile.open(temp,'w') as tar:
            for ref in {x['sha256']:x for x in refs if (x['path'],x['sha256']) not in static}.values():tar.add(ref['path'],arcname=ref['sha256'],recursive=False)
        evidence_sha=sha256_file_hex_cached(temp);name='role-contract-'+digest+'.tar';transport.upload_file(temp,name)
        transport.json('POST','/v1/lease',{'worker':worker,'session':saved['session'],'contract_sha256':saved['contract_sha256']})
        result=transport.json('POST','/v1/register-role-contract',{'worker':worker,'session':saved['session'],'contract':item,'evidence_name':name,'evidence_sha256':evidence_sha})
        if result.get('sha256')!=digest or result.get('status') not in ('registered','already_registered'):raise ValueError('coordinator did not seal exact on-policy contract')
        return result
    finally:temp.unlink(missing_ok=True)


def register_supervised_stage(args,trainer_args,saved):
    """Return from a drained JOLT stage to the exact supervised recipe."""
    settings=DilocoSettings.from_dict(json.loads(_flag(trainer_args,'--diloco')));worker=settings.resolved_worker_id
    if (saved.get('worker')!=worker or not saved.get('session') or not saved.get('contract_sha256') or saved.get('pending') or saved.get('awaiting_global') or any(saved.get('steps_since') or []) or any(saved.get('tokens_since') or [])):
        raise ValueError('supervised continuation requires a drained JOLT checkpoint')
    checkpoint=Path(args.continue_from).resolve(strict=True);parent_sha=sha256_file_hex_cached(checkpoint)
    recipe=json.loads(_flag(trainer_args,'--recipe-binding'));paths=[Path(recipe['path'])]
    for name in ('--native-graphs','--held-native-graphs','--foundation-native-graphs','--pending-focus','--ce-supervision-policy'):
        if name in trainer_args:paths.append(Path(_flag(trainer_args,name)))
    for source in (Path(_flag(trainer_args,'--native-graphs')),Path(_flag(trainer_args,'--held-native-graphs'))):paths.extend(_logical_source_paths(source))
    refs=[];seen=set()
    for path in paths:
        path=Path(os.path.abspath(path))
        if not path.is_file():raise ValueError('stage dependency missing')
        if str(path) not in seen:refs.append({'path':str(path),'sha256':sha256_file_hex_cached(path)});seen.add(str(path))
    item={'kind':'sft','identity':identity(checkpoint,trainer_args,settings),'dependencies':refs,
          'continuation_parent_sha256':parent_sha,'continuation_parent':{'path':str(checkpoint),'sha256':parent_sha},
          'predecessor_contract_sha256':saved['contract_sha256']}
    return _publish(item,saved,settings,args.out)


def register_source_append(args,trainer_args,saved,transition):
    """Register the trainer-verified exact source append at a drained boundary."""
    settings=DilocoSettings.from_dict(json.loads(_flag(trainer_args,'--diloco')));worker=settings.resolved_worker_id
    if (saved.get('worker')!=worker or not saved.get('session') or not saved.get('contract_sha256') or saved.get('pending') or saved.get('awaiting_global') or any(saved.get('steps_since') or []) or any(saved.get('tokens_since') or [])):
        raise ValueError('source append needs a drained exact worker checkpoint')
    checkpoint=Path(args.out)/'checkpoint.pt'
    if sha256_file_hex_cached(checkpoint)!=transition.get('checkpoint_sha256'):raise ValueError('source append checkpoint differs')
    recipe=json.loads(_flag(trainer_args,'--recipe-binding'));paths=[Path(recipe['path'])]
    graph_flags=('--native-graphs','--held-native-graphs','--foundation-native-graphs','--continued-native-graphs','--continued-held-native-graphs')
    for name in (*graph_flags,'--pending-focus','--ce-supervision-policy'):
        if name in trainer_args:paths.append(Path(_flag(trainer_args,name)))
    for name in graph_flags:
        if name in trainer_args:paths.extend(_logical_source_paths(Path(_flag(trainer_args,name))))
    proof_pairs=[]
    for provenance in recipe.get('external_provenances') or []:
        content=provenance.get('content_audit');protection=provenance.get('clean_closure_audit')
        if content and protection:
            for name in ('closure','mechanisms_archive','content_audit','clean_closure_audit'):
                if provenance.get(name):paths.append(Path(provenance[name]['path']))
            doc=json.loads(Path(content['path']).read_text())
            if doc.get('schema')=='natlang.big2-multi-content-audit/1' and (doc.get('base') or {}).get('sha256')==transition['prior_source_sha256']:
                proof_pairs.append((content,protection))
    if len(proof_pairs)!=1:raise ValueError('source append requires one authenticated exact content and protection proof')
    transition={**transition,'content_audit':proof_pairs[0][0],'protection_audit':proof_pairs[0][1]}
    if '--ce-supervision-policy' in trainer_args:
        from ..train.native_supervision import ce_policy_dependencies
        ce=Path(_flag(trainer_args,'--ce-supervision-policy'));paths.extend(Path(x['path']) for x in ce_policy_dependencies({'path':str(ce),'sha256':sha256_file_hex_cached(ce)}))
    refs=[];seen=set()
    for path in paths:
        path=Path(os.path.abspath(path))
        if not path.is_file():raise ValueError('stage dependency missing')
        if str(path) not in seen:refs.append({'path':str(path),'sha256':sha256_file_hex_cached(path)});seen.add(str(path))
    parent=Path(args.continue_from).resolve(strict=True);parent_sha=sha256_file_hex_cached(parent)
    item={'kind':'sft','identity':identity(parent,trainer_args,settings),'dependencies':refs,
          'continuation_parent_sha256':parent_sha,'continuation_parent':{'path':str(parent),'sha256':parent_sha},
          'predecessor_contract_sha256':saved['contract_sha256'],'source_transition':transition}
    return _publish(item,saved,settings,args.out),item['identity']
