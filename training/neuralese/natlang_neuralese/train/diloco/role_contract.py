"""Exact per-worker native trainer contracts for heterogeneous DiLoCo rounds."""
import copy,json,re
from pathlib import Path
from ...common.hashing import canonical_json_sha256_hex,sha256_file_hex_cached
SCHEMA='natlang.diloco-heterogeneous-identity/1'

def _digest(value):return isinstance(value,str) and re.fullmatch('[a-f0-9]{64}',value)
def _flag(argv,name):
    found=[argv[i+1] for i,x in enumerate(argv[:-1]) if x==name]
    if len(found)!=1:raise ValueError('role identity must bind exactly one '+name)
    return found[0]

def validate(contract,workers,*,verify_files=False,only_digest=None,resolver=None,verify_parent=False):
    if contract.get('schema')!=SCHEMA or not _digest(contract.get('baseline_sha256')) or not _digest(contract.get('recipe_lineage_sha256')):raise ValueError('invalid heterogeneous authority')
    roles=contract.get('roles')
    if not isinstance(roles,dict) or set(roles)!={str(i) for i in range(workers)}:raise ValueError('heterogeneous authority must cover each exact worker')
    for rank,entries in roles.items():
        if not isinstance(entries,list) or not entries:raise ValueError('worker requires allowed stage contracts')
        seen=set()
        for item in entries:
            if item.get('kind') not in ('sft','jolt') or not isinstance(item.get('identity'),dict):raise ValueError('invalid declared stage contract')
            identity=item['identity'];argv=identity.get('trainer_argv')
            if identity.get('schema')!='natlang.native-diloco-identity/1' or not _digest(identity.get('parent_checkpoint_sha256')) or identity.get('diloco')!=contract.get('diloco') or not isinstance(argv,list) or not all(isinstance(x,str) for x in argv):raise ValueError('stage identity or shared optimizer settings differs')
            binding=json.loads(_flag(argv,'--recipe-binding'))
            if binding.get('lineage_sha256')!=contract['recipe_lineage_sha256']:raise ValueError('stage recipe lineage differs')
            _flag(argv,'--stage-id')
            if item['kind']=='sft' and identity['parent_checkpoint_sha256']!=contract['baseline_sha256']:
                if (item.get('continuation_parent_sha256')!=identity['parent_checkpoint_sha256'] or (item.get('continuation_parent') or {}).get('sha256')!=identity['parent_checkpoint_sha256'] or not _digest(item.get('predecessor_contract_sha256'))):
                    raise ValueError('SFT continuation parent or predecessor differs')
            if item['kind']=='jolt':
                _flag(argv,'--objective-spec')
                if item.get('on_policy_parent_sha256')!=identity['parent_checkpoint_sha256'] or not item.get('collector_receipt') or (item.get('on_policy_parent') or {}).get('sha256')!=item['on_policy_parent_sha256']:raise ValueError('JOLT requires exact on-policy parent and collector receipt')
            refs=item.get('dependencies')
            if not isinstance(refs,list) or not refs:raise ValueError('stage requires explicit source/objective dependencies')
            if item['kind']=='jolt' and item['collector_receipt'] not in refs:raise ValueError('collector receipt missing from role dependencies')
            required_paths=[_flag(argv,'--native-graphs'),_flag(argv,'--held-native-graphs'),binding.get('path')]
            if item['kind']=='jolt':required_paths.append(_flag(argv,'--objective-spec'))
            for optional in ('--foundation-native-graphs','--continued-native-graphs','--continued-held-native-graphs','--pending-focus','--ce-supervision-policy'):
                if optional in argv:required_paths.append(_flag(argv,optional))
            if any(not any(x.get('path')==path for x in refs) for path in required_paths):raise ValueError('stage inputs are missing exact dependency bindings')
            if not any(x.get('path')==binding.get('path') and x.get('sha256')==binding.get('file_sha256') for x in refs):raise ValueError('recipe file binding differs')
            digest=canonical_json_sha256_hex(item)
            check_files=verify_files and (only_digest is None or digest==only_digest)
            resolve = resolver or (lambda ref:Path(ref['path']))
            for ref in refs:
                if not isinstance(ref,dict) or not isinstance(ref.get('path'),str) or not Path(ref['path']).is_absolute() or not _digest(ref.get('sha256')):raise ValueError('invalid stage dependency')
                if check_files and sha256_file_hex_cached(resolve(ref))!=ref['sha256']:raise ValueError('stage dependency changed')
            if check_files and item['kind']=='sft' and item.get('source_transition'):
                transition=item['source_transition'];content_ref=transition.get('content_audit');protection_ref=transition.get('protection_audit')
                if content_ref not in refs or protection_ref not in refs:raise ValueError('source append admission proofs are missing')
                if not any(prov.get('content_audit')==content_ref and prov.get('clean_closure_audit')==protection_ref for prov in binding.get('external_provenances',[])):
                    raise ValueError('source append proofs differ from the bound recipe provenance')
                content=json.loads(resolve(content_ref).read_text());protection=json.loads(resolve(protection_ref).read_text())
                if content.get('schema')!='natlang.big2-multi-content-audit/1' or content.get('duplicates')!=[] or (content.get('base') or {}).get('sha256')!=transition.get('prior_source_sha256'):
                    raise ValueError('source append exact content audit failed')
                candidates={x['sha256'] for x in content.get('candidates',[]) if isinstance(x,dict) and _digest(x.get('sha256'))}
                source_ref=next(x for x in refs if x['path']==_flag(argv,'--native-graphs'));source=json.loads(resolve(source_ref).read_text())
                source_hashes={x.get('sha256') for x in source.get('sources',[]) if isinstance(x,dict)}
                protected=protection.get('sources') or {};base=protected.get('base') or {};results=[x for key,x in protected.items() if key.startswith('candidate-')]
                if (not candidates or not candidates<=source_hashes or protection.get('schema')!='natlang.fresh-restart-protection-union/1'
                        or (base.get('source') or {}).get('sha256')!=transition.get('prior_source_sha256')
                        or {((x.get('source') or {}).get('sha256')) for x in results}!=candidates
                        or any(x.get('conflicts') or x.get('conflict_counts') or x.get('ungrouped') for x in [base,*results])):
                    raise ValueError('source append protection union or candidate custody failed')
            if item['kind']=='sft' and item.get('optimizer_transition'):
                t=item['optimizer_transition']
                if (t.get('schema')!='natlang.diloco-optimizer-contract-transition/1' or t.get('verified') is not True
                        or not _digest(t.get('checkpoint_sha256')) or (t.get('from'),t.get('to')) not in (('stochastic','kahan'),('kahan','stochastic'))
                        or _flag(argv,'--muon-rounding')!=t.get('to')):
                    raise ValueError('invalid optimizer mode transition')
            if check_files and item['kind']=='sft' and item.get('objective_transition'):
                transition=item['objective_transition'];before=transition.get('old_policy');after=transition.get('new_policy')
                if (transition.get('schema')!='natlang.diloco-objective-contract-transition/1' or transition.get('verified') is not True
                        or not _digest(transition.get('checkpoint_sha256')) or before not in refs or after not in refs
                        or after.get('path')!=_flag(argv,'--ce-supervision-policy')):
                    raise ValueError('objective transition lacks exact policy and checkpoint bindings')
                old_doc=json.loads(resolve(before).read_text());new_doc=json.loads(resolve(after).read_text())
                from ..own_output_supervision import identity,policy_objective
                old_obj,new_obj=policy_objective(old_doc),policy_objective(new_doc)
                left,right=copy.deepcopy(old_obj),copy.deepcopy(new_obj)
                lf,rf=left.get('family_balance') or {},right.get('family_balance') or {}
                if (lf.get('key')!='logical-family' or rf.get('key')!='logical-family'
                        or lf.get('normalization')!='bounded-mean-one/1' or rf.get('normalization')!='declared/1'
                        or lf.get('map')!=rf.get('map') or (new_doc.get('objective_transition') or {}).get('from')!=identity(old_obj)['sha256']):
                    raise ValueError('objective transition differs from its measured family-scale predecessor')
                lf.pop('normalization');rf.pop('normalization');rf.pop('scales',None)
                if left!=right or {k:v for k,v in old_doc.items() if k not in ('objective','objective_transition')}!={k:v for k,v in new_doc.items() if k not in ('objective','objective_transition')}:
                    raise ValueError('objective transition changed unrelated CE law or custody')
                if not any(x.get('path')==lf['map']['path'] and x.get('sha256')==lf['map']['sha256'] for x in refs):
                    raise ValueError('objective transition family map missing')
            if check_files and item['kind']=='jolt':
                receipt=json.loads(resolve(item['collector_receipt']).read_text())
                by_path={x['path']:x['sha256'] for x in refs}
                if (receipt.get('schema')!='natlang.native-rl-admission/1'
                        or (receipt.get('policy') or {}).get('checkpoint_sha256')!=item['on_policy_parent_sha256']
                        or receipt.get('native_graphs_sha256')!=by_path[_flag(argv,'--native-graphs')]
                        or receipt.get('objective_sha256')!=by_path[_flag(argv,'--objective-spec')]
                        or type(receipt.get('episodes')) is not int or receipt['episodes']<1
                        or not receipt.get('rollouts')
                        or any(by_path.get(path)!=digest for path,digest in receipt['rollouts'].items())
                        or not receipt.get('tasks_selected')
                        or any(task.get('split')!='train' for task in receipt['tasks_selected'])):
                    raise ValueError('JOLT admission receipt does not bind on-policy training data')
            if check_files and verify_parent and item['kind']=='jolt' and sha256_file_hex_cached(Path(item['on_policy_parent']['path']))!=item['on_policy_parent_sha256']:
                raise ValueError('on-policy parent changed')
            if check_files and verify_parent and item['kind']=='sft' and item.get('continuation_parent') and sha256_file_hex_cached(Path(item['continuation_parent']['path']))!=item['continuation_parent_sha256']:
                raise ValueError('supervised continuation parent changed')
            if digest in seen:raise ValueError('duplicate role contract')
            seen.add(digest)
    return contract

def select(contract,worker,identity,*,verify_files=True):
    validate(contract,len(contract['roles']))
    matches=[item for item in contract['roles'][str(worker)] if item['identity']==dict(identity)]
    if len(matches)!=1:raise ValueError('worker trainer identity is not an allowed role contract')
    item=matches[0]
    if verify_files:
        validate(contract,len(contract['roles']),verify_files=True,only_digest=canonical_json_sha256_hex(item),verify_parent=True)
    return canonical_json_sha256_hex(item)


def resident_argv(original, payload, stage_id):
    """Bind the real running argv to the committed resident source and policy."""
    from ..recipe import HANDLERS,stage_input_args
    original=list(original);plan=payload['plan'];stage=next(x for x in plan['recipe']['stages'] if x['id']==stage_id)
    if stage['kind']!='native_state_sft':raise ValueError('resident role binding requires native SFT')
    roles=(HANDLERS['native_state_sft']['required_inputs'] | HANDLERS['native_state_sft']['optional_inputs'])-{'continue_from'}
    role_flags={'--'+x.replace('_','-') for x in roles};new_inputs=stage_input_args(plan['stage_inputs'][stage_id],'native_state_sft',target='portable')
    pairs=[(new_inputs[i],new_inputs[i+1]) for i in range(0,len(new_inputs),2) if new_inputs[i]!='--continue-from']
    updates={'--recipe-binding':json.dumps(payload['recipe'],sort_keys=True,separators=(',',':'))}
    for key in ('sft_packed_geometry','memory_qualification'):
        value=payload.get(key) if key=='sft_packed_geometry' and payload.get(key) is not None else stage['parameters'].get(key)
        if value is not None:
            updates['--'+key.replace('_','-')]=json.dumps(value,sort_keys=True,separators=(',',':'))
    out=[];inserted=False;seen=set();i=0
    while i<len(original):
        name=original[i]
        if name in role_flags:
            if i+1>=len(original):raise ValueError('resident role argument has no value')
            if not inserted:
                for flag,value in pairs:out.extend((flag,value))
                inserted=True
            i+=2;continue
        if name in updates:
            if name in seen or i+1>=len(original):raise ValueError('resident argument is duplicated or missing its value')
            out.extend((name,updates[name]));seen.add(name);i+=2;continue
        out.append(name);i+=1
    if not inserted:raise ValueError('resident argv omitted its source bindings')
    for key,value in updates.items():
        if key not in seen:out.extend((key,value))
    return out
