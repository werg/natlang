"""Authenticate complete groups harvested from an interrupted collection."""
from pathlib import Path
import json
from collections import Counter

from ..common.hashing import sha256_file_hex_cached
from ..common.jsonio import iter_jsonl_rows, read_jsonl


def resume_amended_collection(reference, request, *, validate_actors):
    from ..train.native_readiness import _bound
    proof_path = _bound(reference)
    proof = json.loads(proof_path.read_text())
    out = Path(request['out']).resolve(strict=True)
    if (proof.get('schema') not in {'natlang.amended-collection-continuation/1',
                                    'natlang.completed-collection-withdrawal/1'}
            or proof.get('out') != str(out)):
        raise ValueError('amended collection operation differs')
    files = proof.get('files') or {}
    required = {'collection-binding.json', 'score.json', 'rollouts.jsonl', 'packet.jsonl',
                'queue.jsonl', 'journal.jsonl', 'preparation.json', 'sampling-config.json'}
    if not required <= set(files):
        raise ValueError('amended collection omits original evidence')
    for name, ref in files.items():
        path = _bound(ref)
        if Path(name).is_absolute() or '..' in Path(name).parts or path != (out / name).resolve(strict=True) or path.stat().st_size != ref.get('bytes'):
            raise ValueError('amended collection evidence path or size changed')
    binding = json.loads((out/'collection-binding.json').read_text())
    report = json.loads((out/'score.json').read_text())
    completed=report.get('collection_complete') is True
    interrupted=report.get('collection_complete') is False
    completed_withdrawal=proof.get('schema')=='natlang.completed-collection-withdrawal/1'
    if (binding.get('schema') != 'natlang.rl-collection-binding/1'
            or not (interrupted and proof.get('schema')=='natlang.amended-collection-continuation/1'
                    or completed and completed_withdrawal)):
        raise ValueError('amendment must bind an interrupted collection or an explicitly withdrawn complete group')
    defaults = {'offset':0, 'group':4, 'root_mode':'student', 'seed_base':700000,
                'case_seconds':600, 'max_model_requests':128, 'max_turns':160, 'neuralese_server':True}
    expected = {key:request.get(key,default) for key,default in defaults.items()}
    original_cases=Path(request['cases'])
    if completed_withdrawal:
        source_ref=proof.get('source_cases') or {}
        source_path=_bound(source_ref)
        if type(source_ref.get('bytes')) is not int or source_path.stat().st_size!=source_ref['bytes']:
            raise ValueError('completed withdrawal source cases size differs from its binding')
        original_cases=source_path
    expected.update(policy_id=request['policy_id'], rows=request['rows'],
                    cases_sha256=sha256_file_hex_cached(original_cases),temperature=1.,guided=False)
    if any(binding.get(key)!=value for key,value in expected.items()):
        raise ValueError('amended collection policy or sampling differs')
    original_native=binding.get('expected_native_actor')
    comparable=lambda value:{k:v for k,v in (value or {}).items() if k!='python_runtime'}
    if comparable(original_native)!=comparable(request.get('expected_native')):
        raise ValueError('amended collection actor identity differs')
    if binding.get('coefficient_policy') or request.get('coefficient_policy') is not None:
        raise ValueError('amended coefficient policy is not supported')
    validate_actors(report.get('actor_receipts'),policy_id=binding['policy_id'],
                    neuralese_server=binding['neuralese_server'],expected_native=original_native,expected_coefficient=None)
    chosen=proof.get('groups')
    if (not isinstance(chosen,list) or not chosen or any(type(i) is not int or i<0 or i>=binding['rows'] for i in chosen)
            or chosen!=sorted(set(chosen))):
        raise ValueError('amended groups must be unique ordered packet indices')
    entries=list(read_jsonl(out/'queue.jsonl'))
    teacher_attempts=binding.get('jolt_teacher_attempts',0)
    if type(teacher_attempts) is not int or teacher_attempts<0:
        raise ValueError('collection binding has invalid privileged teacher attempt count')
    samples_per_group=binding['group']+teacher_attempts
    if len(entries)!=binding['rows']*samples_per_group:
        raise ValueError('original queue accounting differs')
    queues={(e['group'],e['sample']):e for e in entries}
    if len(queues)!=len(entries):raise ValueError('duplicate original queue sample')
    if completed:
        withdrawals=proof.get('withdrawals')
        if not isinstance(withdrawals,list) or not withdrawals:
            raise ValueError('completed collection withdrawal must name at least one excluded group')
        withdrawn=[item.get('group') for item in withdrawals if isinstance(item,dict)]
        if (len(withdrawn)!=len(withdrawals) or any(type(i) is not int or i<0 or i>=binding['rows'] for i in withdrawn)
                or withdrawn!=sorted(set(withdrawn)) or set(withdrawn)&set(chosen)
                or set(withdrawn)|set(chosen)!=set(range(binding['rows']))):
            raise ValueError('completed collection groups must be partitioned into retained and withdrawn rows')
        selected_path=out/'selected-cases.jsonl'
        if 'selected-cases.jsonl' not in files or _bound(files['selected-cases.jsonl'])!=selected_path.resolve(strict=True):
            raise ValueError('completed withdrawal does not bind the exact selected task rows')
        selected=list(read_jsonl(selected_path))
        packet=list(read_jsonl(out/'packet.jsonl'))
        if (len(selected)!=binding['rows'] or len(packet)!=binding['rows']
                or len({row.get('id') for row in selected})!=binding['rows']
                or [row.get('id') for row in selected]!=[row.get('id') for row in packet]
                or any(json.dumps(a,sort_keys=True,separators=(',',':'))!=json.dumps(b,sort_keys=True,separators=(',',':'))
                    for a,b in zip(selected,packet))):
            raise ValueError('completed withdrawal selected rows differ from the bound collection packet')
        original_rollouts=list(read_jsonl(out/'rollouts.jsonl'))
        if (len(original_rollouts)!=binding['rows']*samples_per_group
                or report.get('rollouts')!=len(original_rollouts)
                or report.get('scored')!=sum(row.get('reward') is not None for row in original_rollouts)):
            raise ValueError('completed collection score counts differ from its exact rollout rows')
        if len({(row.get('group'),row.get('sample')) for row in original_rollouts})!=len(original_rollouts):
            raise ValueError('completed collection repeats a rollout sample')
        original_by_sample={(row.get('group'),row.get('sample')):row for row in original_rollouts}
        for group in range(binding['rows']):
            student_rows=[original_by_sample.get((group,sample)) for sample in range(binding['group'])]
            if any(row is None for row in student_rows):
                raise ValueError('completed collection omits a student rollout')
            if sum(row.get('reward') is not None for row in student_rows)!=report['groups'][group].get('scored'):
                raise ValueError('completed group score differs from its student rollout outcomes')
            outcomes=report['groups'][group].get('outcomes')
            if (not isinstance(outcomes,list) or len(outcomes)!=binding['group']
                    or Counter(student_rows[sample].get('outcome') for sample in range(binding['group']))
                       !=Counter(outcomes)):
                raise ValueError('completed group outcome labels differ from exact student rollouts')
            if any((group,sample) not in original_by_sample
                   for sample in range(binding['group'],samples_per_group)):
                raise ValueError('completed collection omits a privileged rollout')
        by_id={row.get('id'):row for row in selected}
        from .task_source import TOOLACE_SCHEMA_CONTRACT, validate_current_materialized_task_contract
        from .task_source import _canonical_row_sha256
        source_rows={};source_counts={}
        withdrawn_ids={item.get('task_id') for item in withdrawals}
        for _,row in iter_jsonl_rows(original_cases):
            if row.get('id') in withdrawn_ids:
                source_rows[row['id']]=row
                source_counts[row['id']]=source_counts.get(row['id'],0)+1
        if len(source_rows)!=len(withdrawals) or any(source_counts.get(task_id)!=1 for task_id in withdrawn_ids):
            raise ValueError('withdrawn task is absent or duplicated in the source packet bound by the collection')
        for item in withdrawals:
            group=item['group']
            rows_for_group=[entry for (idx,_),entry in queues.items() if idx==group]
            task_id=rows_for_group[0].get('task_id') if rows_for_group else None
            case=by_id.get(task_id);source_case=source_rows.get(task_id)
            if (not case or not source_case or task_id!=item.get('task_id')
                    or _canonical_row_sha256(case)!=item.get('row_sha256')
                    or _canonical_row_sha256(source_case)!=item.get('row_sha256')):
                raise ValueError('withdrawal proof does not bind the exact source task')
            if item.get('reason')=='stale-toolace-schema-contract':
                external=((case.get('curriculum') or {}).get('external') or {})
                if (external.get('source')!='toolace' or TOOLACE_SCHEMA_CONTRACT in case.get('source_revisions',[])
                        or item.get('required_schema_contract')!=TOOLACE_SCHEMA_CONTRACT
                        or (report['groups'][group].get('scored')!=binding['group'])):
                    raise ValueError('withdrawal reason does not match a complete stale ToolACE task')
            elif item.get('reason')=='incomplete-native-execution-group':
                unresolved=item.get('unresolved_samples')
                if not isinstance(unresolved,list) or not unresolved:
                    raise ValueError('incomplete group withdrawal omits exact unresolved execution evidence')
                scored_group=report['groups'][group]
                if (scored_group.get('task_id')!=task_id or scored_group.get('scored')>=binding['group']
                        or scored_group.get('outcomes',[]).count('unresolved')!=1):
                    raise ValueError('incomplete group withdrawal does not match an unresolved scored outcome')
                seen_unresolved=set()
                for failure in unresolved:
                    sample=failure.get('sample')
                    if type(sample) is not int or not 0<=sample<binding['group'] or sample in seen_unresolved:
                        raise ValueError('unresolved group sample index is invalid')
                    seen_unresolved.add(sample)
                    q=queues.get((group,sample));roll=next((r for r in read_jsonl(out/'rollouts.jsonl')
                        if r.get('group')==group and r.get('sample')==sample),None)
                    error_path=(out/failure.get('error_path','')).resolve(strict=True)
                    bound_error=files.get(error_path.relative_to(out).as_posix())
                    if (not q or roll is None or roll.get('outcome')!='unresolved'
                            or roll.get('unscored')!='unresolved' or roll.get('reward') is not None
                            or roll.get('turns') or Path(q.get('output','')).stat().st_size!=0
                            or not bound_error or _bound(bound_error)!=error_path
                            or bound_error.get('sha256')!=failure.get('error_sha256')):
                        raise ValueError('unresolved group proof does not match its exact empty output and error artifact')
                    error=json.loads(error_path.read_text())
                    terminal=None
                    for event in read_jsonl(out/'journal.jsonl'):
                        if event.get('event')!='finish' or event.get('key')!=q.get('key'):continue
                        states=(event.get('output_accounting') or {}).get('job_states') or []
                        terminal=next((state.get('terminal_error') for state in states
                            if state.get('terminal_error')),None)
                        break
                    if (error.get('error')!=failure.get('error')
                            or error.get('program_ir_sha256')!=failure.get('program_ir_sha256')
                            or error.get('source_sha256')!=binding.get('cases_sha256')
                            or error.get('source_sha256')!=failure.get('source_sha256')
                            or 'neuralese-native-grad-invalid' not in failure.get('error','')
                            or terminal is None or terminal.get('path')!=str(error_path)
                            or terminal.get('sha256')!=bound_error.get('sha256')
                            or terminal.get('error')!=failure.get('error')
                            or terminal.get('program_ir_sha256')!=failure.get('program_ir_sha256')
                            or terminal.get('source_sha256')!=binding.get('cases_sha256')
                            or terminal.get('record_binding_verified') is not True):
                        raise ValueError('unresolved failure is not the bound native export custody error')
                actual_unresolved={sample for sample in range(binding['group'])
                    if any(row.get('group')==group and row.get('sample')==sample and row.get('outcome')=='unresolved'
                        for row in read_jsonl(out/'rollouts.jsonl'))}
                if seen_unresolved!=actual_unresolved:
                    raise ValueError('withdrawn unresolved sample list differs from the full source trajectory')
                if (len(scored_group.get('outcomes',[]))!=binding['group']
                        or sum(outcome!='unresolved' for outcome in scored_group['outcomes'])!=scored_group['scored']):
                    raise ValueError('incomplete withdrawal score does not account for every student sample')
            else:
                raise ValueError('unsupported completed collection withdrawal reason')
        for group in chosen:
            case=by_id.get(queues[(group,0)]['task_id'])
            if case is None:raise ValueError('retained task is absent from selected cases')
            validate_current_materialized_task_contract(case)
        if len(report.get('groups',[]))!=binding['rows']:
            raise ValueError('complete collection score omits a selected group')
        for group in range(binding['rows']):
            scored_group=report['groups'][group]
            if scored_group.get('task_id')!=queues[(group,0)].get('task_id'):
                raise ValueError('complete collection score differs from the bound queue groups')
            for sample in range(samples_per_group):
                entry=queues.get((group,sample))
                if (not entry or entry.get('index')!=group or entry.get('task_id')!=scored_group.get('task_id')
                        or ((sample<binding['group']) and entry.get('jolt_role')=='teacher')
                        or ((sample>=binding['group']) and entry.get('jolt_role')!='teacher')
                        or str(Path(entry['output']).resolve().relative_to(out)) not in files):
                    raise ValueError('complete collection is missing an authenticated queue result')
            if group in chosen and scored_group.get('scored')!=binding['group']:
                raise ValueError('retained completed group does not have all student samples scored')
            if group in chosen:
                for sample in range(binding['group'],samples_per_group):
                    teacher=original_by_sample[(group,sample)]
                    if (teacher.get('jolt_role')!='teacher' or teacher.get('outcome')!='task_success'
                            or (teacher.get('reward') or {}).get('quality')!=1
                            or not teacher.get('turns')):
                        raise ValueError('retained privileged teacher row is not a successful bound trajectory')
    for group in chosen:
        for sample in range(samples_per_group):
            entry=queues.get((group,sample))
            if not entry or entry.get('index')!=group:
                raise ValueError('amended group lacks a queue sample')
            path=Path(entry['output']).resolve(strict=True)
            name=str(path.relative_to(out))
            if name not in files or _bound(files[name])!=path:
                raise ValueError('amended group lacks bound completed result')
    filtered=Path(proof['filtered_rollouts']['path']).resolve(strict=True)
    if _bound(proof['filtered_rollouts'])!=filtered or filtered.stat().st_size!=proof['filtered_rollouts']['bytes']:
        raise ValueError('amended rollouts changed')
    # Verify the filtered bytes are exactly the ordered projection of the
    # authenticated original, not merely samples with matching identifiers.
    with (out/'rollouts.jsonl').open('rb') as original, filtered.open('rb') as projected:
        for line in original:
            row=json.loads(line)
            if row.get('group') in chosen and projected.readline()!=line:
                raise ValueError('amended rollout differs from original projection')
        if projected.read(1):raise ValueError('amended rollout has extra data')
    rows=list(read_jsonl(filtered))
    if len(rows)!=len(chosen)*samples_per_group:
        raise ValueError('amended rollout count differs')
    actual={(r.get('group'),r.get('sample')) for r in rows}
    if actual!={(g,s) for g in chosen for s in range(samples_per_group)}:
        raise ValueError('amended rollout sample set differs')
    if any(r.get('task_id') != queues[(r['group'],r['sample'])].get('task_id') for r in rows):
        raise ValueError('amended rollout task differs from original queue')
    if any((r.get('policy') or {}).get('checkpoint_sha256')!=binding['policy_id'] or r.get('reward') is None or not r.get('turns') for r in rows):
        raise ValueError('amended rollout lacks same-policy scored execution')
    from .graph_replay import block_ids, NativeBundleStore
    ids=set(block_ids(rows))
    if ids:
        manifest=Path(report.get('block_bundle','')).resolve(strict=True)
        if manifest!=(out/'blocks/manifest.json').resolve(strict=True) or 'blocks/manifest.json' not in files:
            raise ValueError('amended rollout lacks exact native bundle')
        store=NativeBundleStore(manifest,policy_id=binding['policy_id'])
        for ident in ids:
            record=store.get_native_state(ident) if ident.startswith('nzs1_') else store.get_native_tensor_artifact(ident)
            if record is None:raise ValueError('amended native artifact missing')
    groups=[report['groups'][i] for i in chosen]
    if any(g.get('scored')!=binding['group'] for g in groups):raise ValueError('amended group is not completely scored')
    completed_groups=[report['groups'][g] for g in chosen]
    return {**report,'collection_complete':True if completed else False,
            'historical_collection_complete':completed,
            'groups':groups,'groups_with_spread':sum(bool(g.get('spread')) for g in completed_groups),
            'scored':len(rows),'successes':sum(r['reward'].get('quality')==1 for r in rows),
            'rollouts':len(rows),'amended_complete_groups':chosen,'amended_rollouts':str(filtered),
            'withdrawn_groups':proof.get('withdrawals',[]) if completed else [],
            'amendment_proof':str(proof_path)}
