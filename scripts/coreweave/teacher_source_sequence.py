"""Exact lifecycle/publication plumbing for a supervisor-declared task sequence."""
from __future__ import annotations
import fcntl
import hashlib
import json
from pathlib import Path
import time
try:
    from .teacher_generation_guardian import binding, write, alive, primary_ahead_slots
except ImportError:
    from teacher_generation_guardian import binding, write, alive, primary_ahead_slots


def _read(reference, declaration):
    replacement=(declaration.get('assets') or {}).get(reference['path'],reference)
    if any(replacement.get(key)!=reference.get(key) for key in ('sha256','bytes')):
        raise ValueError('sequence asset relocation changed its bytes')
    path=Path(replacement['path'])
    raw=path.read_bytes()
    if len(raw)!=replacement['bytes'] or hashlib.sha256(raw).hexdigest()!=replacement['sha256']:
        raise ValueError('sequence asset binding changed: '+str(path))
    return path,raw


def _asset(reference,declaration):
    return _read(reference,declaration)[0]


def _reference(reference,declaration):
    return {**reference,'path':str(_asset(reference,declaration))}


def _json(reference, declaration):
    return json.loads(_read(reference,declaration)[1])


def _immutable(path, value):
    if path.exists():
        actual=path.read_bytes() if isinstance(value,bytes)else json.loads(path.read_text())
        if actual!=value:raise ValueError('sequence immutable receipt changed: '+str(path))
    else:write(path,value)


def _drained_boundary(folder, state, config, batch, terminal):
    """Keep incomplete original outcomes while authenticating their repair lane."""
    if terminal.get('batch') != batch:
        raise ValueError('source selection needs the actual preceding completed boundary')
    if terminal.get('batch_complete') is True:
        return None  # Preserve the existing successful-boundary contract.
    declaration = config['source_sequence']
    reference = state['last_boundary']
    expected = Path(folder)/'boundaries'/f'batch-{batch:04d}.json'
    if (terminal.get('schema') != 'natlang.teacher-generation-boundary/1'
            or Path(reference['path']).resolve() != expected.resolve()):
        raise ValueError('sequence incomplete boundary lacks its actual guardian authority')
    source = Path(config['root'])/'evidence-collections-v2'/f'batch-{batch:04d}'
    refs = terminal['bindings']
    for name in ('queue.jsonl', 'journal.jsonl', 'score.json', 'teacher-source.json'):
        if Path(refs[name]['path']).resolve() != (source/name).resolve():
            raise ValueError('sequence boundary belongs to another source batch')
        _read(refs[name], declaration)
    entries = [json.loads(line) for line in _read(refs['queue.jsonl'], declaration)[1].splitlines() if line]
    journal = [json.loads(line) for line in _read(refs['journal.jsonl'], declaration)[1].splitlines() if line]
    try:
        from .teacher_batch_completion import batch_completion
    except ImportError:
        from teacher_batch_completion import batch_completion
    completion = batch_completion(entries, journal, terminal['completion']['worker_returncodes'],
                                  _json(refs['score.json'], declaration))
    exact = lambda value: json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
    if (exact(completion) != exact(terminal['completion'])
            or exact(completion['repair_required']) != exact(terminal['repair_required'])
            or completion['can_advance'] is not False):
        raise ValueError('sequence incomplete boundary accounting changed')
    wanted = [row['key'] for row in completion['repair_required']]
    if not wanted or None in wanted or len(wanted) != len(set(wanted)):
        raise ValueError('sequence boundary has unassigned repair obligations')
    originals = {row['key']: row for row in entries}
    repair = Path(folder)/'repairs'/f'batch-{batch:04d}'
    repair_ref = binding(repair/'queue.jsonl')
    retries = [json.loads(line) for line in _read(repair_ref, declaration)[1].splitlines() if line]
    if len(retries) != len(wanted):
        raise ValueError('sequence repair queue omits original obligations')
    by_key = {row['key']: row for row in retries}
    if len(by_key) != len(retries) or set(by_key) != {f'batch-{batch:04d}:'+key for key in wanted}:
        raise ValueError('sequence repair queue differs from original obligations')
    prepared_refs = []
    for key in wanted:
        if key not in originals or Path(key).name != key or key in ('.', '..'):
            raise ValueError('sequence repair key escaped its original queue')
        prepared_ref = binding(repair/key/'prepared.json')
        prepared = _json(prepared_ref, declaration)
        retry = by_key[f'batch-{batch:04d}:'+key]
        leaf = repair/key
        expected_retry = {**originals[key], 'key':f'batch-{batch:04d}:'+key,
            'jobs':str(leaf/'jobs'), 'output':str(leaf/'results.jsonl'), 'log':str(leaf/'collector.log')}
        if (exact(prepared.get('original')) != exact(originals[key])
                or exact(prepared.get('retry')) != exact(expected_retry)
                or exact(retry) != exact(expected_retry)
                or prepared.get('boundary') != reference):
            raise ValueError('sequence repair changed its original sample/source/seed')
        prepared_refs.append(prepared_ref)
    receipt = Path(folder)/'source-sequence/drained-boundaries'/f'batch-{batch:04d}.json'
    _immutable(receipt, dict(batch=batch, boundary=reference, original_bindings=refs,
        repair_queue=repair_ref, repair_preparations=prepared_refs, repair_required_keys=wanted,
        batch_complete=False, training_admission=False,
        claim='Original batch drained with exact repair debt; no completion or success reclassification'))
    return binding(receipt)


def collection_sampling(configuration):
    """Exact optional source declaration; legacy sources keep their sampler."""
    if 'collection_sampling' not in configuration:
        return {'group':2, 'root_mode':'student'}
    value=configuration['collection_sampling']
    if (not isinstance(value,dict) or set(value)!={'group','root_mode'}
            or type(value['group'])is not int or value['group']<2
            or value['root_mode'] not in ('student','authored')):
        raise ValueError('source collection sampling declaration is malformed')
    return dict(value)


def _jsonl_entries(path):
    return [json.loads(line)for line in Path(path).read_bytes().splitlines()if line]


def verify_collection_sampling(source,entries,configuration):
    """Authenticate new explicit windows without reinterpreting legacy queues."""
    if 'collection_sampling' not in configuration:return
    sampling=collection_sampling(configuration)
    if (source.get('collection_sampling')!=sampling
            or source.get('root_mode')!=sampling['root_mode']
            or source.get('source_provenance')!=configuration):
        raise ValueError('started collection sampling differs from its declared source')
    rows=configuration.get('rows');seed=source.get('seed_base');group=sampling['group']
    if type(rows)is not int or rows<1 or type(seed)is not int or len(entries)!=rows*group:
        raise ValueError('collection queue differs from its declared sample count')
    seen=set();keys=set()
    for row in entries:
        index=row.get('index');sample=row.get('sample')
        if (not isinstance(row.get('key'),str)or not row['key']or row['key']in keys
                or type(index)is not int or not 0<=index<rows
                or type(sample)is not int or not 0<=sample<group
                or (index,sample)in seen or type(row.get('group'))is not int or row['group']!=index
                or type(row.get('count'))is not int or row['count']!=1
                or type(row.get('seed'))is not int or row['seed']!=seed+1000*index+sample
                or row.get('temperature')!=source.get('temperature')
                or row.get('text_neuralese_emulation')is not True):
            raise ValueError('collection queue changed its declared case/sample/seed partition')
        seen.add((index,sample));keys.add(row['key'])


def _packet(entry, declaration, *, allow_empty=False):
    candidate=_json(entry['candidate_packet'],declaration)
    prepared=_json(entry['qualified_packet'],declaration)
    qualification=_json(prepared['qualification'],declaration)
    inputs=_json(qualification['inputs'],declaration)
    sampling_fields={}
    if 'collection_sampling' in candidate or 'collection_sampling' in prepared:
        if ('collection_sampling' not in candidate or 'collection_sampling' not in prepared
                or candidate['collection_sampling']!=prepared['collection_sampling']):
            raise ValueError('candidate and prepared collection sampling disagree')
        if collection_sampling(candidate)!=collection_sampling(prepared):
            raise ValueError('candidate and prepared collection sampling disagree')
        sampling_fields={'collection_sampling':collection_sampling(candidate),
            'collection_sampling_authority':{
                'candidate_packet':_reference(entry['candidate_packet'],declaration),
                'qualified_packet':_reference(entry['qualified_packet'],declaration)}}
    refs=prepared['references']
    selection=_json(refs['selection'],declaration)
    protection=_json(refs['global_protection_join'],declaration)
    cases,cases_raw=_read(refs['selected_cases'],declaration)
    if prepared['cases']!=refs['selected_cases'] or qualification['qualified_cases_binding']!=prepared['cases']:
        raise ValueError('sequence qualified packet cases disagree')
    authority=declaration['authority']
    runtime_digest=hashlib.sha256(json.dumps(inputs['runtime'],ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    runtimes=authority.get('reference_runtime_by_candidate',{})
    valid_sha=lambda value:isinstance(value,str)and len(value)==64 and all(c in '0123456789abcdef'for c in value)
    if not isinstance(runtimes,dict)or any(not valid_sha(k)or not valid_sha(v)for k,v in runtimes.items()):
        raise ValueError('sequence reference runtime bindings are malformed')
    expected_runtime=runtimes.get(entry['candidate_packet']['sha256'],authority['reference_runtime_sha256'])
    protections=authority.get('reference_protection_by_candidate',{})
    if not isinstance(protections,dict):
        raise ValueError('sequence candidate protection bindings are malformed')
    for key,value in protections.items():
        if (not valid_sha(key)or not isinstance(value,dict)
                or set(value)!={'protection','decision_hold_bindings'}
                or not isinstance(value['protection'],dict)
                or not isinstance(value['protection'].get('path'),str)
                or not valid_sha(value['protection'].get('sha256'))
                or not isinstance(value['decision_hold_bindings'],list)
                or any(not isinstance(ref,dict)or not isinstance(ref.get('path'),str)
                       or not valid_sha(ref.get('sha256'))or type(ref.get('bytes'))is not int
                       or ref['bytes']<0 for ref in value['decision_hold_bindings'])):
            raise ValueError('sequence candidate protection bindings are malformed')
    expected_authority=protections.get(entry['candidate_packet']['sha256'],authority)
    if (qualification.get('schema')!='natlang.existing-native-reference-qualification/1'
            or qualification.get('passed')is not True or qualification.get('model_calls')!=0
            or qualification.get('training_admission')is not False
            or qualification.get('reference_traces_are_teacher_generations')is not False
            or qualification.get('runtime_inventory_sha256')!=expected_runtime
            or runtime_digest!=expected_runtime
            or inputs['authority']['protection']!=expected_authority['protection']
            or inputs['authority']['decision_holds']!=expected_authority['decision_hold_bindings']
            or selection.get('protection')!=expected_authority['protection']
            or selection.get('decision_hold_bindings')!=expected_authority['decision_hold_bindings']):
        raise ValueError('sequence reference/protection authority differs')
    if (selection.get('schema')!='natlang.strong-teacher-code-source-selection/1'
            or selection.get('training_admission')is not False
            or selection.get('held_and_pending_intersections')!=0
            or selection.get('cases_sha256')!=refs['selected_cases']['sha256']
            or selection.get('reference_qualification')!=prepared['qualification']
            or protection.get('schema')!='natlang.native-teacher-global-protection-join/1'
            or protection.get('status')!='passed'
            or protection.get('selected_packet_sha256')!=refs['selected_cases']['sha256']
            or protection.get('current_protection')!=expected_authority['protection']
            or protection.get('decision_hold_bindings')!=expected_authority['decision_hold_bindings']
            or any(protection.get(key) for key in ('held_group_intersections','quality_group_intersections','pending_group_intersections'))):
        raise ValueError('sequence packet selection/protection proof differs')
    _,candidate_raw=_read(candidate['cases_binding'],declaration)
    if inputs['cases']!=candidate['cases_binding']:raise ValueError('reference inputs differ from declared candidate packet')
    originals={}
    for raw in candidate_raw.splitlines():
        if raw:originals[hashlib.sha256(raw).hexdigest()]=raw
    if len(originals)!=candidate['cases'] or qualification.get('candidate_cases')!=len(originals):
        raise ValueError('sequence original candidate count differs')
    verdicts={}
    for line in _read(qualification['verdicts'],declaration)[1].splitlines():
        row=json.loads(line);key=row['program_sha256']
        if (key in verdicts or row.get('model_calls')!=0 or row.get('training_admission')is not False
                or row.get('reference_traces_are_teacher_generations')is not False):
            raise ValueError('duplicate or non-reference sequence verdict')
        verdicts[key]=row
    if set(verdicts)!=set(originals):raise ValueError('sequence reference verdict inventory differs')
    expected=[raw for key,raw in originals.items()if verdicts[key].get('passed')is True]
    actual=[raw for raw in cases_raw.splitlines()if raw]
    if (not actual and not allow_empty) or actual!=expected:raise ValueError('sequence cases are not the complete unchanged qualified subset')
    programs=[json.loads(raw)for raw in actual]
    if (selection.get('cases')!=len(programs) or qualification.get('qualified_cases')!=len(programs)
            or len({row.get('id')for row in programs})!=len(programs)
            or any(row.get('version')!='natlang.program/2' or row.get('split')!='train'
                   or not row.get('source_groups') for row in programs)):
        raise ValueError('sequence packet program identity/count/split differs')
    return {**sampling_fields,'cases':str(cases),'cases_sha256':refs['selected_cases']['sha256'],'rows':len(programs),
        'protected_selection':selection,'selection':_reference(refs['selection'],declaration),
        'global_protection_join':_reference(refs['global_protection_join'],declaration),
        'reference_qualification':_reference(prepared['qualification'],declaration),
        'claim':'unchanged reference-qualified TRAIN tasks; generation and training admission remain separate',
        'training_admission':False}


def _reconcile_publication(pending, declaration):
    intent=_json(pending['intent'],declaration)
    target=Path(declaration['source_configuration'])
    current=binding(target)
    if current['sha256']==intent['new_configuration']['sha256']:
        return
    if current!=intent['old_configuration']:raise ValueError('sequence source configuration changed outside its intent')
    value=_json(intent['new_configuration'],declaration)
    write(target,value)
    if binding(target)['sha256']!=intent['new_configuration']['sha256']:
        raise ValueError('sequence source configuration publication failed')


def observe_source_sequence(folder, state, config):
    declaration=config.get('source_sequence');sequence=state.get('source_sequence')or{}
    pending=sequence.get('pending')
    if not declaration or not pending:return
    batch=pending['batch'];root=Path(config['root'])/'evidence-collections-v2'/f'batch-{batch:04d}'
    names=('source-cases.jsonl','queue.jsonl','teacher-source.json')
    if not all((root/name).is_file()for name in names):return
    sealed={name:binding(root/name)for name in names}
    source=json.loads((root/'teacher-source.json').read_text())
    if (sealed['source-cases.jsonl']['sha256']!=pending['cases_sha256']
            or source.get('source_cases_sha256')!=pending['cases_sha256']):
        raise ValueError('sequence started batch differs from its declared packet')
    intent=_json(pending['intent'],declaration)
    configuration=_json(intent['new_configuration'],declaration)
    if 'collection_sampling'in configuration:
        verify_collection_sampling(source,_jsonl_entries(root/'queue.jsonl'),configuration)
    key=str(batch);started=sequence.setdefault('started',{})
    if key not in started:
        receipt=Path(folder)/'source-sequence/started'/f'batch-{batch:04d}.json'
        _immutable(receipt,{'batch':batch,'ordinal':pending['ordinal'],'sealed':sealed,
            'selection_intent':pending['intent'],'claim':'actual sealed batch; no generation admission/exposure'})
        started[key]=binding(receipt);write(Path(folder)/'state.json',state)
    elif _json(started[key],declaration)['sealed']!=sealed:
        raise ValueError('sequence started batch artifacts changed')
    intent=_json(pending['intent'],declaration)
    if intent.get('prefetch')is not None:
        previous=sequence.get('closed',{}).get(str(batch-1))
        if not previous or _json(previous,declaration).get('started')!=intent['prefetch']['predecessor_started']:
            raise ValueError('prefetched source publication precedes its genuine predecessor closure')
        _reconcile_publication(pending,declaration)
    current=binding(declaration['source_configuration'])
    if current['sha256']!=intent['new_configuration']['sha256']:
        raise ValueError('started source configuration differs from publication intent')
    sequence['configuration']=current
    boundary=state.get('last_boundary')
    if boundary is None:return
    terminal=_json(boundary,declaration)
    if terminal.get('batch')!=batch:return
    closure = _drained_boundary(folder, state, config, batch, terminal)
    if any(terminal['bindings'].get(name)!=sealed[name]for name in ('queue.jsonl','teacher-source.json')):
        raise ValueError('sequence terminal boundary differs from its sealed batch')
    receipt=Path(folder)/'source-sequence/closed'/f'batch-{batch:04d}.json'
    _immutable(receipt,{'batch':batch,'ordinal':pending['ordinal'],'started':started[key],
        'boundary':boundary, **({'drained_with_repair_debt':closure} if closure else {}),
        'claim':'existing guardian closure; individual outcomes/admission unchanged'})
    sequence.setdefault('closed',{})[key]=binding(receipt)
    priority=intent.get('priority')
    if priority is not None:
        if (type(priority.get('resume_ordinal'))is not int
                or not 0<=priority['resume_ordinal']<pending['ordinal']):
            raise ValueError('priority packet lost its original ordinary cursor')
        sequence['next_ordinal']=priority['resume_ordinal']
        sequence['priority_streak']=sequence.get('priority_streak',0)+1
    else:
        sequence['next_ordinal']=pending['ordinal']+1
        if 'priority_streak'in sequence:sequence['priority_streak']=0
    sequence.pop('pending')
    write(Path(folder)/'state.json',state)


def _source_context_deferrals(declaration, entries, sequence):
    # Debt survives changes to the current supervisor declaration. Authenticate
    # every stored immutable receipt and its original proof even after its
    # ordinal has passed; removing a current deferral cannot erase that history.
    for key,stored in sequence.get('context_deferred',{}).items():
        receipt=_json(stored,declaration)
        if (not isinstance(key,str)or not key.isdecimal()or str(int(key))!=key
                or type(receipt.get('ordinal'))is not int or receipt['ordinal']!=int(key)
                or receipt['ordinal']>=sequence['next_ordinal']
                or receipt.get('training_admission')is not False
                or receipt.get('generation_admission')is not False):
            raise ValueError('stored context deferral debt changed')
        _json(receipt['context_exclusions'],declaration)
    reference=declaration.get('context_deferrals')
    if reference is None:return None,{},{}
    proof=_json(reference,declaration)
    if (proof.get('schema')!='natlang.teacher-source-context-exclusions/1'
            or proof.get('training_admission')is not False or type(proof.get('model_calls'))is not int
            or proof['model_calls']!=0):
        raise ValueError('unknown teacher context deferral authority')
    original=_json(proof['original_ready'],declaration)
    exact=lambda value:json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'))
    if len(entries)<len(original['entries'])or exact(original['entries'])!=exact(entries[:len(original['entries'])]):
        raise ValueError('context deferrals differ from exact observed ready entries')
    for name in ('source_context_audit','primary_source_audit','original_context_source'):
        _json(proof[name],declaration)
    programs={}
    for item in proof['excluded_programs']:
        ident=item.get('program_id');digest=item.get('program_sha256')
        if (not isinstance(ident,str)or not ident or ident in programs or not isinstance(digest,str)
                or len(digest)!=64 or any(c not in '0123456789abcdef'for c in digest)):
            raise ValueError('invalid exact context exclusion program')
        programs[ident]=item
    deferred={}
    started={_json(ref,declaration)['ordinal']for ref in sequence.get('started',{}).values()}
    pending=sequence.get('pending')
    for item in proof['deferred_packets']:
        ordinal=item.get('ordinal')
        if (type(ordinal)is not int or ordinal<0 or ordinal>=len(original['entries'])or ordinal in deferred
                or ordinal in started or(pending and pending['ordinal']==ordinal)):
            raise ValueError('context deferral must name unique unstarted packets only')
        entry=entries[ordinal]
        if any(exact(item[name])!=exact(entry[name])for name in ('candidate_packet','qualified_packet')):
            raise ValueError('context deferral packet binding differs')
        if ordinal<sequence['next_ordinal']and str(ordinal)not in sequence.get('context_deferred',{}):
            raise ValueError('context deferral cannot reclassify past work')
        deferred[ordinal]=item
    return reference,deferred,programs


def _defer_context_packet(folder,state,declaration,entry,item,reference,programs):
    prepared=_json(entry['qualified_packet'],declaration)
    if item['cases']!=prepared['cases']:raise ValueError('context deferral cases binding differs')
    raw=_read(item['cases'],declaration)[1]
    rows=[json.loads(line)for line in raw.splitlines()if line.strip()]
    identifiers=[row['id']for row in rows]
    excluded=item['excluded_program_ids']
    if (not isinstance(excluded,list)or not excluded or len(set(excluded))!=len(excluded)
            or any(ident not in identifiers or ident not in programs for ident in excluded)
            or len(set(identifiers))!=len(identifiers)):
        raise ValueError('context deferral excludes unknown or repeated task identities')
    for row in rows:
        if row['id']not in excluded:continue
        digest=hashlib.sha256(json.dumps(row,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if digest!=programs[row['id']]['program_sha256']:
            raise ValueError('context deferral original program changed')
    counts={'source_cases':len(rows),'excluded_cases':len(excluded),'retained_cases':len(rows)-len(excluded)}
    if any(type(item.get(key))is not int or item[key]!=value for key,value in counts.items()):
        raise ValueError('context deferral loses unaffected sibling obligations')
    receipt=Path(folder)/'source-sequence/context-deferred'/f"ordinal-{entry['ordinal']:04d}.json"
    _immutable(receipt,{'ordinal':entry['ordinal'],'candidate_packet':entry['candidate_packet'],
        'qualified_packet':entry['qualified_packet'],'cases':item['cases'],'context_exclusions':reference,
        'excluded_program_ids':excluded,**counts,'generation_admission':False,'training_admission':False,
        'debt':'Unaffected sibling cases require fresh reference qualification and a new appended packet identity; original task context targets remain excluded.'})
    sequence=state['source_sequence'];stored=sequence.setdefault('context_deferred',{});key=str(entry['ordinal'])
    actual=binding(receipt)
    if key in stored and stored[key]!=actual:raise ValueError('context deferral receipt changed')
    stored[key]=actual;sequence['next_ordinal']=entry['ordinal']+1
    write(Path(folder)/'state.json',state)


def _priority_policy(reference,declaration,entries):
    proof=_json(reference,declaration)
    original=_json(proof['original_ready'],declaration)
    exact=lambda value:json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'))
    if (proof.get('schema')!='natlang.teacher-priority-packet-continuation/1'
            or proof.get('training_admission')is not False
            or type(proof.get('minimum_original_ordinal'))is not int
            or not 0<=proof['minimum_original_ordinal']<=len(original['entries'])
            or type(proof.get('max_consecutive'))is not int or proof['max_consecutive']<1
            or exact(entries[:len(original['entries'])])!=exact(original['entries'])):
        raise ValueError('priority packet authority differs from retained original work')
    selected={}
    for item in proof['entries']:
        ordinal=item.get('ordinal')
        if (type(ordinal)is not int or ordinal<len(original['entries'])or ordinal>=len(entries)
                or ordinal in selected
                or any(item.get(k)!=entries[ordinal].get(k)for k in('candidate_packet','qualified_packet'))):
            raise ValueError('priority packet must be an exact newly appended entry')
        selected[ordinal]=item
    return proof,selected


def _closed_packet_ordinals(folder,sequence,declaration,entries):
    result={}
    for key,reference in sequence.get('closed',{}).items():
        closed=_json(reference,declaration);ordinal=closed['ordinal'];batch=closed['batch']
        if (key!=str(batch)or type(batch)is not int or type(ordinal)is not int
                or not 0<=ordinal<len(entries)or ordinal in result
                or Path(reference['path']).resolve()!=(Path(folder)/'source-sequence/closed'/f'batch-{batch:04d}.json').resolve()):
            raise ValueError('closed packet cursor authority changed')
        started=_json(closed['started'],declaration);intent=_json(started['selection_intent'],declaration)
        if (started.get('batch')!=batch or intent.get('batch')!=batch
                or started.get('ordinal')!=ordinal or intent.get('ordinal')!=ordinal
                or any(intent.get(k)!=entries[ordinal].get(k)for k in('candidate_packet','qualified_packet'))):
            raise ValueError('closed packet differs from its actual selected entry')
        priority=intent.get('priority')
        if priority is not None:
            proof,selected=_priority_policy(priority['policy'],declaration,entries)
            resume=priority.get('resume_ordinal')
            if (ordinal not in selected or type(resume)is not int
                    or not proof['minimum_original_ordinal']<=resume<ordinal):
                raise ValueError('closed priority packet lost its original cursor/policy')
        result[ordinal]=dict(reference=reference,intent=intent)
    return result


def _selection_authority(folder,sequence,declaration,*,extra_service=()):
    manifest_path=Path(declaration['ready_manifest']);manifest_raw=manifest_path.read_bytes();manifest=json.loads(manifest_raw)
    if manifest.get('schema')!='natlang.strong-teacher-ready-packet-sequence/1':raise ValueError('unknown ready sequence')
    if manifest['candidate_inventory']!=declaration['candidate_inventory']:
        raise ValueError('sequence candidate inventory differs from supervisor declaration')
    inventory=_json(manifest['candidate_inventory'],declaration)
    entries=manifest['entries'];previous=sequence.get('observed_entries')or[]
    if entries[:len(previous)]!=previous:raise ValueError('ready sequence changed its declared prefix')
    if len(entries)>len(inventory['packets']):raise ValueError('ready sequence exceeds declared candidates')
    for index,entry in enumerate(entries):
        if entry.get('ordinal')!=index or entry['candidate_packet']!=inventory['packets'][index]:
            raise ValueError('ready sequence is not the declared candidate order')
    ordinal=sequence['next_ordinal']
    for pending in extra_service:
        intent=_json(pending['intent'],declaration)
        ordinal=(intent['priority']['resume_ordinal'] if intent.get('priority') is not None else pending['ordinal']+1)
    deferral_reference,deferrals,excluded_programs=_source_context_deferrals(declaration,entries,sequence)
    closed_ordinals=_closed_packet_ordinals(folder,sequence,declaration,entries)
    service=[*closed_ordinals.values(),*[{'intent':_json(item['intent'],declaration)} for item in extra_service]]
    actual_streak=0
    for item in sorted(service,key=lambda item:item['intent']['batch'],reverse=True):
        if item['intent'].get('priority')is None:break
        actual_streak+=1
    if not extra_service and (type(sequence.get('priority_streak',0))is not int or sequence.get('priority_streak',0)!=actual_streak):
        raise ValueError('priority service differs from actual closed packet history')
    priority_reference=declaration.get('priority_packets')
    priority_proof,priority_entries=(_priority_policy(priority_reference,declaration,entries)if priority_reference else(None,{}))
    policies=sequence.setdefault('priority_policies',{})
    original_bindings=[]
    for digest,reference in policies.items():
        if digest!=reference['sha256']:raise ValueError('priority policy history changed')
        historical,_=_priority_policy(reference,declaration,entries)
        original_bindings.append(historical['original_ready'])
    for item in closed_ordinals.values():
        past=item['intent'].get('priority')
        if past is not None and policies.get(past['policy']['sha256'])!=past['policy']:
            raise ValueError('closed priority packet lost its observed-prefix history')
    if priority_reference and priority_reference['sha256']not in policies:
        original=_json(priority_proof['original_ready'],declaration)['entries']
        exact=lambda value:json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'))
        if (priority_proof['original_ready']not in original_bindings
                and exact(original)!=exact(previous)):
            raise ValueError('priority original ready differs from actual observed prefix')
        policies[priority_reference['sha256']]=priority_reference
    elif priority_reference and policies[priority_reference['sha256']]!=priority_reference:
        raise ValueError('priority policy history reference changed')
    sequence['observed_entries']=entries
    return dict(declaration=declaration,manifest_path=manifest_path,manifest_raw=manifest_raw,manifest=manifest,entries=entries,ordinal=ordinal,deferral_reference=deferral_reference,deferrals=deferrals,excluded_programs=excluded_programs,closed_ordinals=closed_ordinals,priority_reference=priority_reference,priority_proof=priority_proof,priority_entries=priority_entries,service_streak=actual_streak,service_ordinals={item['ordinal']for item in extra_service})


def _choose_packet(sequence,plan,*,already_reserved=None):
    """The same source/priority decision for serial launch and bounded preview."""
    ordinal=plan['ordinal'];deferrals=plan['deferrals'];closed=plan['closed_ordinals']
    skipped=[]
    while ordinal in deferrals or ordinal in closed:
        if ordinal not in deferrals and closed[ordinal]['intent'].get('priority')is None:
            raise ValueError('ordinary cursor cannot silently skip a non-priority closed packet')
        skipped.append(ordinal);ordinal+=1
    priority_selection=None;proof=plan['priority_proof']
    if proof and ordinal>=proof['minimum_original_ordinal']:
        streak=plan['service_streak']
        if type(streak)is not int or streak<0:raise ValueError('priority service counter changed')
        if streak<proof['max_consecutive']:
            started={_json(ref,plan['declaration'])['ordinal']for ref in sequence.get('started',{}).values()}
            for candidate in plan['priority_entries']:
                if candidate in closed or candidate in plan.get('service_ordinals',set()):continue
                if candidate<ordinal or (candidate in started and candidate!=already_reserved)or candidate in deferrals:
                    raise ValueError('priority declaration cannot reclassify started, past or deferred work')
                if candidate==ordinal:break
                priority_selection=dict(policy=plan['priority_reference'],resume_ordinal=ordinal)
                ordinal=candidate;break
    return ordinal,priority_selection,skipped


def prepare_source_sequence(folder, state, config, batch):
    declaration=config.get('source_sequence')
    if declaration is None:return
    reserved=(state.get('source_sequence')or{}).get('prefetch')
    if reserved and reserved['batch']==batch:
        return adopt_prefetched_source(folder,state,config,batch)
    anchor=declaration['anchor']
    if batch<=anchor['batch']:return  # Never replace the selected 140 before batch21 seals.
    if (Path(declaration['task_runtime']).resolve()!=Path(config['task_runtime']).resolve()
            or not declaration['task_runtime_bindings']):
        raise ValueError('sequence task runtime authority differs')
    for reference in declaration['task_runtime_bindings']:_asset(reference,declaration)
    if alive(state.get('owner')):raise ValueError('cannot select a source under a live feeder')
    if (Path(config['root'])/'evidence-collections-v2'/f'batch-{batch:04d}').exists():
        raise ValueError('cannot replace a materialized teacher batch')
    anchor_root=Path(config['root'])/'evidence-collections-v2'/f"batch-{anchor['batch']:04d}"
    if (binding(anchor_root/'source-cases.jsonl')['sha256']!=anchor['cases_sha256']
            or json.loads((anchor_root/'teacher-source.json').read_text()).get('source_cases_sha256')!=anchor['cases_sha256']
            or not (anchor_root/'queue.jsonl').is_file()):
        raise ValueError('sequence anchor packet was not actually sealed')
    terminal=_json(state['last_boundary'],declaration)
    _drained_boundary(folder, state, config, batch-1, terminal)
    observe_source_sequence(folder,state,config)
    sequence=state.setdefault('source_sequence',{'next_ordinal':0})
    plan=_selection_authority(folder,sequence,declaration)
    manifest_path=plan['manifest_path']
    manifest_raw=plan['manifest_raw']
    manifest=plan['manifest']
    entries=plan['entries']
    ordinal=plan['ordinal']
    deferral_reference=plan['deferral_reference']
    deferrals=plan['deferrals']
    excluded_programs=plan['excluded_programs']
    closed_ordinals=plan['closed_ordinals']
    priority_reference=plan['priority_reference']
    priority_proof=plan['priority_proof']
    priority_entries=plan['priority_entries']
    lock=(Path(config['root'])/'next-source-selection.lock').open('a')
    with lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        pending=sequence.get('pending')
        if pending:
            if pending['batch']!=batch:raise ValueError('previous source packet has not closed')
            expected_value=_packet(entries[pending['ordinal']],declaration)
            intent=_json(pending['intent'],declaration)
            if expected_value!=_json(intent['new_configuration'],declaration):
                raise ValueError('pending packet authority changed before launch')
            _reconcile_publication(pending,declaration)
            sequence['configuration']=binding(declaration['source_configuration'])
            write(Path(folder)/'state.json',state);return
        selected,priority_selection,skipped=_choose_packet(sequence,plan)
        for skipped_ordinal in skipped:
            if skipped_ordinal in deferrals:
                _defer_context_packet(folder,state,declaration,entries[skipped_ordinal],deferrals[skipped_ordinal],deferral_reference,excluded_programs)
            else:
                sequence['next_ordinal']=skipped_ordinal+1;write(Path(folder)/'state.json',state)
        ordinal=selected
        if ordinal>=len(entries):
            if declaration.get('on_unready')!='retain-current':raise ValueError('next declared packet is not ready')
            write(Path(folder)/'state.json',state);return
        target=Path(declaration['source_configuration']);actual=binding(target)
        expected=sequence.get('configuration')or declaration['initial_configuration']
        if actual!=expected:raise ValueError('sequence source configuration authority changed')
        value=_packet(entries[ordinal],declaration)
        home=Path(folder)/'source-sequence';snapshot=home/'config-history'/(actual['sha256']+'.json')
        if snapshot.exists():
            if snapshot.read_bytes()!=target.read_bytes():raise ValueError('sequence old configuration snapshot changed')
        else:write(snapshot,target.read_bytes())
        if binding(snapshot)['sha256']!=actual['sha256']:
            raise ValueError('source configuration changed during snapshot')
        candidate=home/'config-candidates'/f'batch-{batch:04d}.json';_immutable(candidate,value)
        manifest_snapshot=home/'manifests'/(hashlib.sha256(manifest_raw).hexdigest()+'.json')
        _immutable(manifest_snapshot,manifest_raw)
        intent=home/'intents'/f'batch-{batch:04d}.json'
        _immutable(intent,{'batch':batch,'ordinal':ordinal,'old_configuration':actual,
            'old_snapshot':binding(snapshot),'new_configuration':binding(candidate),
            'manifest':binding(manifest_snapshot),'candidate_packet':entries[ordinal]['candidate_packet'],
            'qualified_packet':entries[ordinal]['qualified_packet'],
            **({'priority':priority_selection}if priority_selection is not None else{})})
        pending={'batch':batch,'ordinal':ordinal,'intent':binding(intent),'cases_sha256':value['cases_sha256']}
        sequence['pending']=pending;write(Path(folder)/'state.json',state)
        _reconcile_publication(pending,declaration)
        sequence['configuration']=binding(target);write(Path(folder)/'state.json',state)


def verify_prepared_source(state, config, batch):
    pending=(state.get('source_sequence')or{}).get('pending')
    if pending and pending['batch']==batch:
        declaration=config['source_sequence'];entries=state['source_sequence']['observed_entries']
        expected=_packet(entries[pending['ordinal']],declaration)
        intent=_json(pending['intent'],declaration)
        if expected!=_json(intent['new_configuration'],declaration):
            raise ValueError('prepared packet authority changed before launch')
        _reconcile_publication(pending,declaration)


def _prefetch_start(folder,sequence,declaration,pending,root):
    names=('source-cases.jsonl','queue.jsonl','teacher-source.json')
    if not all((root/name).is_file()for name in names):return None
    sealed={name:binding(root/name)for name in names}
    source=json.loads((root/'teacher-source.json').read_bytes())
    if (source.get('batch')!=pending['batch']
            or sealed['source-cases.jsonl']['sha256']!=pending['cases_sha256']
            or source.get('source_cases_sha256')!=pending['cases_sha256']):
        raise ValueError('reserved batch differs from its immutable source')
    intent=_json(pending['intent'],declaration)
    configuration=_json(intent['new_configuration'],declaration)
    if 'collection_sampling'in configuration:
        verify_collection_sampling(source,_jsonl_entries(root/'queue.jsonl'),configuration)
    if source.get('actual_task_runtime')!=intent['prefetch']['task_runtime']:
        raise ValueError('reserved feeder used a different task runtime')
    receipt=Path(folder)/'source-sequence/started'/f"batch-{pending['batch']:04d}.json"
    _immutable(receipt,{'batch':pending['batch'],'ordinal':pending['ordinal'],'sealed':sealed,
        'selection_intent':pending['intent'],'claim':'actual sealed batch; no generation admission/exposure'})
    sequence.setdefault('started',{})[str(pending['batch'])]=binding(receipt)
    return binding(receipt)


def _prefetch_service_chain(folder,state,config,*,require_live=True):
    """Authenticate ordered reservations without advancing the live source cursor."""
    declaration=config['source_sequence'];sequence=state['source_sequence']
    pending=sequence.get('pending')
    if not pending or pending['batch']!=state.get('active_batch')or (require_live and not alive(state.get('owner'))):
        raise ValueError('prefetch lacks its exact live predecessor selection')
    chain=[pending]
    completed=state.get('primary_completed_ahead',{});active=state.get('primary_active_ahead',{})
    if active and config.get('primary_worker_capacity')is None:raise ValueError('active tails lack their worker envelope')
    for batch,slot in primary_ahead_slots(state).items():
        key=str(batch)
        if key not in completed and key not in active:continue
        if batch!=chain[-1]['batch']+1:raise ValueError('prefetch chain has a gap')
        if key in completed and alive(slot.get('owner')):raise ValueError('completed prefetch still has a live owner')
        if key in active and require_live and not alive(slot.get('owner')):raise ValueError('active prefetch lost its actual owner')
        reserved=sequence.get('prefetch_history',{}).get(key)
        if not reserved or reserved['batch']!=batch or reserved['intent']!=slot['selection_intent']:
            raise ValueError('completed prefetch lost its reservation')
        if key in completed:
            reference=slot['boundary'];terminal=_json(reference,declaration)
            if (Path(reference['path']).resolve()!=(Path(folder)/'boundaries'/f'batch-{batch:04d}.json').resolve()
                    or terminal.get('batch')!=batch):
                raise ValueError('completed prefetch boundary differs')
            started=_json(sequence['started'][key],declaration)
            if any(terminal['bindings'].get(name)!=started['sealed'][name]for name in ('queue.jsonl','teacher-source.json')):
                raise ValueError('completed prefetch boundary changed its sealed source')
            _drained_boundary(folder,{**state,'last_boundary':reference},config,batch,terminal)
        chain.append(reserved)
    for item in chain:
        batch=item['batch'];reference=sequence.get('started',{}).get(str(batch))
        if not reference:raise ValueError('prefetch predecessor source was not actually sealed')
        original=_json(reference,declaration)
        root=Path(config['root'])/'evidence-collections-v2'/f'batch-{batch:04d}'
        if (original.get('batch')!=batch or original.get('ordinal')!=item['ordinal']
                or original.get('selection_intent')!=item['intent']
                or original['sealed']!={name:binding(root/name)for name in ('source-cases.jsonl','queue.jsonl','teacher-source.json')}):
            raise ValueError('prefetch predecessor selection identity or sealed source changed')
    return chain


def park_active_prefetch(folder,state,config,slot):
    """Retain an actual live tail without mislabelling it as completed."""
    if config.get('primary_worker_capacity')is None or not alive(slot.get('owner')):
        raise ValueError('active tail needs its declared envelope and actual owner')
    sequence=state['source_sequence'];reserved=sequence.get('prefetch');key=str(slot['batch'])
    if (not reserved or reserved['batch']!=slot['batch']or reserved['intent']!=slot['selection_intent']
            or state.get('primary_prefetch')!=slot):raise ValueError('active tail lost its exact reservation')
    history=sequence.setdefault('prefetch_history',{})
    if key in history and history[key]!=reserved:raise ValueError('active reservation history changed')
    history[key]=reserved;active=state.setdefault('primary_active_ahead',{})
    if key in active and active[key]!=slot:raise ValueError('active tail custody changed')
    active[key]=slot;sequence.pop('prefetch');state.pop('primary_prefetch')
    _prefetch_service_chain(folder,state,config,require_live=False)
    write(Path(folder)/'state.json',state)


def archive_completed_prefetch(folder,state,config,slot):
    """Retain actual completed work while releasing only its worker capacity."""
    sequence=state['source_sequence'];key=str(slot['batch']);active=state.get('primary_active_ahead',{})
    reserved=sequence.get('prefetch')
    if not reserved or reserved['batch']!=slot['batch']:reserved=sequence.get('prefetch_history',{}).get(key)
    if not reserved or reserved['batch']!=slot['batch']or reserved['intent']!=slot['selection_intent']:
        raise ValueError('completed prefetch lost its exact active reservation')
    history=sequence.setdefault('prefetch_history',{})
    if key in history and history[key]!=reserved:raise ValueError('completed reservation history changed')
    history[key]=reserved;completed=state.setdefault('primary_completed_ahead',{})
    if key in completed and completed[key]!=slot:raise ValueError('completed slot history changed')
    if key in active:active.pop(key)
    elif (state.get('primary_prefetch')or{}).get('batch')==slot['batch']:
        sequence.pop('prefetch');state.pop('primary_prefetch')
    else:raise ValueError('completed batch was not an owned live slot')
    completed[key]=slot
    _prefetch_service_chain(folder,state,config,require_live=False)
    write(Path(folder)/'state.json',state)


def prepare_prefetched_source(folder,state,config,batch):
    """Reserve the next ordered source using released live worker capacity."""
    if config.get('primary_parallel_batches',1)!=2:
        raise ValueError('source prefetch requires explicit two-batch ownership')
    declaration=config['source_sequence'];sequence=state.get('source_sequence')or{}
    chain=_prefetch_service_chain(folder,state,config)
    pending=chain[-1]
    if batch!=pending['batch']+1:raise ValueError('prefetch lacks its exact live predecessor selection')
    predecessor=sequence['started'][str(pending['batch'])]
    if Path(declaration['task_runtime']).resolve()!=Path(config['task_runtime']).resolve()or not declaration['task_runtime_bindings']:
        raise ValueError('prefetch runtime authority differs')
    for reference in declaration['task_runtime_bindings']:_asset(reference,declaration)
    lock=(Path(config['root'])/'next-source-selection.lock').open('a')
    with lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        previous=sequence.get('prefetch')
        if previous:
            if previous['batch']!=batch:raise ValueError('another next source is already reserved')
            _json(previous['intent'],declaration)
            return previous
        if (Path(config['root'])/'evidence-collections-v2'/f'batch-{batch:04d}').exists():
            raise ValueError('cannot reserve a materialized batch without its intent')
        plan=_selection_authority(folder,sequence,declaration,extra_service=chain)
        ordinal,priority,skipped=_choose_packet(sequence,plan)
        if ordinal>=len(plan['entries']):return None
        value=_packet(plan['entries'][ordinal],declaration)
        target=Path(declaration['source_configuration']);actual=binding(target)
        if actual!=(sequence.get('configuration')or declaration['initial_configuration']):
            raise ValueError('prefetch ambient source authority changed')
        old_raw=target.read_bytes()
        if len(chain)>1:
            # Cursor publication remains on the old live source; this future
            # intent must nevertheless chain from its actual reserved predecessor.
            old_raw=_read(_json(pending['intent'],declaration)['new_configuration'],declaration)[1]
            actual={'path':str(target),'sha256':hashlib.sha256(old_raw).hexdigest(),'bytes':len(old_raw)}
        home=Path(folder)/'source-sequence';snapshot=home/'config-history'/(actual['sha256']+'.json')
        _immutable(snapshot,old_raw)
        candidate=home/'config-candidates'/f'batch-{batch:04d}.json';_immutable(candidate,value)
        manifest=home/'manifests'/(hashlib.sha256(plan['manifest_raw']).hexdigest()+'.json');_immutable(manifest,plan['manifest_raw'])
        intent=home/'intents'/f'batch-{batch:04d}.json'
        _immutable(intent,{'batch':batch,'ordinal':ordinal,'old_configuration':actual,'old_snapshot':binding(snapshot),
            'new_configuration':binding(candidate),'manifest':binding(manifest),
            'candidate_packet':plan['entries'][ordinal]['candidate_packet'],
            'qualified_packet':plan['entries'][ordinal]['qualified_packet'],
            **({'priority':priority}if priority is not None else{}),
            'prefetch':{'predecessor_started':predecessor,'predecessor_intent':pending['intent'],
                'task_runtime':config['task_runtime'],'skipped_ordinals':skipped,
                **({'resource_refill':{'guardian_config':state['config_snapshot'],
                    'feeder':config['overlap_feeder'],'workers':config['overlap_workers'],
                    'collection_context':{'repo':config['collection_repo'],
                        'queue_worker':config['collection_worker'],'node_bin':config['node_bin']}}}
                   if config.get('primary_worker_capacity')is not None else{}),
                'selection_policy':{'priority_packets':plan['priority_reference'],
                    'context_deferrals':plan['deferral_reference']},
                'claim':'conditional next selection; predecessor remains unfinished'}})
        reserved={'batch':batch,'ordinal':ordinal,'intent':binding(intent),'cases_sha256':value['cases_sha256']}
        sequence['prefetch']=reserved;write(Path(folder)/'state.json',state)
        return reserved


def observe_prefetched_source(folder,state,config):
    sequence=state.get('source_sequence')or{};pending=sequence.get('prefetch');result=None
    values=[sequence.get('prefetch_history',{}).get(key)for key in state.get('primary_active_ahead',{})]
    if pending is not None:values.append(pending)
    for reserved in values:
        if reserved is None:raise ValueError('active source slot lost its immutable reservation')
        root=Path(config['root'])/'evidence-collections-v2'/f"batch-{reserved['batch']:04d}"
        current=_prefetch_start(folder,sequence,config['source_sequence'],reserved,root)
        if current:write(Path(folder)/'state.json',state)
        if reserved is pending:result=current
    return result


def adopt_prefetched_source(folder,state,config,batch):
    """Confirm the reservation against real closed history before cursor adoption."""
    declaration=config['source_sequence'];sequence=state['source_sequence'];reserved=sequence.get('prefetch')
    if not reserved or reserved.get('batch')!=batch:reserved=sequence.get('prefetch_history',{}).get(str(batch))
    if reserved is None:raise ValueError('prefetch reservation custody is missing')
    if reserved['batch']!=batch or (sequence.get('pending')is not None and sequence['pending']!=reserved):
        raise ValueError('prefetched selection cannot bypass its unfinished predecessor')
    intent=_json(reserved['intent'],declaration);predecessor=intent['prefetch']['predecessor_started']
    closed=sequence.get('closed',{}).get(str(batch-1))
    if not closed or _json(closed,declaration).get('started')!=predecessor:
        raise ValueError('prefetch predecessor has not genuinely closed')
    lock=(Path(config['root'])/'next-source-selection.lock').open('a')
    with lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        # Current authority still authenticates the full retained prefix, source
        # protection and debt. The already-started reservation keeps its exact
        # scheduling policy when later unstarted priorities are appended.
        plan=_selection_authority(folder,sequence,declaration)
        basis=intent['prefetch']['selection_policy']
        if not isinstance(basis,dict)or set(basis)!={'priority_packets','context_deferrals'}:
            raise ValueError('prefetch lost its immutable selection policy')
        frozen={**declaration,**basis}
        ref=basis['priority_packets']
        proof,entries=(_priority_policy(ref,frozen,plan['entries'])if ref else(None,{}))
        dref,deferrals,programs=_source_context_deferrals(frozen,plan['entries'],sequence)
        plan={**plan,'declaration':frozen,'priority_reference':ref,'priority_proof':proof,
            'priority_entries':entries,'deferral_reference':dref,'deferrals':deferrals,
            'excluded_programs':programs}
        ordinal,priority,skipped=_choose_packet(sequence,plan,already_reserved=reserved['ordinal'])
        if (ordinal!=reserved['ordinal']or priority!=intent.get('priority')
                or skipped!=intent['prefetch']['skipped_ordinals']
                or _packet(plan['entries'][ordinal],declaration)!=_json(intent['new_configuration'],declaration)):
            raise ValueError('reserved next source differs from actual serial source policy')
        for skipped_ordinal in skipped:
            if skipped_ordinal in plan['deferrals']:
                _defer_context_packet(folder,state,declaration,plan['entries'][skipped_ordinal],plan['deferrals'][skipped_ordinal],plan['deferral_reference'],plan['excluded_programs'])
            else:sequence['next_ordinal']=skipped_ordinal+1;write(Path(folder)/'state.json',state)
        _prefetch_start(folder,sequence,declaration,reserved,
            Path(config['root'])/'evidence-collections-v2'/f'batch-{batch:04d}')
        sequence['pending']=reserved;write(Path(folder)/'state.json',state)
        _reconcile_publication(reserved,declaration)
        sequence['configuration']=binding(declaration['source_configuration'])
        sequence.setdefault('prefetch_history',{})[str(batch)]=reserved
        if (sequence.get('prefetch')or{}).get('batch')==batch:sequence.pop('prefetch')
        write(Path(folder)/'state.json',state)
        return reserved
