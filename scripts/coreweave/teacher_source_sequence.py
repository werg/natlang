"""Exact lifecycle/publication plumbing for a supervisor-declared task sequence."""
from __future__ import annotations
import fcntl
import hashlib
import json
from pathlib import Path
import time
try:
    from .teacher_generation_guardian import binding, write, alive
except ImportError:
    from teacher_generation_guardian import binding, write, alive


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


def _packet(entry, declaration, *, allow_empty=False):
    candidate=_json(entry['candidate_packet'],declaration)
    prepared=_json(entry['qualified_packet'],declaration)
    qualification=_json(prepared['qualification'],declaration)
    inputs=_json(qualification['inputs'],declaration)
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
    if (qualification.get('schema')!='natlang.existing-native-reference-qualification/1'
            or qualification.get('passed')is not True or qualification.get('model_calls')!=0
            or qualification.get('training_admission')is not False
            or qualification.get('reference_traces_are_teacher_generations')is not False
            or qualification.get('runtime_inventory_sha256')!=expected_runtime
            or runtime_digest!=expected_runtime
            or inputs['authority']['protection']!=authority['protection']
            or inputs['authority']['decision_holds']!=authority['decision_hold_bindings']
            or selection.get('protection')!=authority['protection']
            or selection.get('decision_hold_bindings')!=authority['decision_hold_bindings']):
        raise ValueError('sequence reference/protection authority differs')
    if (selection.get('schema')!='natlang.strong-teacher-code-source-selection/1'
            or selection.get('training_admission')is not False
            or selection.get('held_and_pending_intersections')!=0
            or selection.get('cases_sha256')!=refs['selected_cases']['sha256']
            or selection.get('reference_qualification')!=prepared['qualification']
            or protection.get('schema')!='natlang.native-teacher-global-protection-join/1'
            or protection.get('status')!='passed'
            or protection.get('selected_packet_sha256')!=refs['selected_cases']['sha256']
            or protection.get('current_protection')!=authority['protection']
            or protection.get('decision_hold_bindings')!=authority['decision_hold_bindings']
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
    return {'cases':str(cases),'cases_sha256':refs['selected_cases']['sha256'],'rows':len(programs),
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
    key=str(batch);started=sequence.setdefault('started',{})
    if key not in started:
        receipt=Path(folder)/'source-sequence/started'/f'batch-{batch:04d}.json'
        _immutable(receipt,{'batch':batch,'ordinal':pending['ordinal'],'sealed':sealed,
            'selection_intent':pending['intent'],'claim':'actual sealed batch; no generation admission/exposure'})
        started[key]=binding(receipt);write(Path(folder)/'state.json',state)
    elif _json(started[key],declaration)['sealed']!=sealed:
        raise ValueError('sequence started batch artifacts changed')
    intent=_json(pending['intent'],declaration)
    current=binding(declaration['source_configuration'])
    if current['sha256']!=intent['new_configuration']['sha256']:
        raise ValueError('started source configuration differs from publication intent')
    sequence['configuration']=current
    boundary=state.get('last_boundary')
    if boundary is None:return
    terminal=_json(boundary,declaration)
    if terminal.get('batch')!=batch:return
    if terminal.get('batch_complete')is not True:raise ValueError('sequence batch is not genuinely closed')
    if any(terminal['bindings'].get(name)!=sealed[name]for name in ('queue.jsonl','teacher-source.json')):
        raise ValueError('sequence terminal boundary differs from its sealed batch')
    receipt=Path(folder)/'source-sequence/closed'/f'batch-{batch:04d}.json'
    _immutable(receipt,{'batch':batch,'ordinal':pending['ordinal'],'started':started[key],
        'boundary':boundary,'claim':'existing guardian completion; individual outcomes/admission unchanged'})
    sequence.setdefault('closed',{})[key]=binding(receipt)
    sequence['next_ordinal']=pending['ordinal']+1;sequence.pop('pending')
    write(Path(folder)/'state.json',state)


def prepare_source_sequence(folder, state, config, batch):
    declaration=config.get('source_sequence')
    if declaration is None:return
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
    if terminal.get('batch')!=batch-1 or terminal.get('batch_complete')is not True:
        raise ValueError('source selection needs the actual preceding completed boundary')
    observe_source_sequence(folder,state,config)
    sequence=state.setdefault('source_sequence',{'next_ordinal':0})
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
    sequence['observed_entries']=entries
    ordinal=sequence['next_ordinal']
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
            'qualified_packet':entries[ordinal]['qualified_packet']})
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
