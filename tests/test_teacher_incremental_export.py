import hashlib
import json
from pathlib import Path

import pytest

from scripts.coreweave import source_archive
from scripts.coreweave.teacher_incremental_export import (
    accept_private_receipt, aggregate_worker_journals, completed_entries, export_finished_tranches,
    prune_verified_local_tranche, seal_boundary, seal_tranche,
)


@pytest.fixture(autouse=True)
def small_archive_spool_reserve(monkeypatch):
    """Unit fixtures are tiny; leave the production 20 GiB reserve untouched."""
    original = source_archive.pack
    def pack(source, spool, name, **kwargs):
        kwargs['reserve_bytes'] = 0
        return original(source, spool, name, **kwargs)
    monkeypatch.setattr(source_archive, 'pack', pack)


def _fixture():
    entries = [
        {'key': 's7-task-000-00', 'index': 0, 'count': 1, 'sample': 0, 'seed': 3,
         'source': '/batch/source-cases.jsonl', 'output': '/batch/out/00/results.jsonl',
         'jobs': '/batch/out/00/jobs', 'task_id': 'task-a'},
        {'key': 's7-task-001-00', 'index': 1, 'count': 1, 'sample': 0, 'seed': 4,
         'source': '/batch/source-cases.jsonl', 'output': '/batch/out/01/results.jsonl',
         'jobs': '/batch/out/01/jobs', 'task_id': 'task-b'},
    ]
    queue = b''.join(json.dumps(row).encode() + b'\n' for row in entries)
    finish = lambda key: {'event': 'finish', 'key': key, 'output_accounting': {
        'version': 'natlang.supervisor_output_accounting/2', 'manifest_valid': True,
        'manifest_errors': [], 'complete': True, 'expected_jobs': 1,
        'exact_result_rows': 1, 'output_sha256': 'a'*64,
        'job_states': [{'state': 'result'}]}}
    journal = b''.join(json.dumps({'event': 'start', 'key': row['key']}).encode() + b'\n'
                       + json.dumps(finish(row['key'])).encode() + b'\n' for row in entries)
    cases = b'{"id":"task-a"}\n{"id":"task-b"}\n'
    teacher_source = {'schema': 'natlang.sandbox-strong-teacher-collection/1', 'batch': 19,
        'source_cases_sha256': hashlib.sha256(cases).hexdigest(), 'actor': {'sha256': 'a'*64},
        'actual_teacher_execution_identity': {'sha256': 'b'*64}, 'actual_task_runtime': {'sha256': 'c'*64},
        'collection_controller': {'sha256': 'd'*64}, 'runtime_source_receipt': {'sha256': 'e'*64},
        'teacher_configuration_source': {'sha256': 'f'*64}, 'source_provenance': {'sha256': '0'*64}}
    context = {'teacher-source.json': json.dumps(teacher_source).encode(), 'queue.jsonl': queue,
               'source-cases.jsonl': cases,
               'repair-queue.jsonl': b''}
    outputs = {}
    for entry in entries:
        payload = json.dumps({'trajectory': [{'task': entry['task_id']}]}).encode() + b'\n'
        manifest = json.dumps({'version': 'natlang.teacher_batch.native/1',
            'source': entry['source'], 'source_sha256': hashlib.sha256(context['source-cases.jsonl']).hexdigest(),
            'range': {'start': entry['index'], 'count': 1}, 'completed': 1, 'missing': [],
            'output_sha256': hashlib.sha256(payload).hexdigest()}).encode()
        event = finish(entry['key'])
        event['output_accounting'].update(exact_result_rows=1,
                                          output_sha256=hashlib.sha256(payload).hexdigest())
        # Keep the output accounting digest in the queue-selected journal event.
        journal_events = [json.loads(line) for line in journal.splitlines()]
        journal_events = [event if row.get('key') == entry['key'] and row.get('event') == 'finish' else row
                          for row in journal_events]
        journal = b''.join(json.dumps(row).encode()+b'\n' for row in journal_events)
        outputs[entry['key']] = payload, manifest
    return entries, queue, journal, context, outputs, finish


def test_completed_tranche_frontier_is_ordered_terminal_and_resume_safe():
    entries, queue, journal, context, _, finish = _fixture()
    event = finish(entries[0]['key']); event['output_accounting']['output_sha256'] = 'a'*64
    partial = json.dumps(event).encode() + b'\n'
    assert [row['key'] for row in completed_entries(queue, partial)] == [entries[0]['key']]
    assert completed_entries(queue, partial, after={entries[0]['key']}) == []
    unresolved = json.loads(partial)
    unresolved['output_accounting']['complete'] = False
    assert completed_entries(queue, json.dumps(unresolved).encode() + b'\n') == []


def test_jsonl_parser_preserves_unicode_line_separators_inside_json_strings():
    entries, _, journal, _, _, _ = _fixture()
    first = dict(entries[0], note='line\u2028separator\u2029is string content')
    raw = json.dumps(first, ensure_ascii=False).encode('utf-8') + b'\n'
    finish = json.loads(journal.split(b'\n')[1])
    result = completed_entries(raw, json.dumps(finish, ensure_ascii=False).encode() + b'\n')
    assert result[0]['note'] == first['note']


def test_teacher_source_accepts_actual_runtime_path_string():
    from scripts.coreweave.teacher_incremental_export import _teacher_source_binding
    source = json.loads(_fixture()[3]['teacher-source.json'])
    source['actual_task_runtime'] = '/work/datajobs/runtime/ts-host'
    cases = _fixture()[3]['source-cases.jsonl']
    assert _teacher_source_binding(json.dumps(source).encode(), batch=19,
        source_sha256=hashlib.sha256(cases).hexdigest())['actual_task_runtime'] == source['actual_task_runtime']


def test_owner_poller_reloads_active_config_and_checks_teacher_identity(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import scripts.coreweave.teacher_incremental_export as module
    context = _fixture()[3]
    source = json.loads(context['teacher-source.json'])
    source['actual_task_runtime'] = '/work/runtime/ts-host'
    teacher_raw = json.dumps(source).encode()
    cases_raw = context['source-cases.jsonl']
    remote = {'/batch/teacher-source.json': teacher_raw, '/batch/source-cases.jsonl': cases_raw,
              '/batch/queue.jsonl': b'{"key":"k"}\n',
              '/batch/journal.jsonl': b'{"event":"start","key":"k"}\n'}
    sandbox = SimpleNamespace(exec=lambda *args, **kwargs: SimpleNamespace(result=lambda **_: SimpleNamespace(
        returncode=0, stdout='boot-actual-1\n', stderr='')))
    monkeypatch.setattr(module, '_cwctl_module', lambda: SimpleNamespace(load_env_file=lambda: None))
    monkeypatch.setattr(module, 'scoped_sdk_session', lambda *_args, **_kwargs:
                        (sandbox, remote.__getitem__))
    seen = []
    config = {'sandbox_id': 'sandbox-1', 'batch': 19, 'queue_path': '/batch/queue.jsonl',
        'journal_path': '/batch/journal.jsonl', 'context_paths': {
            'teacher-source.json': '/batch/teacher-source.json',
            'source-cases.jsonl': '/batch/source-cases.jsonl'},
        'directory': str(tmp_path/'exports'), 'role': 'bulk', 'chunk_size': 32,
        'exporter_path': '/owner/export.py', 'flush_partial': False, 'idle_flush_after_seconds': 0,
        'identity': {'sandbox_boot_id': 'boot-actual-1',
            'teacher_source_sha256': hashlib.sha256(teacher_raw).hexdigest(),
            'source_cases_sha256': hashlib.sha256(cases_raw).hexdigest(),
            'actor_sha256': 'a'*64, 'execution_identity_sha256': 'b'*64,
            'teacher_config_sha256': 'f'*64, 'runtime_path': '/work/runtime/ts-host'}}
    config_path = tmp_path/'active.json'
    config_path.write_text(json.dumps(config))
    def poll(**kwargs):
        seen.append((kwargs['batch'], kwargs['flush_partial'], kwargs['journal_snapshot']))
        if len(seen) == 1:
            config['flush_partial'] = True
            config_path.write_text(json.dumps(config))
        return {'state': {'completed_keys': []}, 'training_admission': False, 'exposure': False}
    monkeypatch.setattr(module, 'export_finished_tranches', poll)
    result = module.run_incremental_service(config_path, poll_seconds=0, max_polls=2,
                                             allowed_spool_root=tmp_path)
    assert seen == [(19, True, remote['/batch/journal.jsonl'])] * 2
    assert result['config_changed'] is True
    assert result['identity']['sandbox_boot_id'] == 'boot-actual-1'
    assert json.loads((tmp_path/'exports/service-state.json').read_text())['training_admission'] is False


def test_owner_poller_fails_closed_on_restarted_sandbox_boot(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import scripts.coreweave.teacher_incremental_export as module
    context = _fixture()[3]
    source = json.loads(context['teacher-source.json'])
    source['actual_task_runtime'] = '/work/runtime/ts-host'
    teacher_raw = json.dumps(source).encode()
    cases_raw = context['source-cases.jsonl']
    remote = {'/batch/teacher-source.json': teacher_raw, '/batch/source-cases.jsonl': cases_raw,
              '/batch/queue.jsonl': b'{"key":"k"}\n',
              '/batch/journal.jsonl': b'{"event":"start","key":"k"}\n'}
    sandbox = SimpleNamespace(exec=lambda *args, **kwargs: SimpleNamespace(result=lambda **_: SimpleNamespace(
        returncode=0, stdout='different-boot\n', stderr='')))
    monkeypatch.setattr(module, '_cwctl_module', lambda: SimpleNamespace(load_env_file=lambda: None))
    monkeypatch.setattr(module, 'scoped_sdk_session', lambda *_args, **_kwargs:
                        (sandbox, remote.__getitem__))
    config = {'sandbox_id': 'sandbox-1', 'batch': 19, 'queue_path': '/batch/queue.jsonl',
        'journal_path': '/batch/journal.jsonl', 'context_paths': {
            'teacher-source.json': '/batch/teacher-source.json',
            'source-cases.jsonl': '/batch/source-cases.jsonl'},
        'directory': str(tmp_path/'exports'), 'role': 'bulk', 'chunk_size': 32,
        'exporter_path': '/owner/export.py', 'identity': {
            'sandbox_boot_id': 'expected-boot',
            'teacher_source_sha256': hashlib.sha256(teacher_raw).hexdigest(),
            'source_cases_sha256': hashlib.sha256(cases_raw).hexdigest(),
            'actor_sha256': 'a'*64, 'execution_identity_sha256': 'b'*64,
            'teacher_config_sha256': 'f'*64, 'runtime_path': '/work/runtime/ts-host'}}
    config_path = tmp_path/'active.json'; config_path.write_text(json.dumps(config))
    monkeypatch.setattr(module, 'export_finished_tranches', lambda **_: pytest.fail('export must not run'))
    with pytest.raises(ValueError, match='boot identity differs'):
        module.run_incremental_service(config_path, poll_seconds=0, max_polls=1,
                                       allowed_spool_root=tmp_path)
    state = json.loads((tmp_path/'exports/service-state.json').read_text())
    assert 'boot identity differs' in state['identity_error']


def test_guardian_discovery_accepts_exact_anchor_without_started_receipt():
    from types import SimpleNamespace
    from scripts.coreweave.teacher_incremental_export import discover_guardian_batch
    import hashlib
    cases_path = '/work/generation/evidence-collections-v2/batch-0021/source-cases.jsonl'
    queue_path = '/work/generation/evidence-collections-v2/batch-0021/queue.jsonl'
    teacher_path = '/work/generation/evidence-collections-v2/batch-0021/teacher-source.json'
    cases = b''.join((json.dumps({'program_ir': {'id': f'task-{index}'}})+'\n').encode()
                     for index in range(24))
    queue_rows = [{'key':f's7-task-{index:03d}-00','index':index,'count':1,'sample':0,'seed':index,
        'source':cases_path,'output':f'/work/generation/evidence-collections-v2/batch-0021/out/{index}/results.jsonl',
        'jobs':f'/work/generation/evidence-collections-v2/batch-0021/out/{index}/jobs',
        'task_id':f'task-{index}'} for index in range(24)]
    queue = b''.join((json.dumps(row)+'\n').encode() for row in queue_rows)
    teacher = {'schema':'natlang.sandbox-strong-teacher-collection/1','batch':21,
        'source_cases_sha256':hashlib.sha256(cases).hexdigest(),
        'teacher':{'model':'qwen-test'},
        'actor':{'sha256':'a'*64},'actual_teacher_execution_identity':{'sha256':'b'*64},
        'actual_task_runtime':'/work/runtime','collection_controller':{'sha256':'c'*64},
        'runtime_source_receipt':{'sha256':'d'*64},'teacher_configuration_source':{'sha256':'e'*64},
        'source_provenance':{'sha256':'f'*64}}
    teacher_raw=json.dumps(teacher).encode()
    guardian={'control':'/work/control','root':'/work/generation','feeder':'/work/control/feed.py',
        'task_runtime':'/work/runtime','teacher_identity':{'sha256':'b'*64},
        'source_sequence':{'anchor':{'batch':21,'cases_sha256':hashlib.sha256(cases).hexdigest()}}}
    guardian_raw=json.dumps(guardian).encode()
    guardian_ref={'path':'/work/control/config.json','bytes':len(guardian_raw),
        'sha256':hashlib.sha256(guardian_raw).hexdigest()}
    state={'schema':'natlang.teacher-generation-guardian/1','config':guardian_ref,
        'owner':{'pid':123,'start_ticks':'456','boot_id':'boot-1'},'active_batch':21}
    remote={guardian_ref['path']:guardian_raw,'/work/control/state.json':json.dumps(state).encode(),
        cases_path:cases,queue_path:queue,teacher_path:teacher_raw,
        '/work/generation/feed-evidence-phase.json':json.dumps({'batch':21,'phase':'collecting','worker_count':24,
            'worker_pids':list(range(100,124)),'out':'/work/generation/evidence-collections-v2/batch-0021',
            'cases':24,'teacher':'qwen-test'}).encode()}
    worker_names=[]
    for index,row in enumerate(queue_rows):
        qname=f'queue-worker-{index}.jsonl';jname=f'journal-worker-{index}.jsonl'
        remote[f'/work/generation/evidence-collections-v2/batch-0021/{qname}']=(json.dumps(row)+'\n').encode()
        remote[f'/work/generation/evidence-collections-v2/batch-0021/{jname}']=b''
        worker_names.append((qname,jname))
    def exec_(argv,**kwargs):
        if argv[0]=='cat': out='boot-1\n'
        elif argv[1]=='-c' and 'glob' in argv[2]: out=json.dumps({'queues':[x[0] for x in worker_names],
            'journals':[x[1] for x in worker_names]})
        else: out=json.dumps({'valid':True,'identity':{'pid':123,'start_ticks':'456','boot_id':'boot-1',
            'argv':['python','/work/control/feed.py']}})
        return SimpleNamespace(result=lambda **_:SimpleNamespace(returncode=0,stdout=out,stderr=''))
    sandbox=SimpleNamespace(exec=exec_)
    publisher={'schema':'natlang.teacher-incremental-export-publisher/1','sandbox_id':'sid',
        'sandbox_boot_id':'boot-1','guardian_config':guardian_ref,'control_dir':'/work/control',
        'generation_root':'/work/generation','anchor_batch':21,
        'anchor_cases_sha256':hashlib.sha256(cases).hexdigest(),
        'directory_root':'/external/export','role':'bulk','chunk_size':32,'exporter_path':'/owner/export.py',
        'identity':{'sandbox_boot_id':'boot-1','actor_sha256':'a'*64,
            'execution_identity_sha256':'b'*64,'teacher_config_sha256':'e'*64,'runtime_path':'/work/runtime'}}
    active=discover_guardian_batch(publisher,sandbox,remote.__getitem__)
    assert active['batch']==21 and active['queue_path']==queue_path
    assert active['source_sequence_started_sha256'] is None
    assert active['identity']['source_cases_sha256']==hashlib.sha256(cases).hexdigest()
    assert active['worker_queue_paths']==[f'/work/generation/evidence-collections-v2/batch-0021/queue-worker-{i}.jsonl'
                                          for i in range(24)]


def test_guardian_discovery_requires_exact_started_receipt_after_anchor():
    from types import SimpleNamespace
    from scripts.coreweave.teacher_incremental_export import discover_guardian_batch
    guardian={'control':'/work/control','root':'/work/generation','feeder':'/work/control/feed.py',
        'task_runtime':'/work/runtime','teacher_identity':{'sha256':'b'*64},
        'source_sequence':{'anchor':{'batch':21,'cases_sha256':'a'*64}}}
    guardian_raw=json.dumps(guardian).encode()
    ref={'path':'/work/control/config.json','bytes':len(guardian_raw),
        'sha256':hashlib.sha256(guardian_raw).hexdigest()}
    state={'schema':'natlang.teacher-generation-guardian/1','config':ref,
        'owner':{'pid':123,'start_ticks':'456','boot_id':'boot-1'},'active_batch':22,
        'source_sequence':{'started':{}}}
    remote={ref['path']:guardian_raw,'/work/control/state.json':json.dumps(state).encode()}
    def exec_(argv,**kwargs):
        out='boot-1\n' if argv[0]=='cat' else json.dumps({'valid':True})
        return SimpleNamespace(result=lambda **_:SimpleNamespace(returncode=0,stdout=out,stderr=''))
    publisher={'schema':'natlang.teacher-incremental-export-publisher/1','sandbox_id':'sid',
        'sandbox_boot_id':'boot-1','guardian_config':ref,'control_dir':'/work/control',
        'generation_root':'/work/generation','anchor_batch':21,'anchor_cases_sha256':'a'*64,
        'directory_root':'/external/export','role':'bulk','chunk_size':32,'exporter_path':'/owner/export.py',
        'identity':{'sandbox_boot_id':'boot-1','actor_sha256':'a'*64,'execution_identity_sha256':'b'*64,
            'teacher_config_sha256':'e'*64,'runtime_path':'/work/runtime'}}
    with pytest.raises(ValueError,match='started receipt'):
        discover_guardian_batch(publisher,SimpleNamespace(exec=exec_),remote.__getitem__)


def test_worker_journal_aggregation_checks_exact_24_way_partition_and_trailing_append():
    rows=[]; worker_queues=[]; worker_journals=[]
    for index in range(24):
        row={'key':f's7-task-{index:03d}-00','index':index,'count':1,'sample':0,'seed':index,
            'task_id':f'task-{index}','group':index,'source':'/cases.jsonl',
            'output':f'/out/{index}/results.jsonl','jobs':f'/out/{index}/jobs'}
        rows.append(row)
        q=json.dumps(row).encode()+b'\n';worker_queues.append((f'queue-worker-{index}.jsonl',q))
        events=[{'event':'start','key':row['key']},{'event':'finish','key':row['key']}]
        journal=b''.join(json.dumps(event).encode()+b'\n' for event in events)
        if index==23: journal+=b'{"event":"finish"'  # in-progress append is withheld
        worker_journals.append((f'journal-worker-{index}.jsonl',journal))
    master=b''.join(json.dumps(row).encode()+b'\n' for row in rows)
    aggregate=aggregate_worker_journals(master,worker_queues,worker_journals)
    parsed=[json.loads(line) for line in aggregate.splitlines()]
    assert len(parsed)==48 and parsed[-1]['event']=='finish' and parsed[-1]['key']==rows[-1]['key']
    duplicate=list(worker_queues);duplicate[-1]=(duplicate[-1][0],duplicate[0][1])
    with pytest.raises(ValueError,match='foreign or duplicated'):
        aggregate_worker_journals(master,duplicate,worker_journals)
    reordered=list(worker_queues)
    reordered[1],reordered[10]=reordered[10],reordered[1]
    with pytest.raises(ValueError,match='numeric contiguous order'):
        aggregate_worker_journals(master,reordered,worker_journals)
    changed=json.loads(worker_queues[0][1]);changed['temperature']=0.2
    altered=list(worker_queues);altered[0]=(altered[0][0],(json.dumps(changed)+'\n').encode())
    with pytest.raises(ValueError,match='identity differs'):
        aggregate_worker_journals(master,altered,worker_journals)


def test_guardian_publisher_resumes_tuple_frontier_and_flushes_idle_partial(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import time
    import scripts.coreweave.teacher_incremental_export as module
    publisher={'schema':'natlang.teacher-incremental-export-publisher/1',
        'sandbox_id':'sid','directory_root':str(tmp_path/'spool'),'role':'bulk','exporter_path':'/owner/export.py',
        'idle_flush_after_seconds':1,'chunk_size':32}
    publisher_path=tmp_path/'publisher.json';publisher_path.write_text(json.dumps(publisher))
    active={'batch':21,'directory':str(tmp_path/'spool'/'batch-0021'),'queue_path':'/batch/queue.jsonl',
        'journal_path':'/batch/journal.jsonl','context_paths':{'teacher-source.json':'/batch/source.json',
        'source-cases.jsonl':'/batch/cases.jsonl'},'chunk_size':32,'role':'bulk','exporter_path':'/owner/export.py',
        'identity':{'sandbox_boot_id':'boot','teacher_source_sha256':'t'*64},
        'queue_sha256':hashlib.sha256(b'queue').hexdigest(),'guardian_state_sha256':'g'*64,'source_sequence_started_sha256':None,
        'worker_queue_paths':[],'worker_journal_paths':{}}
    sandbox=object();remote={'/batch/queue.jsonl':b'queue','/batch/journal.jsonl':b'journal'}
    monkeypatch.setattr(module,'_cwctl_module',lambda:SimpleNamespace(load_env_file=lambda:None))
    monkeypatch.setattr(module,'scoped_sdk_session',lambda *_a,**_k:(sandbox,remote.__getitem__))
    monkeypatch.setattr(module,'discover_guardian_batch',lambda *_a:active)
    monkeypatch.setattr(module,'_validate_live_identity',lambda *_a:active['identity'])
    spool=tmp_path/'spool'/'batch-0021';spool.mkdir(parents=True)
    (tmp_path/'spool'/'publisher-state.json').write_text(json.dumps({'active_identity':['sid',21,'t'*64],
        'journal_sha256':hashlib.sha256(b'journal').hexdigest(),'journal_changed_at':time.time()-10}))
    flushes=[]
    monkeypatch.setattr(module,'export_finished_tranches',lambda **kw:flushes.append(kw['flush_partial']) or {
        'state':{'completed_keys':[],'tranches':[]},'queue_count':0,'new_tranches':0,
        'durable_completed_keys':0,'training_admission':False,'exposure':False})
    result=module.run_guardian_export_service(publisher_path,poll_seconds=0,max_polls=1,
        allowed_spool_root=tmp_path)
    assert flushes==[True]
    assert result['active'] is True and result['training_admission'] is False


def test_guardian_discovery_advances_from_anchor_to_started_batch_with_same_stable_identity():
    from types import SimpleNamespace
    from scripts.coreweave.teacher_incremental_export import discover_guardian_batch
    root='/work/generation'; control='/work/control'; batch=22
    source_root=f'{root}/evidence-collections-v2/batch-0022'
    cases_path=f'{source_root}/source-cases.jsonl'; queue_path=f'{source_root}/queue.jsonl'
    teacher_path=f'{source_root}/teacher-source.json'
    cases=b'{"program_ir":{"id":"new-task"}}\n'
    queue_row={'key':'s7-task-000-00','index':0,'count':1,'sample':0,'seed':7,'task_id':'new-task',
        'source':cases_path,'output':source_root+'/out/0/results.jsonl','jobs':source_root+'/out/0/jobs'}
    queue=json.dumps(queue_row).encode()+b'\n'
    teacher={'schema':'natlang.sandbox-strong-teacher-collection/1','batch':batch,
        'source_cases_sha256':hashlib.sha256(cases).hexdigest(),'actor':{'sha256':'a'*64},
        'teacher':{'model':'qwen-test'},
        'actual_teacher_execution_identity':{'sha256':'b'*64},'actual_task_runtime':'/work/runtime',
        'collection_controller':{'sha256':'c'*64},'runtime_source_receipt':{'sha256':'d'*64},
        'teacher_configuration_source':{'sha256':'e'*64},'source_provenance':{'sha256':'f'*64}}
    teacher_raw=json.dumps(teacher).encode()
    sealed={'source-cases.jsonl':{'path':cases_path,'bytes':len(cases),'sha256':hashlib.sha256(cases).hexdigest()},
        'queue.jsonl':{'path':queue_path,'bytes':len(queue),'sha256':hashlib.sha256(queue).hexdigest()},
        'teacher-source.json':{'path':teacher_path,'bytes':len(teacher_raw),'sha256':hashlib.sha256(teacher_raw).hexdigest()}}
    receipt=json.dumps({'batch':batch,'ordinal':1,'sealed':sealed,'selection_intent':{'sha256':'1'*64},
        'claim':'actual sealed batch'}).encode()
    started_path=control+'/source-sequence/started/batch-0022.json'
    started={'path':started_path,'bytes':len(receipt),'sha256':hashlib.sha256(receipt).hexdigest()}
    guardian={'control':control,'root':root,'feeder':control+'/feed.py','task_runtime':'/work/runtime',
        'teacher_identity':{'sha256':'b'*64},'source_sequence':{'anchor':{'batch':21,'cases_sha256':'a'*64}}}
    guardian_raw=json.dumps(guardian).encode(); config_path=control+'/config.json'
    config_ref={'path':config_path,'bytes':len(guardian_raw),'sha256':hashlib.sha256(guardian_raw).hexdigest()}
    state={'schema':'natlang.teacher-generation-guardian/1','config':config_ref,
        'owner':{'pid':42,'start_ticks':'777','boot_id':'boot'},'active_batch':batch,
        'source_sequence':{'started':{'22':started}}}
    remote={config_path:guardian_raw,control+'/state.json':json.dumps(state).encode(),started_path:receipt,
        cases_path:cases,queue_path:queue,teacher_path:teacher_raw,
        root+'/feed-evidence-phase.json':json.dumps({'batch':batch,'phase':'collecting','worker_count':1,
            'worker_pids':[500],'out':source_root,'cases':1,'teacher':'qwen-test'}).encode(),
        source_root+'/queue-worker-0.jsonl':queue,source_root+'/journal-worker-0.jsonl':b''}
    def exec_(argv,**kwargs):
        out='boot\n' if argv[0]=='cat' else (
            json.dumps({'queues':['queue-worker-0.jsonl'],'journals':['journal-worker-0.jsonl']})
            if 'glob' in argv[2] else json.dumps({'valid':True}))
        return SimpleNamespace(result=lambda **_:SimpleNamespace(returncode=0,stdout=out,stderr=''))
    publisher={'schema':'natlang.teacher-incremental-export-publisher/1','sandbox_id':'sid',
        'sandbox_boot_id':'boot','guardian_config':config_ref,'control_dir':control,'generation_root':root,
        'anchor_batch':21,'anchor_cases_sha256':'a'*64,'directory_root':'/external/export','role':'bulk',
        'chunk_size':32,'exporter_path':'/owner/export.py','identity':{'sandbox_boot_id':'boot',
        'actor_sha256':'a'*64,'execution_identity_sha256':'b'*64,'teacher_config_sha256':'e'*64,
        'runtime_path':'/work/runtime'}}
    active=discover_guardian_batch(publisher,SimpleNamespace(exec=exec_),remote.__getitem__)
    assert active['batch']==22 and active['source_sequence_started_sha256']==started['sha256']
    assert active['identity']['source_cases_sha256']==hashlib.sha256(cases).hexdigest()


def test_guardian_closed_boundary_flushes_partial_and_seals_exact_outcomes(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import scripts.coreweave.teacher_incremental_export as module
    from scripts.coreweave.teacher_generation_guardian import assess
    entries, queue, journal, context, _, finish = _fixture()
    teacher_value=json.loads(context['teacher-source.json']);teacher_value['actual_task_runtime']='/work/runtime'
    context['teacher-source.json']=json.dumps(teacher_value).encode()
    score={'schema':'natlang.s7-task-rollout-score/1','rollouts':2,'outcome_counts':{'task_success':2}}
    completion=assess(entries,[json.loads(line) for line in journal.splitlines() if json.loads(line).get('event')=='finish'],[0],score)
    source_root='/generation/evidence-collections-v2/batch-0019'
    completion.update(batch=19,bindings={name:{'path':path,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
        for name,path,raw in [('queue.jsonl',source_root+'/queue.jsonl',queue),
            ('journal.jsonl',source_root+'/journal.jsonl',journal),
            ('score.json',source_root+'/score.json',json.dumps(score).encode()),
            ('teacher-source.json',source_root+'/teacher-source.json',context['teacher-source.json'])]})
    completion_raw=json.dumps(completion).encode()
    paths={'teacher-source.json':source_root+'/teacher-source.json',
        'source-cases.jsonl':source_root+'/source-cases.jsonl',
        'queue.jsonl':source_root+'/queue.jsonl',
        'journal.jsonl':source_root+'/journal.jsonl',
        'score.json':source_root+'/score.json',
        'rewards.jsonl':source_root+'/rewards.jsonl',
        'completion.json':'/control/boundaries/batch-0019.json',
        'repair-queue.jsonl':'/control/repairs/batch-0019/queue.jsonl'}
    cases=context['source-cases.jsonl'];teacher=context['teacher-source.json']
    remote={paths['teacher-source.json']:teacher,paths['source-cases.jsonl']:cases,
        paths['queue.jsonl']:queue,paths['journal.jsonl']:journal,
        paths['score.json']:json.dumps(score).encode(),paths['rewards.jsonl']:b'{}\n',
        paths['completion.json']:completion_raw}
    boundary_ref={'path':paths['completion.json'],'bytes':len(completion_raw),
        'sha256':hashlib.sha256(completion_raw).hexdigest()}
    publisher={'anchor_batch':19,'anchor_cases_sha256':hashlib.sha256(cases).hexdigest(),
        'directory_root':str(tmp_path/'spool'),
        'generation_root':'/generation','control_dir':'/control','chunk_size':32,
        'exporter_path':'/owner/export.py','role':'bulk','identity':{'actor_sha256':'a'*64,
        'execution_identity_sha256':'b'*64,'teacher_config_sha256':'f'*64,'runtime_path':'/work/runtime'}}
    active={'closed_boundary':boundary_ref,'guardian_state':{'last_boundary':boundary_ref},
        'guardian_state_sha256':'b'*64}
    flush=[]; sealed=[]
    monkeypatch.setattr(module,'export_finished_tranches',lambda **kw:flush.append(kw) or {'training_admission':False})
    def export_boundary(**kw):
        sealed.append(kw)
        folder=kw['directory'];folder.mkdir(parents=True,exist_ok=True)
        (folder/'boundary-private-export.json').write_text(json.dumps({'private_export':{'identity':{'bytes':123,'sha256':'c'*64}}}))
    monkeypatch.setattr(module,'export_final_boundary',export_boundary)
    sandbox=SimpleNamespace(exec=lambda *_a,**_k:SimpleNamespace(result=lambda **_r:SimpleNamespace(returncode=1)))
    result=module.export_guardian_closed_boundary(publisher,active,sandbox=sandbox,
        read_file=remote.__getitem__,spool_root=tmp_path/'spool')
    assert flush[0]['flush_partial'] is True
    assert flush[0]['journal_snapshot']==journal and flush[0]['queue_snapshot']==queue
    assert sealed[0]['exported_keys']==[row['key'] for row in entries]
    assert sealed[0]['repair_keys']==[] and sealed[0]['unstarted_keys']==[]
    assert result['training_admission'] is False and result['exposure'] is False


def test_guardian_exporter_persists_and_drains_every_closed_boundary_once(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import scripts.coreweave.teacher_incremental_export as module
    publisher={'schema':'natlang.teacher-incremental-export-publisher/1','sandbox_id':'sid',
        'directory_root':str(tmp_path/'spool'),'role':'bulk','exporter_path':'/owner/export.py',
        'chunk_size':32}
    publisher_path=tmp_path/'publisher.json';publisher_path.write_text(json.dumps(publisher))
    refs=[{'path':f'/work/control/boundaries/batch-{batch:04d}.json',
           'sha256':str(batch)*64,'bytes':batch} for batch in (21,22,23)]
    active={'draining':True,'batch':24,'guardian_state':{},'guardian_state_sha256':'g'*64,
            'closed_boundaries':refs}
    calls=[]
    monkeypatch.setattr(module,'_cwctl_module',lambda:SimpleNamespace(load_env_file=lambda:None))
    monkeypatch.setattr(module,'scoped_sdk_session',lambda *_a,**_k:(object(),lambda _p:b''))
    monkeypatch.setattr(module,'discover_guardian_batch',lambda *_a:active)
    monkeypatch.setattr(module,'export_guardian_closed_boundary',
        lambda _pub, value, **_kw: calls.append(value['closed_boundary']['path']) or
            {'batch':int(value['closed_boundary']['path'][-9:-5]),'training_admission':False,'exposure':False})
    first=module.run_guardian_export_service(publisher_path,poll_seconds=0,max_polls=1,
        allowed_spool_root=tmp_path)
    assert len(calls)==3
    assert set(first['closed_boundaries'])=={'0021','0022','0023'}
    second=module.run_guardian_export_service(publisher_path,poll_seconds=0,max_polls=1,
        allowed_spool_root=tmp_path)
    assert len(calls)==3
    assert second['closed_boundaries']==first['closed_boundaries']


@pytest.mark.parametrize('mutation', ['foreign-key', 'duplicate-finish', 'unsafe-key'])
def test_completed_frontier_rejects_forged_or_ambiguous_identity(tmp_path, mutation):
    entries, queue, journal, context, _, finish = _fixture()
    if mutation == 'foreign-key':
        journal = json.dumps(finish('foreign')).encode() + b'\n'
    elif mutation == 'duplicate-finish':
        event = finish(entries[0]['key'])
        conflicting = json.loads(json.dumps(event)); conflicting['status'] = 'rewritten'
        journal = json.dumps(event).encode() + b'\n' + json.dumps(conflicting).encode() + b'\n'
    else:
        row = dict(entries[0], key='../escape')
        queue = json.dumps(row).encode() + b'\n'
        event = finish('../escape'); event['output_accounting']['output_sha256'] = 'a'*64
        journal = json.dumps(event).encode() + b'\n'
        entries = [row]
    if mutation == 'unsafe-key':
        selected = completed_entries(queue, journal)
        outputs = {'../escape': (b'{}\n', b'{}')}
        with pytest.raises(ValueError, match='safe'):
            seal_tranche(batch=19, number=0, entries=selected,
                context={**context, 'queue.jsonl': queue},
                outputs=outputs, directory=tmp_path/'unsafe-tranche-test')
    else:
        with pytest.raises(ValueError):
            completed_entries(queue, journal)


def test_seal_binds_exact_raw_outputs_context_and_private_verification(tmp_path):
    entries, queue, journal, context, outputs, _ = _fixture()
    selected = completed_entries(queue, journal)
    tranche = seal_tranche(batch=19, number=0, entries=selected, context=context,
                           outputs=outputs, directory=tmp_path/'tranche-0000')
    assert tranche['keys'] == [row['key'] for row in entries]
    assert tranche['training_admission'] is False and tranche['exposure'] is False
    assert tranche['archive_proof']['status'] == 'passed'
    assert seal_tranche(batch=19, number=0, entries=selected, context=context,
                        outputs=outputs, directory=tmp_path/'tranche-0000') == tranche
    receipt = {'private': True, 'run': 'werg/natlang/test', 'name': 'batch/tranche.tar.zst',
        'identity': {'bytes': tranche['archive_receipt']['archive_bytes'],
                     'sha256': tranche['archive_receipt']['archive_sha256']},
        'independent_verification': {'bytes': tranche['archive_receipt']['archive_bytes'],
                     'sha256': tranche['archive_receipt']['archive_sha256']}, 'verified_at': 123.0}
    identity = receipt['identity']
    receipt_path = tmp_path/'private.json'; receipt_path.write_text(json.dumps(receipt))
    accepted = accept_private_receipt(receipt_path, tranche)
    assert accepted['private_export']['identity'] == receipt['identity']
    receipt['independent_verification']['sha256'] = '0'*64
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='independent full-object'):
        accept_private_receipt(receipt_path, tranche)
    receipt['independent_verification'] = {
        'status': 'awaiting_target_readback', 'bytes': identity['bytes'], 'sha256': identity['sha256']}
    receipt['identity'] = identity
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='independent full-object'):
        accept_private_receipt(receipt_path, tranche)


def test_seal_rejects_output_manifest_mismatch_without_rewriting_tranche(tmp_path):
    entries, queue, journal, context, outputs, _ = _fixture()
    selected = completed_entries(queue, journal)
    broken = dict(outputs)
    raw, manifest = broken[entries[0]['key']]
    value = json.loads(manifest); value['source'] = '/other/source.jsonl'
    broken[entries[0]['key']] = raw, json.dumps(value).encode()
    with pytest.raises(ValueError, match='does not bind'):
        seal_tranche(batch=19, number=0, entries=selected, context=context,
                     outputs=broken, directory=tmp_path/'tranche')


def test_retry_with_changed_tranche_bytes_does_not_mutate_sealed_payload(tmp_path):
    entries, queue, journal, context, outputs, _ = _fixture()
    selected = completed_entries(queue, journal)
    directory = tmp_path/'tranche'
    seal_tranche(batch=19, number=0, entries=selected, context=context,
                 outputs=outputs, directory=directory)
    result_path = directory/'outputs/0000-s7-task-000-00.results.jsonl'
    before = result_path.read_bytes()
    changed = dict(outputs)
    changed_payload = b'{"changed":true}\n'
    changed_manifest = json.loads(outputs[entries[0]['key']][1])
    changed_sha = hashlib.sha256(changed_payload).hexdigest()
    changed_manifest['output_sha256'] = changed_sha
    changed[entries[0]['key']] = (changed_payload, json.dumps(changed_manifest).encode())
    changed_entries = json.loads(json.dumps(selected))
    changed_entries[0]['finish_event']['output_accounting']['output_sha256'] = changed_sha
    with pytest.raises(ValueError, match='identity or payload bindings differ'):
        seal_tranche(batch=19, number=0, entries=changed_entries, context=context,
                     outputs=changed, directory=directory)
    assert result_path.read_bytes() == before


def test_spool_prune_requires_private_full_get_and_retains_member_manifest(tmp_path):
    entries, queue, journal, context, outputs, _ = _fixture()
    selected = completed_entries(queue, journal)
    directory = tmp_path/'tranche'
    tranche = seal_tranche(batch=19, number=0, entries=selected, context=context,
        outputs=outputs, directory=directory)
    archive = Path(tranche['archive_receipt']['archive'])
    raw_member = directory/'outputs/0000-s7-task-000-00.results.jsonl'
    receipt_path = tmp_path/'private.json'
    receipt_path.write_text(json.dumps({'private': True, 'run': 'werg/natlang/test',
        'name': 'tranche.tar.zst', 'identity': {'bytes': archive.stat().st_size,
            'sha256': tranche['archive_receipt']['archive_sha256']},
        'independent_verification': {'bytes': archive.stat().st_size,
            'sha256': tranche['archive_receipt']['archive_sha256']}}))
    with pytest.raises(ValueError, match='private receipt binding'):
        prune_verified_local_tranche(tranche)
    assert archive.exists() and raw_member.exists()
    accepted = accept_private_receipt(receipt_path, tranche)
    pruning = prune_verified_local_tranche(accepted)
    assert pruning['pruned'] is True and pruning['archive_bytes_removed'] > 0
    assert not archive.exists() and not raw_member.exists()
    retained = json.loads((directory/'tranche-0000.json').read_text())
    assert retained['archive_receipt']['manifest']['files']
    assert (tmp_path/'teacher-batch-0019-tranche-0000.receipt.json').is_file()


def test_tranche_receipt_resumes_after_archive_was_committed(tmp_path):
    entries, queue, journal, context, outputs, _ = _fixture()
    selected = completed_entries(queue, journal)
    directory = tmp_path/'tranche'
    sealed = seal_tranche(batch=19, number=0, entries=selected, context=context,
                          outputs=outputs, directory=directory)
    (directory/'tranche-0000.json').unlink()
    resumed = seal_tranche(batch=19, number=0, entries=selected, context=context,
                           outputs=outputs, directory=directory)
    assert resumed['archive_receipt']['archive_sha256'] == sealed['archive_receipt']['archive_sha256']
    assert json.loads((directory/'tranche-0000.json').read_text())['keys'] == sealed['keys']


def test_owner_exporter_checkpoints_chunks_and_resumes_sealed_tranche(tmp_path):
    entries, queue, journal, context, outputs, _ = _fixture()
    context_paths = {'teacher-source.json': '/batch/teacher-source.json',
                     'source-cases.jsonl': '/batch/source-cases.jsonl',
                     'repair-queue.jsonl': '/control/repair-queue.jsonl'}
    remote = {'/batch/queue.jsonl': queue, '/batch/journal.jsonl': journal,
              '/batch/teacher-source.json': context['teacher-source.json'],
              '/batch/source-cases.jsonl': context['source-cases.jsonl'],
              '/control/repair-queue.jsonl': context['repair-queue.jsonl']}
    for row in entries:
        remote[row['output']] = outputs[row['key']][0]
        remote[row['output']+'.manifest.json'] = outputs[row['key']][1]
    calls = []
    def export(tranche, receipt_path):
        calls.append(tranche['keys'])
        ident = {'bytes': tranche['archive_receipt']['archive_bytes'],
                 'sha256': tranche['archive_receipt']['archive_sha256']}
        if len(calls) == 1:
            raise RuntimeError('simulate owner process interruption after sealing')
        return {**tranche, 'private_export': {'identity': ident}}
    kwargs = {'batch': 19, 'queue_path': '/batch/queue.jsonl',
        'journal_path': '/batch/journal.jsonl', 'context_paths': context_paths,
        'directory': tmp_path/'exports', 'read_file': remote.__getitem__,
        'export_file': export, 'chunk_size': 2}
    with pytest.raises(RuntimeError, match='interruption'):
        export_finished_tranches(**kwargs)
    sealed = json.loads((tmp_path/'exports/tranche-0000/tranche-0000.json').read_text())
    assert sealed['keys'] == [row['key'] for row in entries]
    assert sealed['private_export'] is None
    result = export_finished_tranches(**kwargs)
    assert result['durable_completed_keys'] == 2
    assert calls[0] == calls[1] == [row['key'] for row in entries]
    assert export_finished_tranches(**kwargs)['durable_completed_keys'] == 2
    with pytest.raises(ValueError, match='different batch or queue'):
        export_finished_tranches(**{**kwargs, 'batch': 20})


def test_verified_private_receipt_survives_crash_before_frontier_advance(tmp_path):
    entries, queue, journal, context, outputs, _ = _fixture()
    context_paths = {'teacher-source.json': '/batch/teacher-source.json',
                     'source-cases.jsonl': '/batch/source-cases.jsonl'}
    remote = {'/batch/queue.jsonl': queue, '/batch/journal.jsonl': journal,
              '/batch/teacher-source.json': context['teacher-source.json'],
              '/batch/source-cases.jsonl': context['source-cases.jsonl']}
    for row in entries:
        remote[row['output']] = outputs[row['key']][0]
        remote[row['output']+'.manifest.json'] = outputs[row['key']][1]
    calls = []
    def export(tranche, receipt_path):
        calls.append(tranche['keys'])
        identity = {'bytes': tranche['archive_receipt']['archive_bytes'],
                    'sha256': tranche['archive_receipt']['archive_sha256']}
        receipt = {'private': True, 'run': 'werg/natlang/test', 'name': 'tranche.tar.zst',
            'identity': identity, 'independent_verification': identity, 'verified_at': 124.0}
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(json.dumps(receipt))
        accept_private_receipt(receipt_path, tranche)
        raise RuntimeError('crash after private verification, before frontier commit')
    kwargs = {'batch': 19, 'queue_path': '/batch/queue.jsonl',
        'journal_path': '/batch/journal.jsonl', 'context_paths': context_paths,
        'directory': tmp_path/'exports', 'read_file': remote.__getitem__,
        'export_file': export, 'chunk_size': 2, 'prune_after_private_verification': True}
    with pytest.raises(RuntimeError, match='before frontier commit'):
        export_finished_tranches(**kwargs)
    result = export_finished_tranches(**{**kwargs,
        'export_file': lambda *_: pytest.fail('already verified private object must not upload again')})
    assert result['durable_completed_keys'] == 2
    assert len(calls) == 1
    tranche_dir = tmp_path/'exports/tranche-0000'
    durable_manifest = json.loads((tranche_dir/'tranche-0000.json').read_text())
    assert not Path(durable_manifest['archive_receipt']['archive']).exists()
    assert sorted(path.name for path in tranche_dir.iterdir()) == ['tranche-0000.json']


def test_final_boundary_requires_exact_partition_and_all_raw_control_files(tmp_path):
    from scripts.coreweave.teacher_generation_guardian import assess
    entries, queue, journal, context, _, finish = _fixture()
    events = [json.loads(line) for line in journal.splitlines()]
    finishes = [row for row in events if row.get('event') == 'finish']
    score = {'schema': 'natlang.s7-task-rollout-score/1', 'rollouts': 2,
             'outcome_counts': {'task_success': 2}}
    completion = assess(entries, finishes, [0], score)
    files = {**context, 'journal.jsonl': journal, 'score.json': json.dumps(score).encode(),
             'rewards.jsonl': b'{}\n', 'completion.json': json.dumps(completion).encode(),
             'repair-queue.jsonl': context['repair-queue.jsonl']}
    completion.update(batch=19, bindings={name: {'bytes': len(files[name]),
        'sha256': hashlib.sha256(files[name]).hexdigest()} for name in (
            'queue.jsonl', 'journal.jsonl', 'score.json', 'teacher-source.json')})
    files['completion.json'] = json.dumps(completion).encode()
    keys = [row['key'] for row in entries]
    with pytest.raises(ValueError, match='do not partition'):
        seal_boundary(batch=19, queue_raw=queue, exported_keys=keys[:1], repair_keys=[],
            unstarted_keys=[], files=files, directory=tmp_path/'bad')
    boundary = seal_boundary(batch=19, queue_raw=queue, exported_keys=keys,
        repair_keys=[], unstarted_keys=[], files=files, directory=tmp_path/'final')
    assert boundary['archive_proof']['status'] == 'passed'
    assert boundary['boundary']['training_admission'] is False
    resumed = seal_boundary(batch=19, queue_raw=queue, exported_keys=keys,
        repair_keys=[], unstarted_keys=[], files=files, directory=tmp_path/'final')
    assert resumed['archive_receipt']['archive_sha256'] == boundary['archive_receipt']['archive_sha256']
    altered = dict(files, **{'score.json': b'{"changed":true}'})
    with pytest.raises(ValueError, match='does not bind current score'):
        seal_boundary(batch=19, queue_raw=queue, exported_keys=keys,
            repair_keys=[], unstarted_keys=[], files=altered, directory=tmp_path/'final')


def test_final_boundary_preserves_actual_repair_error_without_complete_batch_claim(tmp_path):
    from scripts.coreweave.teacher_generation_guardian import assess
    entries, queue, journal, context, _, finish = _fixture()
    row = entries[0]
    error = {'index': 0, 'program_id': row['task_id'], 'program_ir_sha256': '1'*64,
             'source_sha256': hashlib.sha256(context['source-cases.jsonl']).hexdigest(),
             'error': 'provider unavailable'}
    error_raw = json.dumps(error).encode()
    failed_accounting = {'version': 'natlang.supervisor_output_accounting/2', 'complete': False,
        'manifest_valid': False, 'manifest_errors': ['terminal_error'], 'expected_jobs': 1,
        'job_states': [{'index': 0, 'program_id': row['task_id'], 'digest': '1'*64,
                        'state': 'failed', 'terminal_error': {'path': '/batch/jobs/000000.error.json',
                            'sha256': hashlib.sha256(error_raw).hexdigest()}}]}
    failed = {'event': 'finish', 'key': row['key'], 'output_accounting': failed_accounting}
    outcomes = [dict(entry, finish=failed if entry['key'] == row['key'] else None) for entry in entries]
    # The shared assessor is authoritative for repair disposition; its exact
    # schema contains no synthetic batch/can_advance fields at top level.
    score = {'schema': 'natlang.s7-task-rollout-score/1', 'rollouts': 2,
             'outcome_counts': {'task_success': 1, 'provider_failure': 1}}
    completion = assess(entries, [failed], [2], score)
    journal_bytes = b''.join(json.dumps(item).encode()+b'\n' for item in [
        {'event': 'start', 'key': row['key']}, failed])
    completion.update(batch=19, individual_outcomes=outcomes, bindings={
        'queue.jsonl': {'bytes': len(queue), 'sha256': hashlib.sha256(queue).hexdigest()},
        'journal.jsonl': {'bytes': len(journal_bytes), 'sha256': hashlib.sha256(journal_bytes).hexdigest()},
        'score.json': {'bytes': len(json.dumps(score).encode()),
                       'sha256': hashlib.sha256(json.dumps(score).encode()).hexdigest()},
        'teacher-source.json': {'bytes': len(context['teacher-source.json']),
            'sha256': hashlib.sha256(context['teacher-source.json']).hexdigest()}})
    files = {**context, 'queue.jsonl': queue, 'journal.jsonl': journal_bytes,
        'score.json': json.dumps(score).encode(), 'rewards.jsonl': b'{}\n',
        'completion.json': json.dumps(completion).encode(), 'repair-queue.jsonl': (json.dumps(
            {'key': f"batch-0019:{row['key']}"})+'\n').encode(),
        f"repair-payloads/{row['key']}/000000.error.json": error_raw}
    boundary = seal_boundary(batch=19, queue_raw=queue, exported_keys=[],
        repair_keys=[row['key']], unstarted_keys=[entries[1]['key']], files=files,
        directory=tmp_path/'repair-boundary')
    assert boundary['boundary']['batch_can_advance'] is False
    assert boundary['boundary']['training_admission'] is False
