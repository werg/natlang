"""Durable boundary supervision for the existing self-hosted teacher feeder."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
import uuid
try:
    from .teacher_batch_completion import batch_completion
except ImportError:
    from teacher_batch_completion import batch_completion

TOKEN = 'NATLANG_TEACHER_GENERATION_TOKEN'

def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + '.pending')
    raw=value if isinstance(value,bytes) else (json.dumps(value,indent=2)+'\n').encode()
    with pending.open('wb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    os.replace(pending, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)

def binding(path):
    path = Path(path)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': str(path), 'sha256': digest, 'bytes': path.stat().st_size}

def process(pid):
    try:
        root = Path('/proc', str(pid)); stat = (root/'stat').read_text().rsplit(')', 1)[1].split()
        if stat[0] == 'Z': return None
        return {'pid': int(pid), 'start_ticks': stat[19],
                'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                'argv': (root/'cmdline').read_bytes().split(b'\0')[:-1]}
    except (FileNotFoundError, ProcessLookupError): return None

def alive(identity):
    current = process(identity['pid']) if identity else None
    return bool(current and all(current[key] == identity[key] for key in ('start_ticks', 'boot_id')))

def verify_teacher_identity(path, *, server, model_id):
    """Verify the existing serving producer's immutable bytes and actual owners."""
    path=Path(path);descriptor=json.loads(path.read_text())
    if descriptor.get('schema')!='natlang.self-hosted-teacher-execution-identity/1':
        raise ValueError('teacher execution identity schema differs')
    if descriptor.get('server')!=server or descriptor.get('model_id')!=model_id:
        raise ValueError('teacher endpoint/model differs from serving identity')
    actor=descriptor['actor'];actual=binding(actor['path'])
    if actual['sha256']!=actor['sha256']:
        raise ValueError('teacher serving identity producer bytes changed')
    owners=descriptor.get('processes') or []
    if not owners:raise ValueError('teacher serving identity needs actual engine owners')
    for expected in owners:
        current=process(expected['pid'])
        if (not alive(expected) or current is None
                or hashlib.sha256(b'\0'.join(current['argv'])).hexdigest()!=expected['argv_sha256']):
            raise ValueError('actual teacher engine process identity changed')
    return {'identity':binding(path),'actor':actual,'processes':owners,
        'server':server,'model_id':model_id,'claim':'actual independent retry actor, not the original source-batch actor'}

def load_teacher_source(reference, *, identity, server, model_id):
    """Authenticate a selected teacher without relabelling historical batches."""
    actual = binding(reference['path'])
    if actual != reference:
        raise ValueError('teacher source descriptor binding changed')
    value = json.loads(Path(reference['path']).read_bytes())
    if (value.get('schema') != 'natlang.self-hosted-teacher-source/1'
            or value.get('server') != server or value.get('model_id') != model_id
            or value.get('execution_identity') != binding(identity)):
        raise ValueError('teacher source descriptor differs from actual serving identity')
    teacher = value.get('teacher') or {}
    if not all(isinstance(teacher.get(key), str) and teacher[key] for key in ('model', 'revision')):
        raise ValueError('teacher source needs actual model and revision')
    sampling = value.get('sampling') or {}
    temperature = sampling.get('temperature')
    if (type(temperature) not in (int, float) or not math.isfinite(temperature)
            or temperature < 0):
        raise ValueError('teacher source needs explicit finite sampling temperature')
    if sampling.get('request_config'):
        if binding(sampling['request_config']['path']) != sampling['request_config']:
            raise ValueError('teacher request configuration binding changed')
    verify_teacher_identity(identity, server=server, model_id=model_id)
    return value

def teacher_environment(config):
    """Transport the declared teacher identically to every collection feeder."""
    env = {'NATLANG_TEACHER_SERVER': config['server'],
           'NATLANG_TEACHER_MODEL_ID': config['model_id']}
    if config.get('teacher_source'):
        descriptor = load_teacher_source(config['teacher_source'], identity=config['teacher_identity']['path'],
                                         server=config['server'], model_id=config['model_id'])
        if descriptor['execution_identity'] != config['teacher_identity']:
            raise ValueError('guardian teacher identity binding differs from selected source')
        env['NATLANG_TEACHER_SOURCE_BINDING'] = json.dumps(config['teacher_source'], sort_keys=True)
    elif os.environ.get('NATLANG_TEACHER_SOURCE_BINDING'):
        raise ValueError('ambient teacher descriptor is not declared by guardian configuration')
    return env

def _feeder_invocation(current, path):
    if current is None:return False
    argv=current['argv'];argument=str(path).encode()
    launching=(len(argv)==6 and Path(argv[0].decode()).name=='nice'
               and argv[1:]==[b'-n',b'10',b'python',b'-u',argument])
    return argument in argv[:5] or launching

def record_primary_phase(state, config):
    """Bind a canonical primary feeder's per-batch phase without relaunching it."""
    directory=config.get('primary_phase_directory')
    if directory is None:return False  # Historical feeders use the root phase.
    if Path(directory)!=Path(config['root'])/'feeder-phases':
        raise ValueError('primary phase directory differs from canonical generation root')
    batch=state.get('active_batch');owner=state.get('owner')
    if type(batch)is not int or not alive(owner):return False
    launches=state.setdefault('primary_launches',{})
    if str(batch)in launches:return False  # Never rewrite an existing slot's custody.
    feeder=state['feeder'];reference=feeder if isinstance(feeder,dict)else binding(feeder)
    if binding(reference['path'])!=reference or not _feeder_invocation(process(owner['pid']),reference['path']):
        raise ValueError('primary phase lacks its exact live feeder custody')
    launches[str(batch)]={'batch':batch,'owner':owner,'feeder':reference,
        'task_runtime':state['task_runtime'],'phase_path':str(Path(directory)/f'batch-{batch:04d}.json')}
    return True

def recover_token(token, *, script_prefix='feed-evidence-'):
    found = []
    for root in Path('/proc').iterdir():
        if not root.name.isdigit(): continue
        try:
            if (TOKEN+'='+token).encode() not in (root/'environ').read_bytes().split(b'\0'): continue
            current = process(root.name)
            if current: found.append(current)
        except (FileNotFoundError, PermissionError, ProcessLookupError): pass
    # Only the feeder root is an owner; descendants inherit the token.
    roots = [row for row in found if any(Path(arg.decode(errors='replace')).name.startswith(script_prefix) for arg in row['argv'][:5])]
    if len(roots) > 1: raise ValueError('multiple feeders own the same prepared process token')
    return roots[0] if roots else None

def public_identity(identity):
    return {key: value for key, value in identity.items() if key != 'argv'}


def fanout_frontier(queue, journals, workers):
    """Partition exact unstarted keys after collectors have finished their cases."""
    if type(workers) is not int or workers < 1:
        raise ValueError('fanout needs a positive worker count')
    entries = [json.loads(line) for line in Path(queue).read_text().splitlines() if line]
    by_key = {entry['key']: entry for entry in entries}
    if not entries or len(by_key) != len(entries):
        raise ValueError('fanout queue needs distinct original keys')
    started, finished = set(), {}
    for journal in journals:
        for line in Path(journal).read_text().splitlines():
            if not line: continue
            row = json.loads(line)
            if row.get('event') not in ('start', 'finish'): continue
            key = row['key']
            if key not in by_key: raise ValueError('fanout journal contains a foreign key')
            if row['event'] == 'start': started.add(key)
            else:
                if key in finished and finished[key] != row:
                    raise ValueError('fanout journal has conflicting terminal observations')
                finished[key] = row
    if started - finished.keys():
        raise ValueError('fanout still has an unfinished started case')
    remaining = [entry for entry in entries if entry['key'] not in finished]
    return {'schema':'natlang.teacher-drained-fanout/1', 'queue':binding(queue),
        'journals':[binding(path) for path in journals], 'workers':workers,
        'finished_keys':sorted(finished), 'unstarted_keys':[entry['key'] for entry in remaining],
        'shards':[remaining[index::workers] for index in range(workers)],
        'training_admission':False, 'original_completion_claim':False}


def adopt_drained_fanout(folder, transition_path):
    """CAS a drained same-batch owner; preserve original results and repair debt.

    The caller holds guardian.lock and has already launched a waiting successor.
    ARMED is published only after durable owner adoption, never before it.
    """
    folder, transition_path = Path(folder), Path(transition_path)
    transition = json.loads(transition_path.read_text())
    if transition.get('schema') != 'natlang.teacher-drained-fanout-transition/1':
        raise ValueError('fanout transition schema differs')
    def checked(reference):
        if binding(reference['path']) != reference:
            raise ValueError('fanout transition binding changed')
        return json.loads(Path(reference['path']).read_text())
    identity = binding(transition_path)
    transition_id = transition['id']
    if (not isinstance(transition_id,str) or not transition_id
            or Path(transition_id).name != transition_id or transition_id in ('.','..')):
        raise ValueError('fanout transition id escaped its control directory')
    state = json.loads((folder/'state.json').read_text())
    prior = state.get('fanout_transitions', [])
    arm = folder/'fanout'/transition['id']/'ARMED.json'
    if any(row['transition'] == identity for row in prior):
        if state.get('owner') == transition['owner'] and not arm.exists():
            write(arm, {'transition':identity,'owner':transition['owner']})
        return state
    original = checked(transition['original_state'])
    if state != original or state.get('active_batch') != transition['batch']:
        raise ValueError('fanout no longer owns the exact original batch/state')
    intent = checked(transition['drain'])
    if intent.get('owner') != state['owner'] or not Path(intent['drain_file']).exists():
        raise ValueError('fanout lacks its exact operator drain')
    if alive(state.get('owner')) or any(alive(child) for child in intent['children']):
        raise ValueError('fanout original owner or collectors remain alive')
    config = checked(transition['config'])
    old_config = checked(original['config'])
    if ({key:value for key,value in config.items() if key != 'feeder'} !=
            {key:value for key,value in old_config.items() if key != 'feeder'}):
        raise ValueError('fanout changed actor/runtime or unrelated configuration')
    if config['control'] != str(folder) or config['drain_file'] != intent['drain_file']:
        raise ValueError('fanout changed its control/drain namespace')
    if binding(config['feeder']) != transition['feeder']:
        raise ValueError('fanout feeder bytes changed')
    plan = checked(transition['frontier'])
    actual = fanout_frontier(plan['queue']['path'],
        [item['path'] for item in plan['journals']], plan['workers'])
    if actual != plan: raise ValueError('fanout queue/journal frontier changed')
    expected_root = Path(config['root'])/'evidence-collections-v2'/f"batch-{transition['batch']:04d}"
    if Path(plan['queue']['path']) != expected_root/'queue.jsonl':
        raise ValueError('fanout belongs to another original queue')
    if ({Path(item['path']) for item in plan['journals']} !=
            set(expected_root.glob('journal*.jsonl'))):
        raise ValueError('fanout omitted an original journal')
    if live_batch_workers(expected_root):
        raise ValueError('fanout still has actual live original collectors')
    if binding(expected_root/'teacher-source.json') != transition['source']:
        raise ValueError('fanout original source actor changed')
    owner = recover_token(transition['process_token'])
    if (owner is None or public_identity(owner) != transition['owner']
            or config['feeder'].encode() not in owner['argv']):
        raise ValueError('fanout successor is not its exact waiting owner/token')
    verify_teacher_identity(config['teacher_identity']['path'],server=config['server'],model_id=config['model_id'])
    config_revision(folder,state,transition['config']['path'],publish=False)
    state['owner'] = transition['owner']
    state.setdefault('fanout_transitions', []).append({'transition':identity,
        'frontier':transition['frontier'], 'batch':transition['batch'],
        'original_source':transition['source'], 'claim':'same batch; original outcomes unchanged'})
    write(folder/'state.json',state)
    write(arm, {'transition':identity,'owner':state['owner']})
    return state

def assess(entries, journal, codes, score):
    completion = batch_completion(entries, journal, codes, score)
    finishes = {row['key']: row for row in journal if row.get('event') == 'finish'}
    return {'schema': 'natlang.teacher-generation-boundary/1', 'completion': completion,
            'batch_complete': bool(completion['can_advance']),
            'repair_required': completion['repair_required'],
            'individual_outcomes': [{'key': entry['key'], 'source': entry['source'],
                'index': entry['index'], 'seed': entry['seed'],
                'finish': finishes.get(entry['key'])} for entry in entries]}

def boundary(root, folder, expected_batch=None, *, phase_path=None):
    phase = json.loads(Path(phase_path or root/'feed-evidence-phase.json').read_text())
    if phase['phase'] != 'batch_complete': raise ValueError('no drained batch boundary; preserve state for targeted repair')
    batch = phase['batch']
    if expected_batch is not None and batch != expected_batch: raise ValueError('feeder exited at a different batch boundary')
    source = root/'evidence-collections-v2'/f'batch-{batch:04d}'
    if live_batch_workers(source): raise ValueError('batch collectors have not drained')
    entries = [json.loads(line) for line in (source/'queue.jsonl').read_text().splitlines() if line]
    journal = [json.loads(line) for line in (source/'journal.jsonl').read_text().splitlines() if line]
    decision = assess(entries, journal, phase['returncodes'], json.loads((source/'score.json').read_text()))
    decision.update(at=time.time(), batch=batch, bindings={name: binding(source/name)
        for name in ('queue.jsonl', 'journal.jsonl', 'score.json', 'teacher-source.json')})
    receipt = folder/'boundaries'/f'batch-{batch:04d}.json'
    if receipt.exists():
        previous = json.loads(receipt.read_text())
        if {k:v for k,v in previous.items() if k != 'at'} != {k:v for k,v in decision.items() if k != 'at'}:
            raise ValueError('previously recorded immutable batch boundary changed')
        decision = previous
    else: write(receipt, decision)
    return decision, entries

def prepare_retry(folder, decision, entries):
    """One distinct reviewed retry per unresolved key; original outputs are untouched."""
    wanted = {row['key'] for row in decision['repair_required']}
    rows = []
    for entry in entries:
        if entry['key'] not in wanted: continue
        leaf = folder/'repairs'/f"batch-{decision['batch']:04d}"/entry['key']
        if Path(entry['key']).name != entry['key'] or entry['key'] in ('.', '..'):
            raise ValueError('repair key is not a plain queue key')
        revised = {**entry, 'key': f"batch-{decision['batch']:04d}:"+entry['key'],
            'jobs': str(leaf/'jobs'), 'output': str(leaf/'results.jsonl'), 'log': str(leaf/'collector.log')}
        prepared = {'original': entry, 'retry': revised,
            'boundary': binding(folder/'boundaries'/f"batch-{decision['batch']:04d}.json"),
            'claim': 'independent retry; original batch stays incomplete unless separately reconciled'}
        if (leaf/'prepared.json').exists():
            if json.loads((leaf/'prepared.json').read_text()) != prepared:
                raise ValueError('repair source/seed/boundary changed')
        else: write(leaf/'prepared.json', prepared)
        rows.append(revised)
    if rows:
        path = folder/'repairs'/f"batch-{decision['batch']:04d}"/'queue.jsonl'
        # One atomic queue publication; no changes to a queue already consumed.
        text = ''.join(json.dumps(row)+'\n' for row in rows)
        if path.exists():
            if path.read_text() != text: raise ValueError('published repair queue changed')
            return rows
        pending = path.with_suffix('.pending'); pending.write_text(text)
        with pending.open('rb') as stream: os.fsync(stream.fileno())
        os.replace(pending, path)
    return rows

def _source_sequence_helpers():
    try:
        from . import teacher_source_sequence
    except ImportError:
        import teacher_source_sequence
    return teacher_source_sequence


def launch_feeder(folder, state, config, batch):
    target = folder/'state.json'
    launch = state.get('prepared_launch')
    if launch and launch['batch'] != batch: raise ValueError('prepared launch batch mismatch')
    if launch:
        recovered = recover_token(launch['token'])
        if recovered:
            state['owner'] = public_identity(recovered); state['active_batch'] = batch
            state.pop('prepared_launch', None); write(target, state); return
        # A prepared token with no live owner may have run and exited before the
        # supervisor recovered. A materialized batch must be reconciled first.
        if (Path(config['root'])/'evidence-collections-v2'/f'batch-{batch:04d}').exists():
            state.update(owner=None, active_batch=batch); state.pop('prepared_launch', None)
            write(target, state); return
    else:
        if (Path(config['root'])/'evidence-collections-v2'/f'batch-{batch:04d}').exists():
            raise ValueError('next batch already exists without this prepared launch')
        if config.get('source_sequence'):
            _source_sequence_helpers().prepare_source_sequence(folder,state,config,batch)
        launch = {'batch': batch, 'token': uuid.uuid4().hex}; state['prepared_launch'] = launch; write(target, state)
    for root in Path('/proc').iterdir():
        if not root.name.isdigit(): continue
        current = process(root.name)
        if current and any(Path(arg.decode(errors='replace')).name.startswith('feed-evidence-') for arg in current['argv'][:5]):
            raise ValueError('another live feeder already owns generation')
    env = {**os.environ, TOKEN: launch['token'], 'NATLANG_TEACHER_START_BATCH': str(batch),
           'NATLANG_TEACHER_SINGLE_BATCH': '1', 'NATLANG_TEACHER_TASK_RUNTIME': config['task_runtime'],
           'NATLANG_TEACHER_IDENTITY':config['teacher_identity']['path']}
    env.update(teacher_environment(config))
    if config.get('source_sequence'):
        _source_sequence_helpers().verify_prepared_source(state,config,batch)
    log = (folder/'feeder.log').open('ab')
    child = subprocess.Popen(['nice', '-n', '10', 'python', '-u', config['feeder']],
        env=env, stdout=log, stderr=log, start_new_session=True)
    identity = process(child.pid)
    if identity is None: raise ValueError('new feeder exited before process identity publication')
    state.update(owner=public_identity(identity), active_batch=batch, feeder=binding(config['feeder']),
        task_runtime=config['task_runtime']); state.pop('prepared_launch', None); write(target, state)
    if record_primary_phase(state,config):write(target,state)

def repair_outcomes(queue, journal):
    entries = [json.loads(line) for line in queue.read_text().splitlines() if line]
    keys = {entry['key'] for entry in entries}
    rows = [json.loads(line) for line in journal.read_text().splitlines() if line] if journal.exists() else []
    if any(row.get('key') not in keys for row in rows): raise ValueError('repair journal has an unknown queue key')
    finishes = {row['key']: row for row in rows if row.get('event') == 'finish'}
    return entries, finishes


def _launch_one_repair(folder, state, config, *, source_queue=None, per_batch_lock=False):
    """Recover one shared process-token protocol, then retry only unfinished keys."""
    active = state.get('repair_owner')
    if active and alive(active): return
    queues = [source_queue] if source_queue is not None else sorted((folder/'repairs').glob('batch-*/queue.jsonl'))
    for source_queue in queues:
        receipt_path = source_queue.parent/'launch.json'
        receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {
            'schema': 'natlang.teacher-repair-attempts/1', 'source_queue': binding(source_queue),
            'terminal_keys': {}, 'attempts': [], 'current': None}
        if receipt['source_queue'] != binding(source_queue): raise ValueError('repair source queue changed')
        attempt = receipt.get('current')
        if attempt:
            recovered = recover_token(attempt['token'], script_prefix='run_bonsai_queue.py')
            if recovered:
                attempt['owner'] = public_identity(recovered); attempt['state'] = 'launched'
                state['repair_owner'] = attempt['owner']; write(receipt_path, receipt); write(folder/'state.json', state); return
            if alive(attempt.get('owner')):
                state['repair_owner'] = attempt['owner']; write(folder/'state.json', state); return
            queue, journal = Path(attempt['queue']), Path(attempt['journal'])
            _, finishes = repair_outcomes(queue, journal)
            receipt['terminal_keys'].update(finishes)
            if journal.exists():
                attempt.update(state='drained', journal_binding=binding(journal),
                    terminal_keys=sorted(finishes))
                receipt['attempts'].append(attempt); receipt['current'] = None; attempt = None
            if attempt is not None and not journal.exists() and any(attempt.get(key)!=config.get(field) for key,field in
                    [('runtime','task_runtime'),('server','server'),('model_id','model_id'),('teacher_identity','teacher_identity'),('teacher_source','teacher_source')]):
                attempt['state']='superseded_unstarted_configuration_revision';receipt['attempts'].append(attempt)
                receipt['current']=None;attempt=None
            # No journal and no live token means the intent never reached an
            # observable request. Recover this SAME intent rather than losing it.
        originals = [json.loads(line) for line in source_queue.read_text().splitlines() if line]
        remaining = [row for row in originals if row['key'] not in receipt['terminal_keys']]
        if not remaining:
            receipt['state'] = 'attempt_exhausted'; receipt['current'] = None
            receipt['claim'] = 'actual terminal retry outcomes retained; original batch is not relabelled complete'
            write(receipt_path, receipt); continue
        if attempt is None:
            selected = None
            if config.get('teacher_source'):
                teacher_environment(config)
                selected = load_teacher_source(config['teacher_source'], identity=config['teacher_identity']['path'],
                                               server=config['server'], model_id=config['model_id'])
            number = len(receipt['attempts'])+1; leaf = source_queue.parent/f'attempt-{number:04d}'
            leaf.mkdir(exist_ok=True)
            rows = [{**row, 'jobs': str(leaf/row['key']/'jobs'),
                'output': str(leaf/row['key']/'results.jsonl'), 'log': str(leaf/row['key']/'collector.log')} for row in remaining]
            if selected:
                for row in rows:
                    row['temperature'] = selected['sampling']['temperature']
            queue = leaf/'queue.jsonl'; text = ''.join(json.dumps(row)+'\n' for row in rows)
            if queue.exists() and queue.read_text() != text: raise ValueError('repair attempt queue changed')
            if not queue.exists():
                pending=queue.with_suffix('.pending');pending.write_text(text)
                with pending.open('rb') as stream: os.fsync(stream.fileno())
                os.replace(pending,queue)
            attempt = {'number':number,'state':'prepared','token':uuid.uuid4().hex,
                'queue':str(queue),'queue_binding':binding(queue),'journal':str(leaf/'journal.jsonl'),
                'runtime':config['task_runtime'],'server':config['server'],'model_id':config['model_id'],
                'node_bin':config['node_bin'],'queue_worker':config['queue_worker'],
                'teacher_identity':config.get('teacher_identity')}
            if selected:
                attempt['teacher_source'] = config['teacher_source']
                attempt['sampling'] = selected['sampling']
            receipt['current'] = attempt; write(receipt_path, receipt)
        queue = Path(attempt['queue'])
        if binding(queue) != attempt['queue_binding']: raise ValueError('prepared repair queue changed')
        env = {**os.environ, TOKEN:attempt['token'], 'PATH':attempt['node_bin']+':'+os.environ.get('PATH',''),
               'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1'}
        command=['flock','-n',str(source_queue.parent/'collector.lock' if per_batch_lock else folder/'repair-collector.lock'),'nice','-n','19','python','-u',attempt['queue_worker'],
            str(queue),attempt['journal'],'--runtime',attempt['runtime'],'--server',attempt['server'],
            '--chat-completions-url',attempt['server']+'/v1/chat/completions','--model-id',attempt['model_id'],
            '--model-concurrency','1','--max-batch-cases','1','--case-seconds','7200','--identical-turn-limit','8']
        if attempt.get('teacher_identity'):
            identity_binding=attempt['teacher_identity']
            if binding(identity_binding['path'])['sha256']!=identity_binding['sha256']:
                raise ValueError('configured teacher execution identity changed')
            attempt['actual_teacher_identity']=verify_teacher_identity(identity_binding['path'],server=attempt['server'],model_id=attempt['model_id'])
            command += ['--teacher-identity',identity_binding['path']]
        if attempt.get('teacher_source'):
            selected = load_teacher_source(attempt['teacher_source'], identity=attempt['teacher_identity']['path'],
                                           server=attempt['server'], model_id=attempt['model_id'])
            if selected['sampling'] != attempt['sampling']:
                raise ValueError('prepared repair sampling differs from selected teacher')
            if attempt['sampling'].get('request_config'):
                command += ['--chat-request-config', attempt['sampling']['request_config']['path']]
        attempt['argv']=command;write(receipt_path,receipt)
        log=(queue.parent/'worker.log').open('ab');child=subprocess.Popen(command,env=env,stdout=log,stderr=log,start_new_session=True)
        identity=process(child.pid)
        if identity:
            attempt.update(state='launched',owner=public_identity(identity));state['repair_owner']=attempt['owner']
        write(receipt_path,receipt);write(folder/'state.json',state);return


def launch_repair(folder, state, config):
    """Fill explicit slots with disjoint batches; preserve all live trajectories."""
    limit = config.get('repair_parallel_batches', 1)
    if type(limit) is not int or limit < 1:
        raise ValueError('repair_parallel_batches must be a positive integer')
    if limit == 1 and not state.get('repair_batch_owners'):
        return _launch_one_repair(folder, state, config)
    queues = sorted((folder/'repairs').glob('batch-*/queue.jsonl'))
    owners = {}
    legacy = state.get('repair_owner')
    for queue in queues:
        receipt_path = queue.parent/'launch.json'
        if not receipt_path.exists():
            continue
        receipt = json.loads(receipt_path.read_text())
        if receipt['source_queue'] != binding(queue):
            raise ValueError('repair source queue changed')
        attempt = receipt.get('current')
        if not attempt:
            continue
        owner = attempt.get('owner')
        # A live flock wrapper and its token-bearing Python child are ONE
        # trajectory. Preserve its authenticated stored owner before recovery.
        recovered = None if alive(owner) else recover_token(
            attempt['token'], script_prefix='run_bonsai_queue.py')
        if recovered:
            owner = public_identity(recovered)
        if not alive(owner):
            continue
        if recovered and attempt.get('owner') != owner:
            attempt.update(owner=owner, state='launched')
            write(receipt_path, receipt)
        owners[str(queue)] = owner
    # A legacy owner without a matching published attempt is still a live
    # obligation. Reserve its slot rather than guessing or killing its work.
    if alive(legacy) and legacy not in owners.values():
        owners['legacy-owner'] = legacy
    def publish():
        state['repair_batch_owners'] = dict(owners)
        state['repair_owner'] = legacy if alive(legacy) else next(iter(owners.values()), None)
        write(folder/'state.json', state)
    publish()
    for queue in queues:
        if len(owners) >= limit:
            break
        if str(queue) in owners:
            continue
        # The old launcher owns the entire immutable batch attempt protocol.
        # Its interim state publication is recovered from the same attempt
        # tokens if the guardian dies before publishing the final owner map.
        serial_state = dict(state)
        serial_state['repair_owner'] = None
        _launch_one_repair(folder, serial_state, config,
                          source_queue=queue, per_batch_lock=True)
        owner = serial_state.get('repair_owner')
        if alive(owner):
            owners[str(queue)] = owner
        publish()


def live_batch_workers(source):
    result=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdigit(): continue
        row=process(path.name)
        if not row or not row['argv'] or not Path(row['argv'][0].decode(errors='replace')).name.startswith('python'): continue
        if (any(Path(arg.decode(errors='replace')).name=='run_bonsai_queue.py' for arg in row['argv'][:5])
                and any(arg.decode(errors='replace').startswith(str(source)+'/') for arg in row['argv'])):
            result.append(public_identity(row))
    return result


def config_revision(folder, state, path, *, publish=True):
    raw=Path(path).read_bytes();config=json.loads(raw);current=binding(path)
    if state.get('config')==current: return config
    previous=state.get('config_value')
    allowed={'task_runtime','server','model_id','teacher_source','primary_phase_directory','feeder','queue_worker','node_bin','drain_file','teacher_identity','source_sequence','repair_parallel_batches','primary_parallel_batches','primary_tail_max_running','primary_worker_capacity','overlap_workers','overlap_feeder','collection_repo','collection_worker'}
    changes={key for key in set(previous or {})|set(config) if (previous or {}).get(key)!=config.get(key)}
    if previous is not None and changes-allowed: raise ValueError('guardian root/ownership configuration changed; preserve and explicitly migrate this state')
    snapshot=folder/'config-history'/(current['sha256']+'.json');snapshot.parent.mkdir(parents=True,exist_ok=True)
    if snapshot.exists():
        if snapshot.read_bytes()!=raw:raise ValueError('config revision bytes changed')
    else:
        pending=snapshot.with_suffix('.pending')
        with pending.open('wb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
        os.replace(pending,snapshot)
    history=state.setdefault('config_history',[])
    history.append({'at':time.time(),'previous':state.get('config_snapshot'),'current':binding(snapshot),
        'changed_fields':sorted(changes),'policy':'active owner retains its actual old actor/runtime until natural boundary'})
    state.update(config=current,config_snapshot=binding(snapshot),config_value=config)
    if publish: write(folder/'state.json',state)
    return config

def _primary_overlap_settings(config):
    limit=config.get('primary_parallel_batches',1)
    if type(limit)is not int or limit not in (1,2):
        raise ValueError('primary_parallel_batches must be one or two')
    threshold=config.get('primary_tail_max_running',6);workers=config.get('overlap_workers',12)
    if type(threshold)is not int or not 1<=threshold<=6 or type(workers)is not int or not 1<=workers<=24:
        raise ValueError('primary overlap limits differ from the bounded policy')
    if limit==2:
        if not isinstance(config.get('collection_repo'),str)or not Path(config['collection_repo']).is_absolute():
            raise ValueError('overlap collection needs its explicit actual source repository')
        worker=config.get('collection_worker')
        if (not isinstance(worker,dict)or worker.get('path')!=config.get('queue_worker')
                or binding(worker['path'])!=worker):
            raise ValueError('overlap collection needs its exact configured queue-worker bytes')
        feeder=config.get('overlap_feeder')
        if not isinstance(feeder,dict)or binding(feeder['path'])!=feeder:
            raise ValueError('overlap feeder lacks an exact source binding')
    capacity=config.get('primary_worker_capacity')
    if capacity is not None and (limit!=2 or type(capacity)is not int or capacity<workers):
        raise ValueError('primary worker capacity needs an explicit positive overlap envelope')
    return limit,threshold,workers


def primary_ahead_slots(state):
    """One ordered custody inventory, including active tails and completed work."""
    result={};expected=state.get('active_batch')
    if expected is None and type(state.get('next_batch'))is int:expected=state['next_batch']-1
    for name in ('primary_completed_ahead','primary_active_ahead'):
        values=state.get(name,{})
        if not isinstance(values,dict):raise ValueError('guardian ahead slots are malformed')
        for key,slot in values.items():
            if (not isinstance(slot,dict)or type(slot.get('batch'))is not int
                    or key!=str(slot['batch'])or slot['batch']in result
                    or (name=='primary_completed_ahead'and not isinstance(slot.get('boundary'),dict))):
                raise ValueError('guardian ahead slot repeats a batch or loses its boundary')
            result[slot['batch']]=slot
    current=state.get('primary_prefetch')
    if current is not None:
        if not isinstance(current,dict)or type(current.get('batch'))is not int or current['batch']in result:
            raise ValueError('guardian live slot repeats an owned batch')
        result[current['batch']]=current
    for batch in sorted(result):
        if type(expected)is not int or batch!=expected+1:
            raise ValueError('guardian ahead slots have a gap')
        expected=batch
    return {batch:result[batch]for batch in sorted(result)}


def verify_preserved_primary_slot(root, folder, state, expected_slot, expected_started,
                                 expected_state_slot, *, allow_uncaptured_active=False):
    """Verify one handover batch against its live or sealed guardian custody.

    ``expected_slot`` and ``expected_started`` come from the immutable handover
    manifest.  This is read-only: it never adopts, archives, or relabels work.
    A dead process is accepted only when the same batch has its exact complete
    guardian boundary; a missing slot or unauthenticated boundary is an error.
    """
    root, folder = Path(root), Path(folder)
    if (not isinstance(expected_slot, dict) or type(expected_slot.get('batch')) is not int
            or not isinstance(expected_started, dict)
            or (expected_state_slot is not None and not isinstance(expected_state_slot, dict))
            or (expected_state_slot is None and not allow_uncaptured_active)):
        raise ValueError('preserved slot needs its exact manifest, state slot, and started receipt')
    batch = expected_slot['batch']; key = str(batch)
    expected_owner = expected_slot.get('owner')
    if isinstance(expected_owner, dict) and 'identity' in expected_owner:
        expected_owner = expected_owner['identity']
    identity_fields = ('pid', 'start_ticks', 'boot_id')
    if (not isinstance(expected_owner, dict)
            or any(expected_owner.get(name) is None for name in identity_fields)):
        raise ValueError('preserved slot lacks its original process identity')
    expected_sealed = expected_slot.get('sealed')
    names = ('source-cases.jsonl', 'queue.jsonl', 'teacher-source.json')
    if not isinstance(expected_sealed, dict) or set(expected_sealed) != set(names):
        raise ValueError('preserved slot lacks its exact source, queue, and teacher bindings')
    batch_root = root / 'evidence-collections-v2' / f'batch-{batch:04d}'
    for name in names:
        expected_path = batch_root / name
        if (expected_sealed[name].get('path') != str(expected_path)
                or binding(expected_path) != expected_sealed[name]):
            raise ValueError(f'preserved batch {name} differs from its original sealed bytes')

    sequence = state.get('source_sequence')
    if not isinstance(sequence, dict) or sequence.get('started', {}).get(key) != expected_started:
        raise ValueError('preserved slot lost its exact source-sequence start binding')
    started_path = folder / 'source-sequence' / 'started' / f'batch-{batch:04d}.json'
    if (expected_started.get('path') != str(started_path)
            or binding(started_path) != expected_started):
        raise ValueError('preserved slot started receipt changed or escaped its control directory')
    started = json.loads(started_path.read_bytes())
    if (started.get('batch') != batch or started.get('sealed') != expected_sealed
            or not isinstance(started.get('selection_intent'), dict)):
        raise ValueError('preserved slot start receipt disagrees with its original batch')
    intent_ref = started['selection_intent']
    if binding(intent_ref['path']) != intent_ref:
        raise ValueError('preserved slot selection intent changed')
    intent = json.loads(Path(intent_ref['path']).read_bytes())
    if intent.get('batch') != batch:
        raise ValueError('preserved slot selection intent names another batch')
    prefetch = intent.get('prefetch') or {}
    runtime = prefetch.get('task_runtime')
    if runtime is None:
        configuration_ref = intent.get('new_configuration')
        if not isinstance(configuration_ref, dict) or binding(configuration_ref['path']) != configuration_ref:
            raise ValueError('preserved slot has no authenticated runtime configuration epoch')
        runtime = json.loads(Path(configuration_ref['path']).read_bytes()).get('task_runtime')
    if not isinstance(runtime, str) or not runtime:
        raise ValueError('preserved slot runtime epoch is missing')
    source = json.loads((batch_root / 'teacher-source.json').read_bytes())
    if source.get('actual_task_runtime') != runtime:
        raise ValueError('preserved slot source was sealed under another task-runtime epoch')

    closed_ref = sequence.get('closed', {}).get(key)
    actual_slot = primary_ahead_slots(state).get(batch)
    if actual_slot is None:
        actual_slot = (state.get('primary_launches') or {}).get(key)
    if actual_slot is None and state.get('active_batch') == batch:
        actual_slot = {'batch': batch, 'owner': state.get('owner'),
            'feeder': state.get('feeder'), 'task_runtime': state.get('task_runtime')}
    if actual_slot is None and closed_ref is not None and not alive(expected_owner):
        # Natural source-sequence closure may archive a completed slot out of
        # both ahead maps and primary_launches. Its exact immutable handover
        # slot remains the expected owner record; the closed receipt and
        # boundary below must still match its original start and sealed bytes.
        actual_slot = expected_state_slot or {'batch': batch, 'owner': expected_owner}
    if not isinstance(actual_slot, dict) or actual_slot.get('batch') != batch:
        raise ValueError('preserved batch has no matching live, completed, or adopted guardian slot')
    if expected_state_slot is not None and actual_slot != expected_state_slot:
        raise ValueError('preserved guardian slot differs from its captured handover state')
    actual_owner = actual_slot.get('owner')
    if not isinstance(actual_owner, dict) or any(actual_owner.get(name) != expected_owner[name]
                                                   for name in identity_fields):
        raise ValueError('preserved batch owner identity changed')
    if isinstance(expected_slot.get('owner'), dict) and expected_slot['owner'].get('argv'):
        current = process(expected_owner['pid']) if alive(expected_owner) else None
        if current is not None and current['argv'] != [item.encode() if isinstance(item, str) else item
                                                        for item in expected_slot['owner']['argv']]:
            raise ValueError('preserved live batch feeder argv changed')
    if actual_slot.get('selection_intent') is not None and actual_slot['selection_intent'] != intent_ref:
        raise ValueError('preserved slot selection intent changed')
    if actual_slot.get('task_runtime') is not None and actual_slot['task_runtime'] != runtime:
        raise ValueError('preserved slot runtime epoch changed')
    source_configuration = actual_slot.get('source_configuration')
    if source_configuration is not None and intent.get('new_configuration') != source_configuration:
        raise ValueError('preserved slot source configuration epoch changed')

    live = alive(expected_owner)
    route = 'live-owner'
    boundary_ref = None
    completed = str(batch) in state.get('primary_completed_ahead', {})
    if not live:
        if completed:
            boundary_ref = actual_slot.get('boundary')
            route = 'authenticated-completed-ahead'
        elif closed_ref is not None:
            if binding(closed_ref['path']) != closed_ref:
                raise ValueError('preserved slot closed source receipt changed')
            closed = json.loads(Path(closed_ref['path']).read_bytes())
            if closed.get('batch') != batch or closed.get('started') != expected_started:
                raise ValueError('preserved slot closed receipt changed owner/source epoch')
            boundary_ref = closed.get('boundary')
            if (expected_state_slot is not None and expected_state_slot.get('boundary') is not None
                    and boundary_ref != expected_state_slot['boundary']):
                raise ValueError('preserved archived slot boundary differs from its captured owner record')
            route = 'authenticated-closed-boundary'
        else:
            raise ValueError('dead preserved slot has no authenticated completion boundary')
        expected_boundary = folder / 'boundaries' / f'batch-{batch:04d}.json'
        if (not isinstance(boundary_ref, dict) or boundary_ref.get('path') != str(expected_boundary)
                or binding(expected_boundary) != boundary_ref):
            raise ValueError('preserved slot boundary is missing or not its exact guardian receipt')
        boundary_value = json.loads(expected_boundary.read_bytes())
        if (boundary_value.get('batch') != batch or boundary_value.get('batch_complete') is not True
                or not isinstance(boundary_value.get('bindings'), dict)):
            raise ValueError('preserved slot boundary is incomplete or names another batch')
        for reference in boundary_value['bindings'].values():
            if binding(reference['path']) != reference:
                raise ValueError('preserved slot boundary dependency changed')
        for name in ('queue.jsonl', 'teacher-source.json'):
            if boundary_value['bindings'].get(name) != expected_sealed[name]:
                raise ValueError('preserved slot boundary does not bind its original queue/source')
    elif completed:
        raise ValueError('completed-ahead slot still has a live owner')
    return {'batch': batch, 'owner': expected_owner, 'sealed': expected_sealed,
        'task_runtime': runtime, 'custody_route': route, 'boundary': boundary_ref,
        'started': expected_started}


def verify_preserved_primary_slots(root, folder, state, handover):
    """Operator entry point for a manifest's complete preserved-slot set.

    New handover manifests capture the exact source-sequence cursor and every
    state representation which may own a batch. This function resolves each
    expected slot from those immutable snapshots and delegates all checking to
    ``verify_preserved_primary_slot``; it does not choose a transition.
    """
    if not isinstance(handover, dict) or 'slots' not in handover or 'bound_source_sequence' not in handover:
        raise ValueError('handover manifest lacks exact preserved-slot inventory and source sequence')
    # Accept both the deployed v3 manifest names and the descriptive names
    # used by newer manifests. Missing launch snapshots are empty because old
    # manifests captured active/completed/prefetch slots but not that map.
    def captured(primary, deployed, default=None):
        if primary in handover:
            value = handover[primary]
            if deployed in handover and handover[deployed] != value:
                raise ValueError('handover contains conflicting preserved-slot snapshots')
            return value
        return handover.get(deployed, default)

    expected_slots = handover['slots']; sequence = handover['bound_source_sequence']
    if not isinstance(expected_slots, list) or not isinstance(sequence, dict):
        raise ValueError('handover preserved-slot inventory is malformed')
    started = sequence.get('started')
    if not isinstance(started, dict):
        raise ValueError('handover manifest lacks original started-source receipts')
    state_maps = [captured('bound_primary_completed_ahead', 'bound_completed_ahead', {}),
                  captured('bound_primary_active_ahead', 'bound_active_ahead', {}),
                  captured('bound_primary_launches', 'bound_primary_launches', {})]
    prefetch = captured('bound_primary_prefetch', 'bound_prefetch')
    if not all(isinstance(value, dict) for value in state_maps):
        raise ValueError('handover slot snapshots are malformed')
    if prefetch is not None and not isinstance(prefetch, dict):
        raise ValueError('handover primary-prefetch snapshot is malformed')
    results = []; seen = set()
    for expected in expected_slots:
        if not isinstance(expected, dict) or type(expected.get('batch')) is not int:
            raise ValueError('handover slot manifest entry is malformed')
        batch = expected['batch']; key = str(batch)
        if batch in seen or key not in started:
            raise ValueError('handover slot repeats or lacks its original started receipt')
        seen.add(batch)
        candidates = [mapping[key] for mapping in state_maps if key in mapping]
        if prefetch is not None and prefetch.get('batch') == batch:
            candidates.append(prefetch)
        if not candidates and batch == handover.get('active_batch'):
            active_owner = handover.get('original_owner')
            expected_owner = expected.get('owner', {})
            if isinstance(expected_owner, dict) and 'identity' in expected_owner:
                expected_owner = expected_owner['identity']
            if not isinstance(active_owner, dict) or any(
                    active_owner.get(name) != expected_owner.get(name)
                    for name in ('pid', 'start_ticks', 'boot_id')):
                raise ValueError('handover active owner differs from its preserved slot identity')
            candidates = [None]
        if not candidates or any(candidate != candidates[0] for candidate in candidates[1:]):
            raise ValueError('handover manifest has no unique exact guardian slot snapshot')
        results.append(verify_preserved_primary_slot(root, folder, state,
            expected, started[key], candidates[0],
            allow_uncaptured_active=(candidates[0] is None and batch == handover.get('active_batch'))))
    return {'schema': 'natlang.teacher-preserved-slots/1', 'slots': results,
        'batches': sorted(seen), 'training_admission': False}


def _primary_work_frontier(root,batch):
    """Observe immutable queue custody and actual worker journal capacity."""
    home=Path(root)/'evidence-collections-v2'/f'batch-{batch:04d}';queue=home/'queue.jsonl'
    if not queue.is_file():return None
    entries=[json.loads(line)for line in queue.read_text().splitlines()if line]
    keys={row['key']for row in entries}
    if not entries or len(keys)!=len(entries):raise ValueError('tail queue has duplicate identities')
    journals=sorted(home.glob('journal*worker-*.jsonl'))
    if not journals:return None
    before={str(p):binding(p)for p in [queue,*journals]};started=set();finished={}
    for journal in journals:
        raw=journal.read_bytes()
        # Active writers can be sampled between fragments of one append. Do
        # not interpret a partial event as either completion or a fatal batch.
        if raw and not raw.endswith(b'\n'):return None
        for line in raw.split(b'\n'):
            if not line:continue
            row=json.loads(line)
            if row.get('event')not in ('start','finish'):continue
            if row.get('key')not in keys:raise ValueError('tail observation includes another batch key')
            if row['event']=='start':started.add(row['key'])
            else:
                if row['key']in finished and finished[row['key']]!=row:
                    raise ValueError('tail observation has conflicting finishes')
                finished[row['key']]=row
    if not set(finished)<=started:raise ValueError('tail finish lacks its actual start')
    if before!={str(p):binding(p)for p in [queue,*journals]}:return None
    remaining=keys-set(finished)
    active=started-set(finished)
    unstarted=keys-started
    # Static shard tails can leave pending keys behind a few live workers after
    # other workers exit. They retain those keys; the separate second batch
    # uses available serving capacity. Do not interpret startup as a tail.
    return {'schema':'natlang.teacher-primary-tail-capacity/2','batch':batch,
        'queue':before[str(queue)],'journals':[before[str(p)]for p in journals],
        'started_keys':sorted(started),'finished_keys':sorted(finished),
        'remaining_keys':sorted(remaining),'unstarted_keys':sorted(unstarted),
        'active_keys':sorted(active),
        'claim':'actual capacity observation only; no original completion or admission'}


def primary_tail_frontier(root,batch,threshold):
    value=_primary_work_frontier(root,batch)
    if (value is None or not value['finished_keys']or not value['remaining_keys']
            or len(value['active_keys'])>threshold):return None
    return value


PRIMARY_REFILL_POLICY = """Refill within the owned worker envelope.
Keep every original live batch and its unstarted keys with its pinned owner.
Observe completed work in an unfinished batch. Count every unfinished key, bounded
by that batch's pinned workers, including all live future slots. Reserve the
next qualified source only when its full workers fit the explicitly declared
envelope. Never grant capacity from unknown, torn or unsealed custody.
"""


def primary_refill_admission(observation,workers,capacity):
    """Named scheduling stage; crisp arithmetic verifies the declared NL policy."""
    if type(workers)is not int or workers<1 or type(capacity)is not int or capacity<workers:
        raise ValueError('refill requires declared positive worker capacity')
    if not observation or observation.get('schema')!='natlang.teacher-owned-worker-capacity/1':return None
    demand=0;tails=[];batches=set()
    for slot in observation['slots']:
        count=slot['workers'];frontier=slot['frontier']
        if type(count)is not int or count<1 or frontier is None:return None
        if (type(slot['batch'])is not int or slot['batch']in batches
                or frontier.get('batch')!=slot['batch']):raise ValueError('refill repeats or changes an owned batch')
        batches.add(slot['batch'])
        sets={name:set(frontier[name])for name in ('started_keys','finished_keys','remaining_keys','active_keys','unstarted_keys')}
        if (any(len(sets[name])!=len(frontier[name])for name in sets)
                or not sets['finished_keys']<=sets['started_keys']
                or sets['active_keys']!=sets['started_keys']-sets['finished_keys']
                or sets['unstarted_keys']&sets['started_keys']
                or sets['remaining_keys']!=sets['active_keys']|sets['unstarted_keys']):
            raise ValueError('refill frontier has inconsistent work partitions')
        remaining=frontier['remaining_keys']
        demand+=min(count,len(remaining))
        # In explicit-envelope mode the exact unfinished-worker sum is the
        # admission bound. The legacy low-water mark must not add idle time
        # after useful capacity has already become available.
        if frontier['finished_keys']and remaining:tails.append(slot['batch'])
    if not tails or demand+workers>capacity:return None
    return {'schema':'natlang.teacher-owned-worker-refill/1','policy':'refill-within-owned-worker-envelope',
        'potential_workers':demand,'new_workers':workers,'worker_capacity':capacity,
        'eligible_tail_batches':tails,'observation':observation,'training_admission':False}


def primary_capacity_observation(folder,state,config):
    """Authenticate exact owner/source/worker bounds before giving capacity credit."""
    sequence=state.get('source_sequence')or{};declaration=config['source_sequence']
    active=state.get('active_batch')
    if type(active)is not int or not alive(state.get('owner')):return None
    ahead=primary_ahead_slots(state);completed=state.get('primary_completed_ahead',{})
    primary=(state.get('primary_launches')or{}).get(str(active),{})
    slots=[dict(primary,batch=active,owner=state['owner'],feeder=primary.get('feeder',state.get('feeder',config['feeder'])))]+[
        slot for batch,slot in ahead.items()if str(batch)not in completed]
    observed=[]
    for slot in slots:
        owner=slot.get('owner');current=process(owner['pid'])if owner else None
        if not current or not alive(owner):return None
        feeder=slot['feeder'];path=feeder['path']if isinstance(feeder,dict)else feeder
        if isinstance(feeder,dict)and binding(path)!=feeder:raise ValueError('owned refill feeder changed')
        # Popen can publish the same owner's identity before nice execs Python.
        # Accept only our exact launcher while it performs that normal exec;
        # source sealing below must still establish actual collection custody.
        if not _feeder_invocation(current,path):
            raise ValueError('refill owner no longer executes its exact feeder')
        batch=slot['batch'];ref=sequence.get('started',{}).get(str(batch))
        if ref is None:return None
        started=_source_sequence_helpers()._json(ref,declaration)
        home=Path(config['root'])/'evidence-collections-v2'/f'batch-{batch:04d}'
        sealed=started['sealed']
        if (started.get('batch')!=batch or any(sealed[name]!=binding(home/name)
                for name in('source-cases.jsonl','queue.jsonl','teacher-source.json'))):
            raise ValueError('refill source differs from actual sealed custody')
        source=_source_sequence_helpers()._json(sealed['teacher-source.json'],declaration)
        workers=source.get('concurrency')
        if type(workers)is not int or workers<1:raise ValueError('refill lacks actual pinned source workers')
        if slot.get('workers',workers)!=workers:raise ValueError('refill changed pinned batch workers')
        frontier=_primary_work_frontier(config['root'],batch)
        if frontier is None:return None
        if frontier['queue']!=sealed['queue.jsonl']:raise ValueError('refill journal queue differs from source')
        observed.append({'batch':batch,'workers':workers,'owner':owner,
            'owner_argv_sha256':hashlib.sha256(b'\0'.join(current['argv'])).hexdigest(),
            'started':ref,'frontier':frontier})
    return {'schema':'natlang.teacher-owned-worker-capacity/1','slots':observed,
        'tail_max_running':config.get('primary_tail_max_running',6)}


def _capacity_still_current(observation):
    for slot in observation['slots']:
        owner=slot['owner'];current=process(owner['pid'])
        if (not current or not alive(owner)
                or hashlib.sha256(b'\0'.join(current['argv'])).hexdigest()!=slot['owner_argv_sha256']):return False
        frontier=slot['frontier']
        if any(binding(ref['path'])!=ref for ref in [frontier['queue'],*frontier['journals']]):return False
    return True


def _spawn_primary_prefetch(folder,state,config,slot,workers):
    recovered=recover_token(slot['token'],script_prefix=Path(slot['feeder']['path']).name)
    if recovered:
        slot['owner']=public_identity(recovered);write(Path(folder)/'state.json',state);return
    if alive(slot.get('owner')):return
    home=Path(config['root'])/'evidence-collections-v2'/f"batch-{slot['batch']:04d}"
    if home.exists():return  # A materialized orphan requires reconciliation, never a duplicate spawn.
    if config.get('drain_file')and Path(config['drain_file']).exists():return
    if binding(slot['feeder']['path'])!=slot['feeder']:
        raise ValueError('reserved feeder mechanism changed')
    for item in Path('/proc').iterdir():
        if not item.name.isdigit():continue
        current=process(item.name)
        if not current:continue
        is_feeder=any(Path(arg.decode(errors='replace')).name.startswith('feed-evidence-')
            or arg.decode(errors='replace')==slot['feeder']['path']for arg in current['argv'][:5])
        owners=[state.get('owner'),*[item.get('owner')for item in primary_ahead_slots(state).values()]]
        if is_feeder and public_identity(current)not in owners:
            raise ValueError('unowned feeder prevents next-batch overlap')
    if binding(slot['source_configuration']['path'])!=slot['source_configuration']:
        raise ValueError('reserved immutable source configuration changed')
    context=slot['collection_context']
    if binding(context['queue_worker']['path'])!=context['queue_worker']:
        raise ValueError('reserved collection queue worker changed')
    env={**os.environ,TOKEN:slot['token'],'NATLANG_TEACHER_START_BATCH':str(slot['batch']),
        'NATLANG_TEACHER_SINGLE_BATCH':'1','NATLANG_TEACHER_TASK_RUNTIME':slot['task_runtime'],
        'NATLANG_TEACHER_GUARDIAN_MODULE_ROOT':str(Path(__file__).parent),
        'NATLANG_TEACHER_GENERATION_ROOT':config['root'],'NATLANG_REPO':context['repo'],
        'NATLANG_TEACHER_QUEUE_WORKER':context['queue_worker']['path'],'NATLANG_TEACHER_NODE_BIN':context['node_bin'],
        'NATLANG_TEACHER_IDENTITY':config['teacher_identity']['path'],
        'NATLANG_TEACHER_SOURCE_CONFIGURATION':slot['source_configuration']['path'],
        'NATLANG_TEACHER_SOURCE_CONFIGURATION_SHA256':slot['source_configuration']['sha256'],
        'NATLANG_TEACHER_SOURCE_CONFIGURATION_BYTES':str(slot['source_configuration']['bytes']),
        'NATLANG_TEACHER_WORKER_COUNT':str(workers)}
    env.update(teacher_environment(config))
    verify_teacher_identity(config['teacher_identity']['path'],server=config['server'],model_id=config['model_id'])
    log=(Path(folder)/f"prefetch-{slot['batch']:04d}.log").open('ab')
    child=subprocess.Popen(['nice','-n','10','python','-u',slot['feeder']['path']],
        env=env,stdout=log,stderr=log,start_new_session=True)
    identity=process(child.pid)
    if identity is None:raise ValueError('reserved feeder exited before custody publication')
    slot['owner']=public_identity(identity);write(Path(folder)/'state.json',state)


def maintain_primary_prefetch(folder,state,config):
    limit,threshold,workers=_primary_overlap_settings(config);capacity=config.get('primary_worker_capacity')
    if limit==1:
        if primary_ahead_slots(state):raise ValueError('cannot disable overlap with an unreconciled next batch')
        return
    if state.get('primary_active_ahead')and capacity is None:
        raise ValueError('active future tails require their declared worker envelope')
    completed=state.get('primary_completed_ahead',{})
    for batch,slot in list(primary_ahead_slots(state).items()):
        if str(batch)in completed:continue
        _spawn_primary_prefetch(folder,state,config,slot,slot['workers'])
        _source_sequence_helpers().observe_prefetched_source(folder,state,config)
        phase=Path(slot['phase_path'])
        if not alive(slot.get('owner'))and phase.is_file():
            value=json.loads(phase.read_text())
            if value.get('phase')=='batch_complete':
                decision,entries=boundary(Path(config['root']),Path(folder),batch,phase_path=phase)
                prepare_retry(Path(folder),decision,entries)
                slot['boundary']=binding(Path(folder)/'boundaries'/f'batch-{batch:04d}.json')
                write(Path(folder)/'state.json',state)
                _source_sequence_helpers().archive_completed_prefetch(folder,state,config,dict(slot))
    if not alive(state.get('owner'))or Path(config['drain_file']).exists():return
    current=state.get('primary_prefetch');admission=None
    if capacity is None:
        if current:return
        tail=primary_tail_frontier(config['root'],state['active_batch'],threshold)
        if tail is None:return
    else:
        # An interrupted reservation may precede the guardian slot write.
        # Its pinned context survives a later configuration revision.
        reserved=(state.get('source_sequence')or{}).get('prefetch')
        if reserved and not current:
            prior=_source_sequence_helpers()._json(reserved['intent'],config['source_sequence'])
            workers=(prior.get('prefetch',{}).get('resource_refill')or{}).get('workers',workers)
        observation=primary_capacity_observation(folder,state,config)
        admission=primary_refill_admission(observation,workers,capacity)
        if admission is None or not _capacity_still_current(observation):return
        tail=next(item['frontier']for item in observation['slots']if item['batch']in admission['eligible_tail_batches'])
        if current:_source_sequence_helpers().park_active_prefetch(folder,state,config,dict(current))
        if not alive(state.get('owner')):return
    ahead=primary_ahead_slots(state)
    batch=max(ahead,default=state['active_batch'])+1
    pending=_source_sequence_helpers().prepare_prefetched_source(folder,state,config,batch)
    if pending is None:return
    declaration=config['source_sequence'];intent=_source_sequence_helpers()._json(pending['intent'],declaration)
    preserved=intent['prefetch'].get('resource_refill')or{}
    slot={'batch':batch,'token':uuid.uuid4().hex,'owner':None,
        'feeder':preserved.get('feeder',config['overlap_feeder']),
        'guardian_config':preserved.get('guardian_config',state['config_snapshot']),
        'collection_context':preserved.get('collection_context',{'repo':config['collection_repo'],
            'queue_worker':config['collection_worker'],'node_bin':config['node_bin']}),
        'task_runtime':intent['prefetch']['task_runtime'],'workers':preserved.get('workers',workers),
        'source_configuration':intent['new_configuration'],'selection_intent':pending['intent'],
        'tail_observation':tail,'predecessor_owner':state['owner'],
        'phase_path':str(Path(config['root'])/'feeder-phases'/f'batch-{batch:04d}.json')}
    if admission is not None:
        if slot['workers']!=admission['new_workers']:raise ValueError('reserved workers differ from capacity admission')
        raw=(json.dumps(admission,sort_keys=True,indent=2)+'\n').encode()
        receipt=Path(folder)/'worker-capacity'/f"batch-{batch:04d}-{hashlib.sha256(raw).hexdigest()}.json"
        _source_sequence_helpers()._immutable(receipt,raw);slot['capacity_admission']=binding(receipt)
    state['primary_prefetch']=slot;write(Path(folder)/'state.json',state)
    _spawn_primary_prefetch(folder,state,config,slot,slot['workers'])


def adopt_primary_prefetch(folder,state,config):
    completed=state.get('primary_completed_ahead',{})
    slot=primary_ahead_slots(state).get(state['next_batch'])
    if not slot:return False
    if slot['batch']!=state['next_batch']or state.get('owner')is not None:
        raise ValueError('prefetch adoption differs from actual primary closure')
    # A crash can follow durable reservation or Popen but precede the owner
    # write, including while the predecessor exits. Recover the same token
    # before considering another launch or publishing the next active owner.
    _spawn_primary_prefetch(folder,state,config,slot,slot['workers'])
    home=Path(config['root'])/'evidence-collections-v2'/f"batch-{slot['batch']:04d}"
    if not alive(slot.get('owner'))and not home.exists():return False
    _source_sequence_helpers().adopt_prefetched_source(folder,state,config,slot['batch'])
    state.setdefault('primary_launches',{})[str(slot['batch'])]=dict(slot)
    state.update(owner=slot.get('owner'),active_batch=slot['batch'],feeder=slot['feeder'],task_runtime=slot['task_runtime'])
    if str(slot['batch'])in completed:completed.pop(str(slot['batch']))
    elif str(slot['batch'])in state.get('primary_active_ahead',{}):state['primary_active_ahead'].pop(str(slot['batch']))
    else:state.pop('primary_prefetch')
    write(Path(folder)/'state.json',state)
    return True


def run(config_path, *, external_transition=None, external_reconciliation=None, drained_fanout_transition=None):
    config_path = Path(config_path); config = json.loads(config_path.read_text()); folder = Path(config['control']); folder.mkdir(parents=True, exist_ok=True)
    lock = (folder/'guardian.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if not config.get('teacher_identity'):raise ValueError('production guardian needs explicit actual teacher execution identity')
    root = Path(config['root']); target = folder/'state.json'
    if target.exists(): state = json.loads(target.read_text())
    else:
        predecessor = process(config['predecessor_pid'])
        if predecessor and config['predecessor_script'].encode() not in predecessor['argv']: raise ValueError('predecessor exact argv mismatch')
        state = {'schema': 'natlang.teacher-generation-guardian/1', 'config': binding(config_path),
                 'owner': public_identity(predecessor) if predecessor else None, 'active_batch': config['predecessor_batch']}
        write(target, state)
    if bool(external_transition) != bool(external_reconciliation):
        raise ValueError('external transition requires its authenticated reconciliation')
    if drained_fanout_transition is not None:
        if external_transition is not None:
            raise ValueError('choose one exact owner transition')
        state = adopt_drained_fanout(folder,drained_fanout_transition)
    if external_transition is not None:
        try:
            from .teacher_external_reconciliation import adopt_external_transition
        except ImportError:
            from teacher_external_reconciliation import adopt_external_transition
        state = adopt_external_transition(folder, config_path, external_transition, external_reconciliation)
    # Preserve older state receipts while introducing explicit same-control revisions.
    if 'config_value' not in state and state.get('config')==binding(config_path):
        state.pop('config',None)
    config=config_revision(folder,state,config_path)
    while True:
        config=config_revision(folder,state,config_path)
        if record_primary_phase(state,config):write(target,state)
        if config.get('source_sequence'):
            _source_sequence_helpers().observe_source_sequence(folder,state,config)
        if alive(state.get('owner')):
            maintain_primary_prefetch(folder,state,config)
            if not Path(config['drain_file']).exists(): launch_repair(folder,state,config)
            write(folder/'phase.json', {'at': time.time(), 'phase': 'waiting_natural_feeder_exit', 'owner': state['owner'], 'drain_requested': Path(config['drain_file']).exists()})
            time.sleep(5); continue
        if state.get('active_batch') is not None:
            launch=(state.get('primary_launches')or{}).get(str(state['active_batch']),{})
            decision, entries = boundary(root, folder, state['active_batch'],phase_path=launch.get('phase_path')); prepare_retry(folder, decision, entries)
            state.update(next_batch=decision['batch']+1, active_batch=None, owner=None,
                last_boundary=binding(folder/'boundaries'/f"batch-{decision['batch']:04d}.json"))
            write(target, state)
            if config.get('source_sequence'):_source_sequence_helpers().observe_source_sequence(folder,state,config)
            if adopt_primary_prefetch(folder,state,config):continue
        if Path(config['drain_file']).exists():
            write(folder/'phase.json', {'at': time.time(), 'phase': 'operator_drained', 'repair_running': alive(state.get('repair_owner')), 'next_batch': state['next_batch']})
            time.sleep(5); continue
        launch_repair(folder, state, config)
        if adopt_primary_prefetch(folder,state,config):continue
        if primary_ahead_slots(state):
            raise ValueError('reserved next primary has not acquired an actual feeder owner')
        launch_feeder(folder, state, config, state['next_batch'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--config', required=True)
    parser.add_argument('--external-transition'); parser.add_argument('--external-reconciliation')
    parser.add_argument('--drained-fanout-transition')
    args = parser.parse_args()
    try: run(args.config, external_transition=args.external_transition, external_reconciliation=args.external_reconciliation,
             drained_fanout_transition=args.drained_fanout_transition)
    except BaseException as error:
        try:
            config = json.loads(Path(args.config).read_text()); write(Path(config['control'])/'error.json',
                {'at': time.time(), 'error': str(error), 'type': type(error).__name__})
        finally: raise
