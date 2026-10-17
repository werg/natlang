"""Durable boundary supervision for the existing self-hosted teacher feeder."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
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

def boundary(root, folder, expected_batch=None):
    phase = json.loads((root/'feed-evidence-phase.json').read_text())
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
    if config.get('source_sequence'):
        _source_sequence_helpers().verify_prepared_source(state,config,batch)
    log = (folder/'feeder.log').open('ab')
    child = subprocess.Popen(['nice', '-n', '10', 'python', '-u', config['feeder']],
        env=env, stdout=log, stderr=log, start_new_session=True)
    identity = process(child.pid)
    if identity is None: raise ValueError('new feeder exited before process identity publication')
    state.update(owner=public_identity(identity), active_batch=batch, feeder=binding(config['feeder']),
        task_runtime=config['task_runtime']); state.pop('prepared_launch', None); write(target, state)

def repair_outcomes(queue, journal):
    entries = [json.loads(line) for line in queue.read_text().splitlines() if line]
    keys = {entry['key'] for entry in entries}
    rows = [json.loads(line) for line in journal.read_text().splitlines() if line] if journal.exists() else []
    if any(row.get('key') not in keys for row in rows): raise ValueError('repair journal has an unknown queue key')
    finishes = {row['key']: row for row in rows if row.get('event') == 'finish'}
    return entries, finishes


def launch_repair(folder, state, config):
    """Recover one shared process-token protocol, then retry only unfinished keys."""
    active = state.get('repair_owner')
    if active and alive(active): return
    for source_queue in sorted((folder/'repairs').glob('batch-*/queue.jsonl')):
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
                    [('runtime','task_runtime'),('server','server'),('model_id','model_id'),('teacher_identity','teacher_identity')]):
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
            number = len(receipt['attempts'])+1; leaf = source_queue.parent/f'attempt-{number:04d}'
            leaf.mkdir(exist_ok=True)
            rows = [{**row, 'jobs': str(leaf/row['key']/'jobs'),
                'output': str(leaf/row['key']/'results.jsonl'), 'log': str(leaf/row['key']/'collector.log')} for row in remaining]
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
            receipt['current'] = attempt; write(receipt_path, receipt)
        queue = Path(attempt['queue'])
        if binding(queue) != attempt['queue_binding']: raise ValueError('prepared repair queue changed')
        env = {**os.environ, TOKEN:attempt['token'], 'PATH':attempt['node_bin']+':'+os.environ.get('PATH',''),
               'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1'}
        command=['flock','-n',str(folder/'repair-collector.lock'),'nice','-n','19','python','-u',attempt['queue_worker'],
            str(queue),attempt['journal'],'--runtime',attempt['runtime'],'--server',attempt['server'],
            '--chat-completions-url',attempt['server']+'/v1/chat/completions','--model-id',attempt['model_id'],
            '--model-concurrency','1','--max-batch-cases','1','--case-seconds','7200','--identical-turn-limit','8']
        if attempt.get('teacher_identity'):
            identity_binding=attempt['teacher_identity']
            if binding(identity_binding['path'])['sha256']!=identity_binding['sha256']:
                raise ValueError('configured teacher execution identity changed')
            attempt['actual_teacher_identity']=verify_teacher_identity(identity_binding['path'],server=attempt['server'],model_id=attempt['model_id'])
            command += ['--teacher-identity',identity_binding['path']]
        attempt['argv']=command;write(receipt_path,receipt)
        log=(queue.parent/'worker.log').open('ab');child=subprocess.Popen(command,env=env,stdout=log,stderr=log,start_new_session=True)
        identity=process(child.pid)
        if identity:
            attempt.update(state='launched',owner=public_identity(identity));state['repair_owner']=attempt['owner']
        write(receipt_path,receipt);write(folder/'state.json',state);return


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
    allowed={'task_runtime','server','model_id','feeder','queue_worker','node_bin','drain_file','teacher_identity','source_sequence'}
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
        if config.get('source_sequence'):
            _source_sequence_helpers().observe_source_sequence(folder,state,config)
        if alive(state.get('owner')):
            if not Path(config['drain_file']).exists(): launch_repair(folder,state,config)
            write(folder/'phase.json', {'at': time.time(), 'phase': 'waiting_natural_feeder_exit', 'owner': state['owner'], 'drain_requested': Path(config['drain_file']).exists()})
            time.sleep(5); continue
        if state.get('active_batch') is not None:
            decision, entries = boundary(root, folder, state['active_batch']); prepare_retry(folder, decision, entries)
            state.update(next_batch=decision['batch']+1, active_batch=None, owner=None,
                last_boundary=binding(folder/'boundaries'/f"batch-{decision['batch']:04d}.json"))
            write(target, state)
        if Path(config['drain_file']).exists():
            write(folder/'phase.json', {'at': time.time(), 'phase': 'operator_drained', 'repair_running': alive(state.get('repair_owner')), 'next_batch': state['next_batch']})
            time.sleep(5); continue
        launch_repair(folder, state, config)
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
