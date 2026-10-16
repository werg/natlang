"""Exact external continuation accounting; never training admission or original completion."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

try:
    from .teacher_batch_completion import batch_completion
except ImportError:
    from teacher_batch_completion import batch_completion


def reconcile_snapshot(index_path, *, original_source, original_queue, continuations):
    # Reuse the authoritative admission verifier's provenance mechanism rather
    # than allowing scheduling to invent a weaker actor/parent identity join.
    from scripts.neuralese_data.build_verified_teacher_trajectory_graphs import collection_provenance
    from natlang_neuralese.common.hashing import sha256_file_hex, canonical_json_sha256_hex
    index_path = Path(index_path)
    index = json.loads(index_path.read_text())
    files = {}
    transferred = {}
    for item in index['files']:
        path = (index_path.parent / item['path']).resolve()
        if (not path.is_relative_to(index_path.parent.resolve())
                or path.stat().st_size != item['bytes'] or sha256_file_hex(path) != item['sha256']):
            raise ValueError('teacher accounting snapshot changed its exact file bytes')
        if item['source'] in files:
            raise ValueError('teacher accounting snapshot repeats a source path')
        files[item['source']] = path
        transferred[item['source']] = [{**item, 'path': str(path)}]
    def read(source):
        if source not in files:
            raise ValueError('teacher accounting snapshot lacks exact source ' + source)
        return json.loads(files[source].read_text())
    def rows(source):
        if source not in files:
            raise ValueError('teacher accounting snapshot lacks exact queue/journal ' + source)
        return [json.loads(line) for line in files[source].read_text().split('\n') if line.strip()]
    def resolve(source):
        if str(source) in files:
            return files[str(source)]
        # Directories are not individual file inventory entries. Resolve them
        # only through the common transferred tree, never into the live host.
        candidate = (index_path.parent / 'files' / str(source).lstrip('/')).resolve()
        if (not candidate.is_relative_to((index_path.parent / 'files').resolve())
                or not candidate.is_dir()):
            raise ValueError('teacher accounting file is absent from the authenticated inventory')
        return candidate
    original = read(original_source)
    originals = rows(original_queue)
    original_by_key = {entry['key']: entry for entry in originals}
    if not originals or len(original_by_key) != len(originals):
        raise ValueError('original teacher queue requires distinct exact keys')
    terminal = {}
    packet_values = {}
    collections = []
    for source, queue in [(original_source, original_queue), *continuations]:
        collection = read(source)
        proofs = collection_provenance(collection, transferred)
        actual = {**collection, **collection.get('continuation', {})}
        if source != original_source:
            reference = actual.get('original_source') or {}
            original_item = transferred[original_source][0]
            if (reference.get('path') != original_source or reference.get('sha256') != original_item['sha256']
                    or reference.get('bytes') != original_item['bytes']):
                raise ValueError('teacher retry is not a continuation of this original collection')
            original_queue_ref = actual.get('original_queue') or {}
            queue_item = transferred[original_queue][0]
            if (original_queue_ref.get('path') != original_queue or original_queue_ref.get('sha256') != queue_item['sha256']
                    or original_queue_ref.get('bytes') != queue_item['bytes']):
                raise ValueError('teacher retry changed its exact original queue')
        selected = collection.get('source_cases')
        if selected not in files or sha256_file_hex(files[selected]) != collection['source_cases_sha256']:
            raise ValueError('teacher selected packet differs from its original bytes')
        if selected not in packet_values:
            packet_values[selected] = canonical_json_sha256_hex(rows(selected))
        entries = rows(queue)
        if len({entry['key'] for entry in entries}) != len(entries):
            raise ValueError('teacher retry queue repeats an original key')
        for entry in entries:
            old = original_by_key.get(entry['key'])
            comparable = lambda value: {k: v for k, v in value.items() if k not in {'jobs', 'output', 'log'}}
            if old is None or comparable(entry) != comparable(old):
                raise ValueError('teacher retry changed an original key/source/seed/execution input')
            if entry['source'] not in packet_values:
                packet_values[entry['source']] = canonical_json_sha256_hex(rows(entry['source']))
            if packet_values[entry['source']] != packet_values[selected]:
                raise ValueError('teacher queue source differs from selected collection packet')
        directory = str(Path(queue).parent)
        journals = sorted(name for name in files if str(Path(name).parent) == directory
                          and Path(name).name.startswith('journal') and name.endswith('.jsonl'))
        if not journals:
            raise ValueError('teacher collection has no actual accounting journal')
        events = [event for journal in journals for event in rows(journal)]
        keys = {entry['key'] for entry in entries}
        starts = {}
        finishes = {}
        for event in events:
            if event.get('event') not in {'start', 'finish'}:
                continue
            key = event.get('key')
            if key not in keys:
                raise ValueError('teacher journal refers to an unknown queue key')
            if event['event'] == 'start':
                starts[key] = event
            else:
                if key in finishes:
                    raise ValueError('teacher collection repeats a terminal key across journals')
                finishes[key] = event
        if source != original_source:
            for key in finishes:
                start = starts.get(key, {})
                identity = start.get('teacher_identity') or {}
                if (identity.get('identity') != actual['actual_teacher_execution_identity']
                        or identity.get('actor') != actual['actor']):
                    raise ValueError('teacher retry finish lacks its actual collection actor start')
        checked = batch_completion(entries, events, [], {}, resolve_path=resolve)
        unresolved = {entry['key'] for entry in checked['repair_required'] if entry['key'] is not None}
        for key in keys - unresolved:
            terminal.setdefault(key, []).append({'collection_source': source, 'queue': queue,
                                                 'finish': finishes[key]})
        collections.append({'source': source, 'queue': queue, 'journals': journals,
                            'continuation_proofs': proofs, 'completion': checked})
    return {'schema': 'natlang.teacher-external-terminal-reconciliation/1',
            'snapshot': {'path': str(index_path), 'sha256': hashlib.sha256(index_path.read_bytes()).hexdigest(),
                         'bytes': index_path.stat().st_size},
            'original_source': original_source, 'original_queue': original_queue,
            'expected_keys': len(originals), 'terminal_keys': terminal,
            'unresolved_keys': sorted(set(original_by_key) - set(terminal)), 'collections': collections,
            'original_batch_complete': False, 'training_admission': False,
            'scope': 'scheduling terminal join only; actual failures retained, original batch is not relabelled'}


def adopt_external_transition(folder, config_path, transition_path, reconciliation_path):
    """Adopt an already-running successor, preserving exact unresolved work.

    Call while holding the existing guardian.lock. Failure before final atomic
    state publication leaves a replayable boundary/repair preparation and the
    original state. This never starts another feeder.
    """
    try:
        from . import teacher_generation_guardian as guardian
    except ImportError:
        import teacher_generation_guardian as guardian
    folder, config_path = Path(folder), Path(config_path)
    transition_path, reconciliation_path = Path(transition_path), Path(reconciliation_path)
    transition = json.loads(transition_path.read_text())
    evidence = json.loads(reconciliation_path.read_text())
    def bound(reference):
        if not isinstance(reference, dict):
            raise ValueError('external transition lacks an immutable file binding')
        actual = guardian.binding(reference['path'])
        if any(actual.get(key) != reference.get(key) for key in ('sha256', 'bytes')):
            raise ValueError('external transition file bytes changed')
        return json.loads(Path(reference['path']).read_text())
    if transition.get('schema') != 'natlang.teacher-generation-external-transition/1':
        raise ValueError('external transition schema differs')
    if transition.get('original_batch_complete') is not False:
        raise ValueError('external transition cannot relabel original batch completion')
    state_path = folder/'state.json'
    state = json.loads(state_path.read_text())
    transition_binding = guardian.binding(transition_path)
    prior = state.get('external_transitions', [])
    if any(item.get('transition') == transition_binding for item in prior):
        return state  # Durable completed adoption, never resurrect a later owner.
    original = bound(transition.get('original_state'))
    if state != original or state.get('active_batch') != transition.get('original_batch'):
        raise ValueError('external transition does not match actual original guardian state')
    if guardian.alive(state.get('owner')):
        raise ValueError('original feeder still owns the guardian state')
    config = bound(transition.get('config'))
    if guardian.binding(config_path) != transition['config'] or config['control'] != str(folder):
        raise ValueError('external transition configuration differs from guardian control')
    if (transition.get('batch') != transition['original_batch'] + 1
            or config['feeder'] != transition['feeder']['path']
            or config['task_runtime'] != transition.get('task_runtime')
            or config.get('teacher_identity') != transition.get('teacher_identity')):
        raise ValueError('external transition changed successor batch/actor/runtime')
    # The feeder is executable source rather than JSON.
    actual_feeder = guardian.binding(transition['feeder']['path'])
    if actual_feeder != transition['feeder']:
        raise ValueError('external successor feeder bytes changed')
    bound(transition.get('source_authority'))
    exits = bound(transition.get('completed_continuation'))
    if (not isinstance(exits.get('returncodes'), list) or not exits['returncodes']
            or any(type(code) is not int for code in exits['returncodes'])):
        raise ValueError('external transition lacks actual drained continuation exits')
    if not isinstance(transition.get('process_token'), str) or not transition['process_token']:
        raise ValueError('external successor lacks its exact process token')
    owner = guardian.recover_token(transition['process_token'])
    if (owner is None or guardian.public_identity(owner) != transition.get('owner')
            or str(config['feeder']).encode() not in owner['argv']):
        raise ValueError('external successor is not the exact actual live feeder/token')
    bound(transition.get('teacher_identity'))
    guardian.verify_teacher_identity(config['teacher_identity']['path'],
                                     server=config['server'], model_id=config['model_id'])
    if (evidence.get('schema') != 'natlang.teacher-external-terminal-reconciliation/1'
            or evidence.get('original_batch_complete') is not False or evidence.get('training_admission') is not False):
        raise ValueError('external scheduling evidence has an invalid completion/admission claim')
    snapshot = evidence['snapshot']
    bound_snapshot = guardian.binding(snapshot['path'])
    if any(bound_snapshot.get(key) != snapshot.get(key) for key in ('sha256', 'bytes')):
        raise ValueError('external accounting snapshot index changed')
    checked = reconcile_snapshot(snapshot['path'], original_source=evidence['original_source'],
        original_queue=evidence['original_queue'],
        continuations=[(entry['source'], entry['queue']) for entry in evidence['collections'][1:]])
    for key in ('terminal_keys', 'unresolved_keys', 'expected_keys', 'original_source', 'original_queue'):
        if checked[key] != evidence[key]:
            raise ValueError('external scheduling evidence differs from actual authenticated terminal accounting')
    expected_root = Path(config['root'])/'evidence-collections-v2'/f"batch-{transition['original_batch']:04d}"
    if (Path(evidence['original_source']) != expected_root/'teacher-source.json'
            or Path(evidence['original_queue']) != expected_root/'queue.jsonl'):
        raise ValueError('external accounting evidence does not belong to original active batch')
    for entry in evidence['collections']:
        if guardian.live_batch_workers(Path(entry['queue']).parent):
            raise ValueError('external accounting collection still has live workers; preserve their ownership')
    # Preserve the exact interrupted boundary. The terminal join is a distinct
    # scheduling observation, never a fabricated original score or returncode.
    decision = {'schema': 'natlang.teacher-generation-boundary/1',
        'batch': transition['original_batch'], 'batch_complete': False,
        'completion': {'training_admission': False, 'scope': 'original interrupted; independent continuation join'},
        'external_reconciliation': guardian.binding(reconciliation_path),
        'repair_required': [{'key': key, 'reason': 'unresolved after authenticated external continuations'}
                            for key in evidence['unresolved_keys']],
        'individual_outcomes': []}
    boundary = folder/'boundaries'/f"batch-{decision['batch']:04d}.json"
    if boundary.exists() and json.loads(boundary.read_text()) != decision:
        raise ValueError('external adoption would overwrite a different immutable boundary')
    if not boundary.exists(): guardian.write(boundary, decision)
    snapshot_index = json.loads(Path(snapshot['path']).read_text())
    original_queue_item = next(item for item in snapshot_index['files'] if item['source'] == evidence['original_queue'])
    current_queue = guardian.binding(evidence['original_queue'])
    if any(current_queue[key] != original_queue_item[key] for key in ('sha256', 'bytes')):
        raise ValueError('live original queue differs from the authenticated snapshot')
    entries = [json.loads(line) for line in Path(evidence['original_queue']).read_text().split('\n') if line]
    source_inventory = {item['source']: item for item in snapshot_index['files']}
    for source in {entry['source'] for entry in entries}:
        actual_source = guardian.binding(source)
        if any(actual_source[key] != source_inventory[source][key] for key in ('sha256', 'bytes')):
            raise ValueError('live original packet differs from the authenticated snapshot')
    guardian.prepare_retry(folder, decision, entries)
    # Prepare config history without exposing an intermediate state that could
    # neither repeat this adoption nor safely own its already-running successor.
    guardian.config_revision(folder, state, config_path, publish=False)
    state.update(owner=transition['owner'], active_batch=transition['batch'],
        next_batch=transition['batch'] + 1, last_boundary=guardian.binding(boundary))
    state.setdefault('external_transitions', []).append({'transition': transition_binding,
        'reconciliation': guardian.binding(reconciliation_path), 'original_batch': transition['original_batch'],
        'unresolved_keys': evidence['unresolved_keys'], 'scope': 'actual successor adopted; original interrupted'})
    guardian.write(state_path, state)
    return state
