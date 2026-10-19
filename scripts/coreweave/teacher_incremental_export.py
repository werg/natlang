"""Immutable, resumable owner-host export tranches for live teacher batches.

The sandbox owns generation and raw result files.  This module only selects
terminal outputs, packages their exact bytes with source/control bindings, and
validates the receipt emitted by the existing private run-file exporter.
It contains no credentials and grants no training admission.
"""
from __future__ import annotations

import hashlib
import json
import os
import fcntl
from pathlib import Path
import re
import subprocess
import sys
import time
import tempfile
from typing import Callable

try:
    from . import source_archive
except ImportError:  # direct execution from a staged scripts/coreweave directory
    import source_archive


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _cwctl_module():
    try:
        from . import cwctl
    except ImportError:  # direct invocation from this scripts/coreweave directory
        import cwctl
    return cwctl


def _external_data_root() -> Path:
    """Resolve the machine's configured external-data root from shared paths."""
    package_root = Path(__file__).resolve().parents[2] / 'training' / 'neuralese'
    if package_root.is_dir() and str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    from natlang_neuralese.common.paths import root
    return root('data_hdd').expanduser().resolve()


def _write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.pending')
    with temporary.open('wb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _json(raw: bytes, label: str):
    try:
        return json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f'{label} is not valid JSON') from error


def _jsonl_rows(raw: bytes, label: str) -> list[dict]:
    """Parse JSONL on LF only; U+2028/U+2029 are legal inside JSON strings."""
    rows = []
    for line in raw.split(b'\n'):
        if not line:
            continue
        value = _json(line, label)
        if not isinstance(value, dict):
            raise ValueError(f'{label} must contain JSON objects')
        rows.append(value)
    return rows


def completed_entries(queue_raw: bytes, journal_raw: bytes, *, after: set[str] | None = None) -> list[dict]:
    """Return newly finished, fully accounted queue rows in original order.

    A finish event is only a readiness signal. Its collector output manifest
    and raw payload are checked separately by ``seal_tranche`` before export.
    """
    queue = _jsonl_rows(queue_raw, 'teacher queue')
    if not queue or any(not isinstance(row, dict) for row in queue):
        raise ValueError('teacher queue must be a nonempty JSONL object sequence')
    by_key = {}
    for row in queue:
        key = row.get('key')
        if not isinstance(key, str) or not key or key in by_key:
            raise ValueError('teacher queue keys must be distinct nonempty strings')
        by_key[key] = row
    finishes = {}
    for event in _jsonl_rows(journal_raw, 'teacher journal'):
        if event.get('event') != 'finish':
            continue
        key = event.get('key')
        if key not in by_key:
            raise ValueError('teacher journal contains a foreign finished key')
        if key in finishes and finishes[key] != event:
            raise ValueError('teacher journal rewrote a finished key')
        finishes[key] = event
    already = after or set()
    selected = []
    for row in queue:
        key = row['key']
        if key in already or key not in finishes:
            continue
        accounting = finishes[key].get('output_accounting') or {}
        states = accounting.get('job_states') or []
        if (accounting.get('version') != 'natlang.supervisor_output_accounting/2'
                or accounting.get('manifest_valid') is not True
                or accounting.get('manifest_errors')
                or accounting.get('complete') is not True
                or accounting.get('expected_jobs') != len(states) or not states
                or any(item.get('state') not in ('result', 'generation_held') for item in states)
                or accounting.get('exact_result_rows') != sum(item.get('state') == 'result' for item in states)
                or not re.fullmatch(r'[0-9a-f]{64}', accounting.get('output_sha256', ''))):
            continue
        selected.append({**row, 'finish_event': finishes[key]})
    return selected


def _completed_outputs_sha256(queue_raw: bytes, journal_raw: bytes) -> str:
    """Activity heartbeats cannot postpone custody of an already finished tail."""
    completed = [{'key': row['key'], 'finish_event': row['finish_event']}
                 for row in completed_entries(queue_raw, journal_raw)]
    return _sha(json.dumps(completed, sort_keys=True, separators=(',', ':')).encode())


def aggregate_worker_journals(queue_raw: bytes, worker_queues: list[tuple[str, bytes]],
                              worker_journals: list[tuple[str, bytes | None]]) -> bytes:
    """Join the real per-worker journals after proving their queue partition.

    Active collectors append to ``journal-worker-N.jsonl``; the shared
    ``journal.jsonl`` is only a drained-boundary artifact. This validates the
    exact worker queue cover and that every complete journal event belongs to
    its worker. A trailing non-LF fragment is an in-progress append and is
    withheld until the next poll; malformed complete records fail closed.
    """
    master_rows = _jsonl_rows(queue_raw, 'master teacher queue')
    master = {row.get('key'): row for row in master_rows}
    if not master_rows or None in master or len(master) != len(master_rows):
        raise ValueError('master queue keys must be unique')
    # The collector assigns worker-local output/job/log locations when it
    # partitions the original queue. Every other field is semantic request
    # identity (including less common sampling/runtime knobs), so compare the
    # complete rows after removing only those three plumbing paths.
    worker_local_fields = {'output', 'jobs', 'log'}
    worker_owner = {}
    worker_numbers = []
    for name, raw in worker_queues:
        if not re.fullmatch(r'queue-worker-[0-9]+\.jsonl', name):
            raise ValueError('worker queue has an unexpected filename')
        worker_numbers.append(int(name[len('queue-worker-'):-len('.jsonl')]))
        rows = _jsonl_rows(raw, name)
        # Batches smaller than the configured worker count have empty shards.
        # Exact master coverage and per-shard event ownership remain mandatory.
        for row in rows:
            key = row.get('key')
            if key not in master or key in worker_owner:
                raise ValueError('worker queue keys are foreign or duplicated')
            master_identity = {field: value for field, value in master[key].items()
                               if field not in worker_local_fields}
            worker_identity = {field: value for field, value in row.items()
                               if field not in worker_local_fields}
            if worker_identity != master_identity:
                raise ValueError('worker queue identity differs from master queue')
            worker_owner[key] = name
    if set(worker_owner) != set(master):
        raise ValueError('worker queues do not exactly partition the master queue')
    if len(worker_numbers) != len(set(worker_numbers)) or worker_numbers != list(range(len(worker_numbers))):
        raise ValueError('worker queues are not in exact numeric contiguous order')
    journal_map = dict(worker_journals)
    expected_names = {name.replace('queue-worker-', 'journal-worker-') for name, _ in worker_queues}
    if set(journal_map) != expected_names:
        raise ValueError('worker journals do not match exact queue worker set')
    combined = bytearray()
    for queue_name, _ in worker_queues:
        journal_name = queue_name.replace('queue-worker-', 'journal-worker-')
        raw = journal_map[journal_name] or b''
        complete = raw if not raw or raw.endswith(b'\n') else raw[:raw.rfind(b'\n') + 1]
        for row in _jsonl_rows(complete, journal_name):
            if row.get('key') not in master or worker_owner.get(row.get('key')) != queue_name:
                raise ValueError('worker journal event is foreign to its exact queue shard')
        combined.extend(complete)
    return bytes(combined)


def _teacher_source_binding(raw: bytes, *, batch: int, source_sha256: str) -> dict:
    source = _json(raw, 'teacher source')
    required = ('actor', 'actual_teacher_execution_identity', 'actual_task_runtime',
                'collection_controller', 'runtime_source_receipt', 'teacher_configuration_source',
                'source_provenance')
    runtime = source.get('actual_task_runtime')
    runtime_present = (isinstance(runtime, str) and bool(runtime.strip())) or (
        isinstance(runtime, dict) and bool(runtime))
    structured = tuple(name for name in required if name != 'actual_task_runtime')
    if (source.get('schema') != 'natlang.sandbox-strong-teacher-collection/1'
            or source.get('batch') != batch or source.get('source_cases_sha256') != source_sha256
            or not runtime_present
            or any(not isinstance(source.get(name), dict) or not source[name] for name in structured)):
        raise ValueError('teacher source lacks exact batch, source, actor, runtime, or controller custody')
    return source


def _execution_source(context: dict[str, bytes], entries: list[dict]) -> tuple[int, str]:
    """Authenticate the declared existing root transformation without collapsing repeats."""
    cases = _jsonl_rows(context['source-cases.jsonl'], 'sealed source cases')
    teacher = _json(context['teacher-source.json'], 'teacher source')
    try:
        from .teacher_source_sequence import collection_sampling, verify_collection_sampling
    except ImportError:
        from teacher_source_sequence import collection_sampling, verify_collection_sampling
    configuration=teacher.get('source_provenance')or{}
    sampling=collection_sampling(configuration)
    sampling_entries=_jsonl_rows(context['queue.jsonl'],'original collection queue')if 'queue.jsonl'in context else entries
    verify_collection_sampling(teacher,sampling_entries,configuration)
    if 'collection_sampling'in configuration:
        master={row['key']:row for row in sampling_entries}
        if any({key:value for key,value in row.items()if key!='finish_event'}!=master.get(row.get('key'))for row in entries):
            raise ValueError('collection tranche entry differs from its original complete queue')
    packet_present = 'packet.jsonl' in context or 'preparation.json' in context
    if 'collection_sampling'in configuration and not packet_present:
        raise ValueError('declared collection sampling lacks its executed preparation')
    if packet_present:
        if not {'packet.jsonl', 'preparation.json'} <= context.keys():
            raise ValueError('prepared execution needs packet and preparation bytes')
        preparation = _json(context['preparation.json'], 'rollout preparation')
        source_path = teacher.get('source_cases')
        if (not isinstance(source_path, str)
                or preparation.get('schema') != 'natlang.task-rollout-preparation/1'
                or preparation.get('root_mode') != sampling['root_mode']
                or preparation.get('source_cases') != source_path
                or preparation.get('source_sha256') != _sha(context['source-cases.jsonl'])
                or preparation.get('packet_sha256') != _sha(context['packet.jsonl'])):
            raise ValueError('execution preparation differs from sealed source/packet')
        package = Path(__file__).resolve().parents[2] / 'training/neuralese'
        if str(package) not in sys.path:
            sys.path.insert(0, str(package))
        from natlang_neuralese.common.task_roots import student_root_row, authored_root_row
        packet = _jsonl_rows(context['packet.jsonl'], 'executed root packet')
        transform=authored_root_row if sampling['root_mode']=='authored'else student_root_row
        canonical = lambda rows: json.dumps(rows, sort_keys=True, ensure_ascii=False,
                                            separators=(',', ':'), allow_nan=False)
        if canonical(packet) != canonical([transform(row) for row in cases]):
            raise ValueError('executed packet differs from the declared existing root transformation')
        expected_source = str(Path(source_path).with_name('packet.jsonl'))
        executed_sha = _sha(context['packet.jsonl'])
    else:
        packet, executed_sha = cases, _sha(context['source-cases.jsonl'])
        expected_source = teacher.get('source_cases')
    if not packet:
        raise ValueError('executed source packet is empty')
    for row in entries:
        index, count = row.get('index'), row.get('count')
        if (type(index) is not int or type(count) is not int or index < 0
                or count < 1 or index + count > len(packet)
                or (expected_source is not None and row.get('source') != expected_source)):
            raise ValueError('queue job is outside its exact executed source range')
    return len(cases), executed_sha


def _execution_paths(entries: list[dict], source_root: str, cases_path: str) -> dict:
    sources = {row.get('source') for row in entries}
    if sources == {cases_path}:
        return {}
    if sources != {f'{source_root}/packet.jsonl'}:
        raise ValueError('active queue does not bind one sealed executable source')
    return {'packet.jsonl': f'{source_root}/packet.jsonl',
            'preparation.json': f'{source_root}/preparation.json'}


def _repair_payload_bindings(completion: dict) -> dict[str, dict]:
    repair_keys = {row['key'] for row in (completion.get('completion') or {}).get('repair_required', [])
                   if isinstance(row, dict) and isinstance(row.get('key'), str)}
    result = {}
    for row in completion.get('individual_outcomes') or []:
        key = row.get('key')
        if key not in repair_keys:
            continue
        finish = row.get('finish') or {}
        states = ((finish.get('output_accounting') or {}).get('job_states') or [])
        for field in ('terminal_error', 'unverified_error'):
            for error in (state.get(field) for state in states):
                if not isinstance(error, dict):
                    continue
                path, digest = error.get('path'), error.get('sha256')
                if not isinstance(path, str) or not re.fullmatch(r'[0-9a-f]{64}', str(digest)):
                    raise ValueError('repair error lacks exact source path and SHA binding')
                name = f'repair-payloads/{key}/{Path(path).name}'
                if name in result:
                    raise ValueError('repair output repeats a payload path')
                result[name] = {'source_path': path, 'sha256': digest}
    return result


def _budget_payload_bindings(completion, queue):
    """Retain failed budget outcomes as control evidence, never positive rows."""
    queued = {row['key']: row for row in queue}
    result = {}; seen = set()
    for failure in (completion.get('completion') or {}).get('model_budget_failures', []):
        key = failure.get('key')
        if (key not in queued or key in ('.', '..') or key in seen or failure.get('training_admission') is not False
                or not re.fullmatch(r'[A-Za-z0-9._:-]+', key)):
            raise ValueError('invalid or repeated terminal budget queue identity')
        seen.add(key)
        result[f'budget-payloads/{key}/output-manifest.json'] = {
            'source_path': queued[key]['output'] + '.manifest.json'}
        errors = failure.get('errors')
        if not isinstance(errors, list) or not errors:
            raise ValueError('terminal budget outcome lacks exact error evidence')
        for error in errors:
            path, digest = error.get('path'), error.get('sha256')
            if not isinstance(path, str) or not re.fullmatch(r'[0-9a-f]{64}', str(digest)):
                raise ValueError('terminal budget error lacks path and SHA binding')
            name = f'budget-payloads/{key}/{Path(path).name}'
            if name in result:
                raise ValueError('terminal budget outcome repeats a payload')
            result[name] = {'source_path': path, 'sha256': digest}
    return result


def _verify_budget_outcomes(completion, queue, files, payloads):
    """Reuse the shared classifier against an exact local mirror of sealed bytes."""
    try:
        from .teacher_batch_completion import batch_completion
    except ImportError:
        from teacher_batch_completion import batch_completion
    with tempfile.TemporaryDirectory(prefix='teacher-budget-boundary-') as temporary:
        root = Path(temporary)
        def resolve(path):
            value = Path(path)
            if '..' in value.parts or not value.is_absolute():
                raise ValueError('budget evidence path must be an exact absolute source path')
            return root / str(value).lstrip('/')
        def put(path, raw):
            target = resolve(path); target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.read_bytes() != raw:
                raise ValueError('budget evidence source aliases conflicting bytes')
            target.write_bytes(raw)
        for entry in queue:
            source_name = 'packet.jsonl' if 'packet.jsonl' in files else 'source-cases.jsonl'
            put(entry['source'], files[source_name])
        for name, binding in payloads.items():
            put(binding['source_path'], files[name])
        verified = completion['completion']
        actual = batch_completion(queue, _jsonl_rows(files['journal.jsonl'], 'budget boundary journal'),
            verified['worker_returncodes'], _json(files['score.json'], 'budget boundary score'),
            resolve_path=resolve)
        budgets = actual['model_budget_failures']
        for failure in budgets:
            for error in failure['errors']:
                error['path'] = '/' + str(Path(error['path']).relative_to(root))
        if budgets != verified.get('model_budget_failures', []):
            raise ValueError('terminal budget disposition differs from exact source/error/manifest verification')


def seal_tranche(*, batch: int, number: int, entries: list[dict], context: dict[str, bytes],
                 outputs: dict[str, tuple], directory: Path) -> dict:
    """Persist one immutable archive of complete raw outputs and source context.

    ``context`` should include teacher-source, the full original queue, source
    cases, and the current repair/control snapshots. ``outputs`` maps queue key
    to (result JSONL bytes, result manifest bytes).
    """
    if type(batch) is not int or batch < 0 or type(number) is not int or number < 0:
        raise ValueError('batch and tranche number must be nonnegative integers')
    if not entries or len({row.get('key') for row in entries}) != len(entries):
        raise ValueError('tranche requires distinct completed queue entries')
    if set(outputs) != {row['key'] for row in entries}:
        raise ValueError('tranche output set must exactly match its finished queue keys')
    if not {'teacher-source.json', 'queue.jsonl'} <= set(context):
        raise ValueError('tranche needs exact teacher-source and original queue snapshots')
    if 'source-cases.jsonl' not in context:
        raise ValueError('tranche needs exact source-case bytes')
    _teacher_source_binding(context['teacher-source.json'], batch=batch,
                            source_sha256=_sha(context['source-cases.jsonl']))
    _, executed_source_sha = _execution_source(context, entries)
    for name in context:
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or not name or '\\' in name:
            raise ValueError('context path escapes tranche')

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    receipt_path = directory / f'tranche-{number:04d}.json'
    input_identity = {name: {'bytes': len(raw), 'sha256': _sha(raw)} for name, raw in sorted(context.items())}
    rows = []
    for index, entry in enumerate(entries):
        key = entry['key']
        if not isinstance(key, str) or not re.fullmatch(r'[A-Za-z0-9._:-]+', key):
            raise ValueError('queue key is not a safe tranche filename component')
        supplied = outputs[key]
        if len(supplied) not in (2, 3):
            raise ValueError('tranche output must include result, manifest, and optional held-payload map')
        output_raw, manifest_raw = supplied[:2]
        extras = supplied[2] if len(supplied) == 3 else {}
        if not isinstance(extras, dict):
            raise ValueError('held payloads must be an exact relative-path map')
        manifest = _json(manifest_raw, f'{key} output manifest')
        source = entry.get('source')
        accounting = entry['finish_event']['output_accounting']
        states = accounting['job_states']
        result_states = [row for row in states if row.get('state') == 'result']
        held_states = [row for row in states if row.get('state') == 'generation_held']
        expected_held_paths = {f"held-errors/{row['index']:06d}.error.json" for row in held_states}
        if set(extras) != expected_held_paths:
            raise ValueError(f'{key} tranche lacks exact held-error payloads')
        for state in held_states:
            held_error = _json(extras[f"held-errors/{state['index']:06d}.error.json"],
                               f'{key} held error')
            if (held_error.get('index') != state['index']
                    or held_error.get('program_id') != state['program_id']
                    or not isinstance(held_error.get('generation_hold'), str)
                    or not held_error['generation_hold']):
                raise ValueError(f'{key} held-error payload does not bind its exact source job')
        output_sha256 = _sha(output_raw)
        missing = [row['index'] for row in states if row.get('state') != 'result']
        if (manifest.get('version') != 'natlang.teacher_batch.native/1'
                or manifest.get('source') != source
                or manifest.get('output_sha256') != output_sha256
                or accounting.get('output_sha256') != output_sha256
                or manifest.get('completed') != len(result_states)
                or manifest.get('missing') != missing
                or 'source-cases.jsonl' not in context
                or manifest.get('source_sha256') != executed_source_sha
                or manifest.get('range') != {'start': entry.get('index'),
                                               'count': (entry.get('finish_event', {}).get('output_accounting', {})
                                                         .get('expected_jobs'))}
                or any(not isinstance(row.get('index'), int) or not isinstance(row.get('program_id'), str)
                       for row in held_states)):
            raise ValueError(f'{key} output manifest does not bind its completed source range')
        prefix = f'outputs/{index:04d}-{key}'
        result_name, manifest_name = prefix + '.results.jsonl', prefix + '.manifest.json'
        held_bindings = {
            name: {'path': f'outputs/{index:04d}-{key}/{name}',
                   'bytes': len(raw), 'sha256': _sha(raw)}
            for name, raw in sorted(extras.items())
        }
        rows.append({
            'key': key, 'task_id': entry.get('task_id'), 'index': entry.get('index'),
            'source': source, 'output': result_name,
            'output_bytes': len(output_raw), 'output_sha256': output_sha256,
            'output_manifest': manifest_name, 'manifest_bytes': len(manifest_raw),
            'manifest_sha256': _sha(manifest_raw),
            'finish_event_sha256': _sha(json.dumps(entry['finish_event'], sort_keys=True,
                                                    separators=(',', ':')).encode()),
            'held_payloads': held_bindings,
        })
    descriptor = {'schema': 'natlang.teacher-incremental-export-tranche/1',
                  'batch': batch, 'number': number, 'keys': [row['key'] for row in rows],
                  'context': input_identity, 'outputs': rows,
                  'training_admission': False, 'exposure': False,
                  'claim': 'exact completed raw teacher outputs plus source/control bindings'}
    descriptor_raw = (json.dumps(descriptor, sort_keys=True, indent=2) + '\n').encode()
    expected_files = {'tranche-manifest.json': descriptor_raw}
    for index, entry in enumerate(entries):
        prefix = f"outputs/{index:04d}-{entry['key']}"
        expected_files[prefix + '.results.jsonl'] = outputs[entry['key']][0]
        expected_files[prefix + '.manifest.json'] = outputs[entry['key']][1]
        extras = outputs[entry['key']][2] if len(outputs[entry['key']]) == 3 else {}
        expected_files.update({prefix + '/' + name: raw for name, raw in extras.items()})
    expected_files.update({'context/' + name: raw for name, raw in context.items()})
    if receipt_path.exists():
        existing = json.loads(receipt_path.read_text())
        expected = {'batch': batch, 'number': number, 'keys': [row['key'] for row in rows],
                    'context': input_identity, 'outputs': rows}
        if any(existing.get(key) != value for key, value in expected.items()):
            raise ValueError('existing tranche identity or payload bindings differ')
        archive_receipt = existing['archive_receipt']
        archive_proof = source_archive.verify(directory.parent,
                                              f'teacher-batch-{batch:04d}-tranche-{number:04d}')
        if archive_proof.get('archive_sha256') != archive_receipt.get('archive_sha256'):
            raise ValueError('existing tranche archive differs from its receipt')
        # Verify the resumable working tree too. Never rewrite an immutable
        # tranche before discovering that a retry supplied different bytes.
        for row in rows:
            for field, byte_field, sha_field in (
                ('output', 'output_bytes', 'output_sha256'),
                ('output_manifest', 'manifest_bytes', 'manifest_sha256'),
            ):
                raw = (directory / row[field]).read_bytes()
                if len(raw) != row[byte_field] or _sha(raw) != row[sha_field]:
                    raise ValueError('existing tranche payload differs from its immutable receipt')
        for name, identity in input_identity.items():
            raw = (directory / 'context' / name).read_bytes()
            if len(raw) != identity['bytes'] or _sha(raw) != identity['sha256']:
                raise ValueError('existing tranche context differs from its immutable receipt')
        for row in rows:
            for binding in row['held_payloads'].values():
                raw = (directory / binding['path']).read_bytes()
                if len(raw) != binding['bytes'] or _sha(raw) != binding['sha256']:
                    raise ValueError('existing held-error payload differs from its immutable receipt')
        manifest_raw = (directory / 'tranche-manifest.json').read_bytes()
        if _sha(manifest_raw) != existing.get('tranche_manifest_sha256'):
            raise ValueError('existing tranche manifest differs from its immutable receipt')
        return existing
    archive_name = f'teacher-batch-{batch:04d}-tranche-{number:04d}'
    archive_receipt_path = directory.parent / (archive_name + '.receipt.json')
    if archive_receipt_path.exists():
        # Recover a crash after archive creation but before the tranche receipt
        # was committed. Compare the entire immutable member inventory first.
        archive_receipt = json.loads(archive_receipt_path.read_text())
        bound = {row['path']: {'bytes': row['bytes'], 'sha256': row['sha256']}
                 for row in archive_receipt.get('manifest', {}).get('files', [])}
        wanted = {name: {'bytes': len(raw), 'sha256': _sha(raw)}
                  for name, raw in expected_files.items()}
        if bound != wanted:
            raise ValueError('orphaned tranche archive has different immutable members')
        for name, raw in expected_files.items():
            current = (directory / name).read_bytes()
            if current != raw:
                raise ValueError('orphaned tranche working tree differs from its archived bytes')
    else:
        for name, raw in expected_files.items():
            _write(directory / name, raw)
        archive_receipt = source_archive.pack(directory, directory.parent, archive_name)
    archive_proof = source_archive.verify(directory.parent, archive_name)
    sealed = {**descriptor, 'directory': str(directory),
              'tranche_manifest_sha256': _sha(descriptor_raw),
              'archive_receipt': archive_receipt, 'archive_proof': archive_proof,
              'private_export': None, 'sealed_at': time.time()}
    _write(receipt_path, (json.dumps(sealed, sort_keys=True, indent=2) + '\n').encode())
    return sealed


def accept_private_receipt(path: Path, tranche: dict) -> dict:
    """Bind a successful existing-exporter receipt to a tranche; fail closed."""
    receipt = json.loads(Path(path).read_text())
    archive = tranche['archive_receipt']
    identity = {'bytes': archive['archive_bytes'], 'sha256': archive['archive_sha256']}
    verification = receipt.get('independent_verification') or {}
    if (receipt.get('private') is not True or receipt.get('identity') != identity
            or verification.get('status') == 'awaiting_target_readback'
            or verification.get('bytes') != identity['bytes']
            or verification.get('sha256') != identity['sha256']):
        raise ValueError('private export receipt lacks matching independent full-object verification')
    accepted = {**tranche, 'private_export': {'receipt_path': str(Path(path).resolve()),
        'run': receipt.get('run'), 'name': receipt.get('name'), 'identity': identity,
        'verified_at': receipt.get('verified_at')}}
    target = (Path(tranche['directory']) / 'boundary-private-export.json'
              if 'boundary' in tranche else
              Path(tranche['directory']) / f"tranche-{tranche['number']:04d}.json")
    _write(target, (json.dumps(accepted, sort_keys=True, indent=2) + '\n').encode())
    return accepted


def _verify_saved_private_receipt(tranche: dict) -> dict:
    saved = tranche.get('private_export')
    if not isinstance(saved, dict) or not isinstance(saved.get('receipt_path'), str):
        raise ValueError('tranche has no saved private receipt binding')
    receipt = json.loads(Path(saved['receipt_path']).read_text())
    identity = {'bytes': tranche['archive_receipt']['archive_bytes'],
                'sha256': tranche['archive_receipt']['archive_sha256']}
    verification = receipt.get('independent_verification') or {}
    if (receipt.get('private') is not True or receipt.get('identity') != identity
            or verification.get('status') == 'awaiting_target_readback'
            or receipt.get('status') == 'awaiting_target_readback'
            or verification.get('bytes') != identity['bytes']
            or verification.get('sha256') != identity['sha256']
            or saved.get('identity') != identity):
        raise ValueError('saved private receipt no longer verifies this tranche')
    return tranche


def prune_verified_local_tranche(tranche: dict) -> dict:
    """Remove local raw/archive duplicates only after the saved private GET proof."""
    _verify_saved_private_receipt(tranche)
    directory = Path(tranche['directory']).resolve(strict=True)
    keep = directory / f"tranche-{tranche['number']:04d}.json"
    if not keep.is_file():
        raise ValueError('durable tranche manifest is missing before spool cleanup')
    archive_name = f"teacher-batch-{tranche['batch']:04d}-tranche-{tranche['number']:04d}.tar.zst"
    archive_path = Path(tranche['archive_receipt']['archive']).resolve()
    if archive_path != directory.parent / archive_name:
        raise ValueError('local tranche archive path differs from its owned spool location')
    archive_bytes_removed = 0
    if archive_path.exists():
        if archive_path.is_symlink() or not archive_path.is_file():
            raise ValueError('local tranche archive is not a regular owned file')
        archive_bytes_removed = archive_path.stat().st_size
        archive_path.unlink()
    # The immutable tranche receipt retains every member SHA/path; private
    # object verification is the recovery source for raw results and controls.
    for child in directory.iterdir():
        if child == keep:
            continue
        if child.is_symlink():
            raise ValueError('refusing to prune an aliased tranche spool entry')
        if child.is_dir():
            import shutil
            shutil.rmtree(child)
        else:
            child.unlink()
    return {'pruned': True, 'archive_bytes_removed': archive_bytes_removed,
            'retained_manifest': str(keep), 'private_identity': tranche['private_export']['identity']}


def export_private_tranche(tranche: dict, *, exporter: Path, role: str,
                           receipt_path: Path) -> dict:
    """Use the existing owner-host private exporter and require its full GET proof."""
    receipt_path = Path(receipt_path)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    archive = tranche['archive_receipt']
    env = {**os.environ, 'CW_EXPORT_LOCAL': '1',
           'CW_EXPORT_EXPECTED_SHA256': archive['archive_sha256'],
           'CW_EXPORT_EXPECTED_BYTES': str(archive['archive_bytes']),
           'CW_EXPORT_RECEIPT_FILE': str(Path(receipt_path).resolve()),
           'CW_EXPORT_UPLOAD_PHASE_FILE': str(Path(receipt_path).with_suffix('.phase.json').resolve())}
    # Force the existing owner-host exporter to perform its private GET; a
    # target-side readback request is only an upload receipt, not durability.
    env.pop('CW_EXPORT_TARGET_VERIFIES', None)
    subprocess.run([sys.executable, str(Path(exporter).resolve()), role,
                    str(Path(archive['archive']).resolve())], check=True, env=env)
    return accept_private_receipt(receipt_path, tranche)


def _read_tranche_files(read_file, paths):
    """Collect once in source order; only scoped SDK readers opt into overlap."""
    paths = list(dict.fromkeys(paths))
    read_many = getattr(read_file, 'read_many', None)
    return read_many(paths) if callable(read_many) else {path: read_file(path) for path in paths}


def export_finished_tranches(*, batch: int, queue_path: str, journal_path: str,
                              context_paths: dict[str, str], directory: Path,
                              read_file: Callable[[str], bytes], export_file: Callable[[dict, Path], dict],
                              chunk_size: int = 32, flush_partial: bool = False,
                              prune_after_private_verification: bool = False,
                              queue_snapshot: bytes | None = None,
                              journal_snapshot: bytes | None = None) -> dict:
    """Export newly completed rows; safe to call repeatedly from an owner-host poller.

    ``read_file`` is the scoped provider SDK read. ``export_file`` calls the
    existing single-file private uploader and returns a receipt-verified
    tranche. A durable index advances only after that receipt is accepted.
    """
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError('tranche chunk size must be positive')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = directory / '.export.lock'
    with lock_path.open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        queue_raw = queue_snapshot if queue_snapshot is not None else read_file(queue_path)
        queue_sha = _sha(queue_raw)
        state_path = directory / 'frontier.json'
        if state_path.exists():
            state = json.loads(state_path.read_text())
            if state.get('schema') != 'natlang.teacher-incremental-export-frontier/1' \
                    or state.get('batch') != batch or state.get('queue_sha256') != queue_sha:
                raise ValueError('durable export frontier belongs to a different batch or queue')
        else:
            state = {'schema': 'natlang.teacher-incremental-export-frontier/1', 'batch': batch,
                     'queue_sha256': queue_sha, 'completed_keys': [], 'tranches': []}

        if prune_after_private_verification:
            # Repair a crash after the durable frontier write but before local
            # spool cleanup; every prior tranche is already private-verified.
            for previous in state['tranches']:
                prior_dir = directory / f"tranche-{previous['number']:04d}"
                prior_path = prior_dir / f"tranche-{previous['number']:04d}.json"
                if prior_path.is_file():
                    prune_verified_local_tranche(json.loads(prior_path.read_text()))

        # A sealed-but-not-uploaded tranche survives interruption. Finish that
        # exact object before reading/packing another tranche.
        while True:
            number = len(state['tranches'])
            tranche_dir = directory / f'tranche-{number:04d}'
            tranche_receipt_path = tranche_dir / f'tranche-{number:04d}.json'
            if tranche_receipt_path.exists():
                tranche = json.loads(tranche_receipt_path.read_text())
                receipt_path = directory / 'private-receipts' / f'tranche-{number:04d}.json'
                if tranche.get('private_export') is not None:
                    accepted = _verify_saved_private_receipt(tranche)
                else:
                    accepted = export_file(tranche, receipt_path)
                state['completed_keys'].extend(tranche['keys'])
                state['tranches'].append({'number': number, 'keys': tranche['keys'],
                    'archive_sha256': tranche['archive_receipt']['archive_sha256'],
                    'private_receipt': str(receipt_path),
                    'private_identity': accepted['private_export']['identity']})
                _write(state_path, (json.dumps(state, sort_keys=True, indent=2) + '\n').encode())
                if prune_after_private_verification:
                    prune_verified_local_tranche(tranche)
                continue

            journal_raw = journal_snapshot if journal_snapshot is not None else read_file(journal_path)
            selected = completed_entries(queue_raw, journal_raw, after=set(state['completed_keys']))
            # Finished rows in a sealed pending tranche were already removed
            # from this frontier; only choose a new sequence when none exists.
            if len(selected) < chunk_size and not (flush_partial and selected):
                break
            selected = selected[:chunk_size]
            selected_context = {'teacher-source.json': context_paths['teacher-source.json'],
                                'source-cases.jsonl': context_paths['source-cases.jsonl']}
            for name, path in context_paths.items():
                if name in ('teacher-source.json', 'source-cases.jsonl', 'queue.jsonl'):
                    continue
                selected_context[name] = path
            paths = list(selected_context.values())
            held_paths = {}
            for row in selected:
                paths.extend((row['output'], row['output'] + '.manifest.json'))
                held_paths[row['key']] = {}
                for job in row['finish_event']['output_accounting']['job_states']:
                    if job.get('state') == 'generation_held':
                        path = f"{row['jobs'].rstrip('/')}/{job['index']:06d}.error.json"
                        held_paths[row['key']][f"held-errors/{job['index']:06d}.error.json"] = path
                        paths.append(path)
            collected = _read_tranche_files(read_file, paths)
            context = {'teacher-source.json': collected[selected_context['teacher-source.json']],
                       'queue.jsonl': queue_raw,
                       'source-cases.jsonl': collected[selected_context['source-cases.jsonl']]}
            for name, path in selected_context.items():
                if name not in context:
                    context[name] = collected[path]
            outputs = {}
            for row in selected:
                outputs[row['key']] = (collected[row['output']], collected[row['output'] + '.manifest.json'])
                held = {name: collected[path] for name, path in held_paths[row['key']].items()}
                if held:
                    outputs[row['key']] = (*outputs[row['key']], held)
            tranche = seal_tranche(batch=batch, number=number, entries=selected,
                context=context, outputs=outputs, directory=tranche_dir)
            receipt_path = directory / 'private-receipts' / f'tranche-{number:04d}.json'
            accepted = export_file(tranche, receipt_path)
            state['completed_keys'].extend(tranche['keys'])
            state['tranches'].append({'number': number, 'keys': tranche['keys'],
                'archive_sha256': tranche['archive_receipt']['archive_sha256'],
                'private_receipt': str(receipt_path),
                'private_identity': accepted['private_export']['identity']})
            _write(state_path, (json.dumps(state, sort_keys=True, indent=2) + '\n').encode())
            if prune_after_private_verification:
                prune_verified_local_tranche(accepted)

        return {'state': state, 'queue_count': len(_jsonl_rows(queue_raw, 'teacher queue')),
                'new_tranches': len(state['tranches']), 'durable_completed_keys': len(state['completed_keys']),
                'training_admission': False, 'exposure': False}


def seal_boundary(*, batch: int, queue_raw: bytes, exported_keys: list[str],
                  repair_keys: list[str], unstarted_keys: list[str], files: dict[str, bytes],
                  directory: Path) -> dict:
    """Seal the final control/repair snapshot after proving the queue partition."""
    queue = _jsonl_rows(queue_raw, 'teacher boundary queue')
    keys = [row.get('key') for row in queue]
    if not keys or any(not isinstance(key, str) for key in keys) or len(set(keys)) != len(keys):
        raise ValueError('final boundary queue keys must be distinct')
    if not {'teacher-source.json', 'source-cases.jsonl', 'queue.jsonl', 'journal.jsonl', 'score.json',
            'rewards.jsonl', 'completion.json', 'repair-queue.jsonl'} <= set(files):
        raise ValueError('final boundary is missing canonical source, accounting or repair control')
    if files['queue.jsonl'] != queue_raw:
        raise ValueError('final boundary queue bytes differ from original queue binding')
    _execution_source(files, queue)
    completion = _json(files['completion.json'], 'completion')
    _teacher_source_binding(files['teacher-source.json'], batch=batch,
                            source_sha256=_sha(files['source-cases.jsonl']))
    if completion.get('schema') != 'natlang.teacher-generation-boundary/1' or completion.get('batch') != batch:
        raise ValueError('final boundary does not bind the actual teacher boundary schema and batch')
    verified = completion.get('completion') or {}
    if verified.get('schema') != 'natlang.strong-teacher-batch-completion/1':
        raise ValueError('final boundary lacks the shared batch-completion proof')
    budget_payloads = _budget_payload_bindings(completion, queue)
    budget_keys = [row['key'] for row in verified.get('model_budget_failures', [])]
    groups = [exported_keys, repair_keys, unstarted_keys, budget_keys]
    flat = [key for group in groups for key in group]
    if len(flat) != len(set(flat)) or set(flat) != set(keys):
        raise ValueError('durable, repair, unstarted and terminal-budget keys do not partition the original queue')
    outcomes = completion.get('individual_outcomes') or []
    outcome_by_key = {row.get('key'): row for row in outcomes if isinstance(row, dict)}
    if len(outcome_by_key) != len(outcomes) or set(outcome_by_key) != set(keys):
        raise ValueError('final boundary outcomes do not cover the exact original queue')
    queue_by_key = {row['key']: row for row in queue}
    for key, row in outcome_by_key.items():
        queued = queue_by_key[key]
        if any(row.get(field) != queued.get(field) for field in ('source', 'index', 'seed')):
            raise ValueError('final boundary outcome identity differs from its exact queue row')
    expected_bindings = {
        'queue.jsonl': ('queue.jsonl',), 'journal.jsonl': ('journal.jsonl',),
        'score.json': ('score.json',), 'teacher-source.json': ('teacher-source.json',),
    }
    bindings = completion.get('bindings') or {}
    for key, names in expected_bindings.items():
        binding = bindings.get(key)
        if (not isinstance(binding, dict) or binding.get('bytes') != len(files[names[0]])
                or binding.get('sha256') != _sha(files[names[0]])):
            raise ValueError(f'final boundary completion does not bind current {key} bytes')
    complete_keys, repair_actual, unfinished_keys = set(), set(), set()
    for key, row in outcome_by_key.items():
        finish = row.get('finish')
        if finish is None:
            unfinished_keys.add(key)
            continue
        if finish.get('key') != key or finish.get('event') != 'finish':
            raise ValueError('final boundary finish event differs from its original queue key')
        accounting = finish.get('output_accounting') or {}
        states = accounting.get('job_states') or []
        if (accounting.get('complete') is True and accounting.get('manifest_valid') is True
                and not accounting.get('manifest_errors') and states
                and all(state.get('state') in ('result', 'generation_held') for state in states)):
            complete_keys.add(key)
        elif key not in budget_keys:
            repair_actual.add(key)
    repair_rows = verified.get('repair_required') or []
    journal = _jsonl_rows(files['journal.jsonl'], 'teacher boundary journal')
    started_keys = {row.get('key') for row in journal if row.get('event') == 'start'}
    if not started_keys <= set(keys):
        raise ValueError('final boundary journal contains a foreign started key')
    journal_finishes = {row.get('key'): row for row in journal if row.get('event') == 'finish'}
    if len(journal_finishes) != sum(row.get('event') == 'finish' for row in journal):
        raise ValueError('final boundary journal repeats a finished key')
    if any(outcome_by_key[key].get('finish') != finish for key, finish in journal_finishes.items()):
        raise ValueError('final boundary completion finish records differ from the full source journal')
    unstarted_expected = unfinished_keys - started_keys
    repair_actual |= unfinished_keys & started_keys
    repair_reported = {row['key'] for row in repair_rows if isinstance(row, dict)
                       and isinstance(row.get('key'), str)}
    required_repairs = repair_reported - unstarted_expected
    repair_queue = _jsonl_rows(files['repair-queue.jsonl'], 'final repair queue')
    expected_retry_keys = {f"batch-{batch:04d}:{key}" for key in required_repairs}
    actual_retry_keys = [row.get('key') for row in repair_queue]
    if (any(not isinstance(key, str) for key in actual_retry_keys)
            or len(actual_retry_keys) != len(set(actual_retry_keys))
            or set(actual_retry_keys) != expected_retry_keys):
        raise ValueError('final repair queue does not exactly preserve the guardian repair frontier')
    if (set(exported_keys) != complete_keys or set(repair_keys) != required_repairs
            or repair_actual != required_repairs or set(unstarted_keys) != unstarted_expected):
        raise ValueError('final boundary key partitions differ from actual per-key batch accounting')
    repair_payloads = _repair_payload_bindings(completion)
    terminal_payloads = {**repair_payloads, **budget_payloads}
    base_files = {'teacher-source.json', 'source-cases.jsonl', 'queue.jsonl', 'journal.jsonl',
                  'score.json', 'rewards.jsonl', 'completion.json', 'repair-queue.jsonl'}
    if 'packet.jsonl' in files:
        base_files.update({'packet.jsonl', 'preparation.json'})
    if set(files) - base_files != set(terminal_payloads):
        raise ValueError('final boundary must retain every exact terminal-error payload needed for repairs')
    for name, binding in terminal_payloads.items():
        if 'sha256' in binding and _sha(files[name]) != binding['sha256']:
            raise ValueError('final boundary repair payload bytes differ from collector binding')
    if budget_keys:
        _verify_budget_outcomes(completion, queue, files, budget_payloads)
    folder = Path(directory); folder.mkdir(parents=True, exist_ok=True)
    manifest = {'schema': 'natlang.teacher-incremental-export-boundary/1', 'batch': batch,
        'queue_sha256': _sha(queue_raw), 'queue_count': len(keys), 'exported_keys': exported_keys,
        'repair_keys': repair_keys, 'unstarted_keys': unstarted_keys,
        **({'model_budget_failure_keys': budget_keys} if budget_keys else {}),
        'files': {name: {'bytes': len(raw), 'sha256': _sha(raw)} for name, raw in sorted(files.items())},
        'batch_can_advance': bool(verified.get('can_advance')),
        'unbound_global_repairs': [row for row in verified.get('repair_required', [])
                                   if not isinstance(row, dict) or not isinstance(row.get('key'), str)],
        'private_durability': 'pending', 'training_admission': False, 'exposure': False}
    boundary_path = folder / 'boundary-manifest.json'
    manifest_raw = (json.dumps(manifest, sort_keys=True, indent=2)+'\n').encode()
    if boundary_path.exists():
        existing = json.loads(boundary_path.read_text())
        if existing != manifest:
            raise ValueError('existing final boundary snapshot is immutable')
        for name, identity in manifest['files'].items():
            raw = (folder / name).read_bytes()
            if len(raw) != identity['bytes'] or _sha(raw) != identity['sha256']:
                raise ValueError('existing final boundary file differs from its immutable manifest')
        archive_name=f'teacher-batch-{batch:04d}-final-boundary'
        archive_receipt_path = folder.parent/(archive_name+'.receipt.json')
        if archive_receipt_path.exists():
            archive_proof=source_archive.verify(folder.parent, archive_name)
            archive= json.loads(archive_receipt_path.read_text())
            if archive_proof.get('status') != 'passed':
                raise ValueError('existing final boundary archive no longer verifies')
            return {'boundary': existing, 'archive_receipt': archive,
                    'archive_proof': archive_proof, 'directory': str(folder)}
        # A crash after the durable boundary manifest but before archive
        # creation can resume without changing any bound bytes.
        archive=source_archive.pack(folder, folder.parent, archive_name)
        archive_proof=source_archive.verify(folder.parent, archive_name)
        return {'boundary': existing, 'archive_receipt': archive,
                'archive_proof': archive_proof, 'directory': str(folder)}
    for name, raw in files.items():
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('boundary file path escapes archive')
        _write(folder/name, raw)
    _write(boundary_path, manifest_raw)
    archive_name=f'teacher-batch-{batch:04d}-final-boundary'
    archive=source_archive.pack(folder, folder.parent, archive_name)
    proof=source_archive.verify(folder.parent, archive_name)
    return {'boundary': manifest, 'archive_receipt': archive, 'archive_proof': proof,
            'directory': str(folder)}


def export_final_boundary(*, batch: int, queue_path: str, exported_keys: list[str],
                          repair_keys: list[str], unstarted_keys: list[str],
                          file_paths: dict[str, str], read_file: Callable[[str], bytes],
                          directory: Path, exporter: Path, role: str) -> dict:
    """Persist the final source/accounting/repair/control set as one private object."""
    queue_raw = read_file(queue_path)
    files = {name: read_file(path) for name, path in file_paths.items()}
    sealed = seal_boundary(batch=batch, queue_raw=queue_raw, exported_keys=exported_keys,
        repair_keys=repair_keys, unstarted_keys=unstarted_keys, files=files,
        directory=Path(directory))
    receipt_path = Path(directory) / 'boundary-private-receipt.json'
    accepted_path = Path(directory) / 'boundary-private-export.json'
    if accepted_path.exists():
        accepted = json.loads(accepted_path.read_text())
        _verify_saved_private_receipt(accepted)
        return accepted
    return export_private_tranche(sealed, exporter=exporter, role=role,
                                  receipt_path=receipt_path)


def _sandbox_boot_read(sandbox):
    """Read the virtual kernel file through the existing process transport."""
    return sandbox.exec(['cat', '/proc/sys/kernel/random/boot_id'],
                        timeout_seconds=30).result(timeout=45)


def scoped_sdk_session(sandbox_id: str, *, timeout: int = 180):
    """Bind one scoped provider session and its read-only file reader."""
    cwctl = _cwctl_module()
    sandbox = cwctl.read_sandbox(sandbox_id, patience=60)

    def read(path: str) -> bytes:
        if path == '/proc/sys/kernel/random/boot_id':
            boot = _sandbox_boot_read(sandbox)
            if boot.returncode != 0:
                raise OSError(f'could not read active teacher boot identity: {boot.stderr}')
            if not boot.stdout.strip():
                raise OSError('active teacher boot identity read was empty')
            return boot.stdout.encode()
        try:
            result = sandbox.read_file(path).result(timeout=timeout)
        except (ValueError, KeyError, TypeError):
            # Keep identity/programming failures fatal in the publisher.
            raise
        except Exception as error:
            # The gateway's generic ReadFile error omits the requested path.
            # Retain it in the durable transport failure instead of forcing a
            # second pass over every source merely to locate the failed read.
            raise OSError(f'teacher ReadFile {path!r}: {type(error).__name__}: {error}') from error
        return result if isinstance(result, bytes) else bytes(result)

    def read_many(paths):
        from concurrent.futures import ThreadPoolExecutor
        paths = list(dict.fromkeys(paths))
        executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix='teacher-scoped-read')
        try:
            return dict(zip(paths, executor.map(read, paths)))
        finally:
            # Keep export custody until all in-flight SDK operations finish.
            # An error must not leave queued reads running after lock release.
            executor.shutdown(wait=True, cancel_futures=True)

    read.read_many = read_many
    return sandbox, read


def scoped_sdk_reader(sandbox_id: str, *, timeout: int = 180) -> Callable[[str], bytes]:
    """Bind one read-only provider session for an owner-host tranche poll."""
    return scoped_sdk_session(sandbox_id, timeout=timeout)[1]


def _validate_live_identity(config: dict, sandbox, read_file: Callable[[str], bytes]) -> dict:
    """Fail closed unless the active sandbox, source and serving identity match config."""
    expected = config.get('identity')
    required = {'sandbox_boot_id', 'teacher_source_sha256', 'source_cases_sha256',
                'actor_sha256', 'execution_identity_sha256', 'teacher_config_sha256',
                'runtime_path'}
    if not isinstance(expected, dict) or not required <= set(expected):
        raise ValueError('continuous poll config needs exact boot, source, actor, runtime and config bindings')
    boot = _sandbox_boot_read(sandbox)
    if boot.returncode != 0:
        raise OSError(f'could not read active teacher boot identity: {boot.stderr}')
    boot_id = boot.stdout.strip()
    if not boot_id or boot_id != expected['sandbox_boot_id']:
        raise ValueError('active teacher sandbox boot identity differs from the armed poll config')
    teacher_raw = read_file(config['context_paths']['teacher-source.json'])
    cases_raw = read_file(config['context_paths']['source-cases.jsonl'])
    teacher_sha, cases_sha = _sha(teacher_raw), _sha(cases_raw)
    if teacher_sha != expected['teacher_source_sha256'] or cases_sha != expected['source_cases_sha256']:
        raise ValueError('active teacher source/config bytes differ from the armed poll config')
    source = _teacher_source_binding(teacher_raw, batch=config['batch'], source_sha256=cases_sha)
    if (source['actor'].get('sha256') != expected['actor_sha256']
            or source['actual_teacher_execution_identity'].get('sha256') != expected['execution_identity_sha256']
            or source['teacher_configuration_source'].get('sha256') != expected['teacher_config_sha256']
            or source['actual_task_runtime'] != expected['runtime_path']):
        raise ValueError('active teacher actor/config/runtime fields differ from armed poll config')
    return {'sandbox_boot_id': boot_id, 'teacher_source_sha256': teacher_sha,
            'source_cases_sha256': cases_sha, 'actor_sha256': expected['actor_sha256'],
            'execution_identity_sha256': expected['execution_identity_sha256'],
            'teacher_config_sha256': expected['teacher_config_sha256'],
            'runtime_path': expected['runtime_path']}


def _closed_guardian_transition(publisher, guardian_raw, state, batch, read_file, *, owner_exited, phase_path=None):
    """Verify an advanced state and its exact old boundary; export no active rows."""
    control = publisher['control_dir']
    config_ref = publisher['guardian_config']

    def bound_json(reference, label):
        if (not isinstance(reference, dict) or not {'path', 'sha256', 'bytes'} <= reference.keys()
                or type(reference['bytes']) is not int or reference['bytes'] < 0):
            raise ValueError(label + ' lacks an exact file binding')
        raw = read_file(reference['path'])
        if len(raw) != reference['bytes'] or _sha(raw) != reference['sha256']:
            raise ValueError(label + ' changed')
        return _json(raw, label)

    if read_file('/proc/sys/kernel/random/boot_id').decode().strip() != publisher['sandbox_boot_id']:
        raise ValueError('guardian transition changed its boot identity')
    if read_file(config_ref['path']) != guardian_raw:
        raise ValueError('guardian config changed during feeder transition')
    fresh_raw = read_file(f'{control}/state.json')
    fresh = _json(fresh_raw, 'guardian transition state')
    if (fresh.get('schema') != state['schema'] or fresh.get('config') != config_ref):
        raise ValueError('guardian transition state changed its config authority')
    next_batch, next_owner = fresh.get('active_batch'), fresh.get('owner')
    advanced = (type(fresh.get('next_batch')) is int and fresh['next_batch'] > batch
                and (next_batch is None and next_owner is None or
                     type(next_batch) is int and next_batch > batch and isinstance(next_owner, dict)))
    if advanced:
        if next_owner is not None and (next_owner.get('boot_id') != publisher['sandbox_boot_id']
                or type(next_owner.get('pid')) is not int
                or not str(next_owner.get('start_ticks', '')).isdigit()):
            raise ValueError('advanced guardian feeder changed its process/boot authority')
        expected_path = f'{control}/boundaries/batch-{batch:04d}.json'
        reference = fresh.get('last_boundary')
        if not isinstance(reference, dict) or reference.get('path') != expected_path:
            closed = (fresh.get('source_sequence') or {}).get('closed', {}).get(str(batch))
            receipt = bound_json(closed, 'closed source-sequence transition')
            if receipt.get('batch') != batch:
                raise ValueError('closed source-sequence transition names another batch')
            reference = receipt.get('boundary')
        if not isinstance(reference, dict) or reference.get('path') != expected_path:
            raise ValueError('advanced guardian lacks its exact prior batch boundary')
        boundary = bound_json(reference, 'closed guardian transition boundary')
        if (boundary.get('schema') != 'natlang.teacher-generation-boundary/1'
                or boundary.get('batch') != batch or not isinstance(boundary.get('completion'), dict)
                or not isinstance(boundary.get('individual_outcomes'), list)):
            raise ValueError('guardian transition boundary has no canonical completion custody')
        bindings = boundary.get('bindings') or {}
        source = f"{publisher['generation_root']}/evidence-collections-v2/batch-{batch:04d}"
        for name in ('queue.jsonl', 'journal.jsonl', 'score.json', 'teacher-source.json'):
            item = bindings.get(name)
            if (not isinstance(item, dict) or item.get('path') != source + '/' + name
                    or type(item.get('bytes')) is not int or item['bytes'] < 0
                    or not isinstance(item.get('sha256'), str)
                    or not re.fullmatch('[0-9a-f]{64}', item['sha256'])):
                raise ValueError('guardian transition boundary lost its exact source bindings')
        return fresh, fresh_raw, reference
    if fresh.get('owner') != state.get('owner') or fresh.get('active_batch') != batch:
        raise ValueError('guardian owner changed without authenticated batch advancement')
    phase = _json(read_file(phase_path or f"{publisher['generation_root']}/feed-evidence-phase.json"),
                  'drained teacher feed phase')
    if owner_exited and phase.get('batch') == batch and phase.get('phase') == 'batch_complete':
        # Closing may need arbitrarily long to durably collect outcomes. This
        # result grants no new export custody; the service simply polls again.
        return fresh, fresh_raw, None
    raise ValueError('missing guardian feeder has no authenticated closed transition')


def _publisher_batch_runtime(publisher: dict, batch: int, read_file: Callable[[str], bytes]) -> str:
    """Resolve explicitly bound runtime epochs without relabeling started work."""
    original = (publisher.get('identity') or {}).get('runtime_path')
    epochs = publisher.get('runtime_epochs')
    if epochs is None:
        return original
    if not isinstance(epochs, list) or len(epochs) < 2:
        raise ValueError('runtime epochs need original and current guardian configurations')
    previous = None
    selected = None
    for index, epoch in enumerate(epochs):
        if not isinstance(epoch, dict) or set(epoch) != {'first_batch', 'guardian_config'}:
            raise ValueError('runtime epoch lacks exact batch and guardian configuration')
        first, reference = epoch['first_batch'], epoch['guardian_config']
        if (type(first) is not int or (index == 0 and first != publisher['anchor_batch'])
                or previous is not None and first <= previous):
            raise ValueError('runtime epochs must start at the anchor and strictly increase')
        if (not isinstance(reference, dict) or set(reference) != {'path', 'sha256', 'bytes'}
                or not isinstance(reference['path'], str) or type(reference['bytes']) is not int
                or reference['bytes'] < 0):
            raise ValueError('runtime epoch lacks an exact guardian file binding')
        raw = read_file(reference['path'])
        if len(raw) != reference['bytes'] or _sha(raw) != reference['sha256']:
            raise ValueError('runtime epoch guardian configuration changed')
        configuration = _json(raw, 'runtime epoch guardian configuration')
        runtime = configuration.get('task_runtime')
        if (not isinstance(runtime, str) or not runtime
                or configuration.get('control') != publisher['control_dir']
                or configuration.get('root') != publisher['generation_root']
                or (configuration.get('teacher_identity') or {}).get('sha256')
                    != publisher['identity']['execution_identity_sha256']):
            raise ValueError('runtime epoch changed controller, generation root or teacher identity')
        if index == 0 and runtime != original:
            raise ValueError('first runtime epoch differs from original publisher identity')
        if index == len(epochs) - 1 and reference != publisher['guardian_config']:
            raise ValueError('last runtime epoch must bind the current guardian configuration')
        if first <= batch:
            selected = runtime
        previous = first
    if selected is None:
        raise ValueError('teacher batch precedes its authorized runtime epochs')
    return selected


def _guardian_lookahead_slots(state):
    """Share the guardian's ordered active/completed source custody inventory."""
    try:
        from .teacher_generation_guardian import primary_ahead_slots
    except ImportError:
        from teacher_generation_guardian import primary_ahead_slots
    return primary_ahead_slots(state)


def _guardian_export_slot(state, batch):
    """Follow the same immutable slot through primary adoption and closure."""
    if type(batch) is not int:
        return None
    slot = _guardian_lookahead_slots(state).get(batch)
    if slot is not None:
        return slot
    launches = state.get('primary_launches') or {}
    if not isinstance(launches, dict):
        raise ValueError('guardian primary launch history is malformed')
    slot = launches.get(str(batch))
    if not isinstance(slot, dict) or slot.get('batch') != batch:
        return None
    active = state.get('active_batch')
    if active == batch:
        if slot.get('owner') != state.get('owner'):
            raise ValueError('adopted primary slot changed its exact owner')
        return slot
    frontier = active if type(active) is int else state.get('next_batch')
    closed = (state.get('source_sequence') or {}).get('closed') or {}
    if not isinstance(closed, dict):
        raise ValueError('guardian closed source history is malformed')
    if type(frontier) is int and frontier > batch and str(batch) in closed:
        return slot
    return None


def discover_guardian_batch(publisher: dict, sandbox,
                            read_file: Callable[[str], bytes], *, batch_slot: int | None = None) -> dict | None:
    """Derive a poll config only from the exact live guardian owner and sealed batch.

    The publisher file is owner-host authority. The remote guardian config,
    state, process identity, and source-sequence receipt are independently
    checked on every poll; a stale batch config cannot follow a new owner.
    The configured anchor is intentionally special: it is sealed by the
    original source-selection binding and has no ``started`` receipt.
    """
    if publisher.get('schema') != 'natlang.teacher-incremental-export-publisher/1':
        raise ValueError('automatic publisher schema differs')
    required = {'sandbox_id', 'sandbox_boot_id', 'guardian_config', 'control_dir',
                'generation_root', 'anchor_batch', 'anchor_cases_sha256',
                'directory_root', 'role', 'chunk_size', 'exporter_path', 'identity'}
    if not required <= publisher.keys():
        raise ValueError('automatic publisher lacks exact guardian and spool authority')
    if type(publisher['anchor_batch']) is not int or type(publisher['chunk_size']) is not int:
        raise ValueError('publisher anchor and chunk size must be integers')
    boot = _sandbox_boot_read(sandbox)
    if boot.returncode != 0 or boot.stdout.strip() != publisher['sandbox_boot_id']:
        raise ValueError('replacement teacher boot differs from publisher authority')

    guardian_ref = publisher['guardian_config']
    if not isinstance(guardian_ref, dict) or not {'path', 'sha256', 'bytes'} <= guardian_ref.keys():
        raise ValueError('publisher must bind exact remote guardian config bytes')
    guardian_raw = read_file(guardian_ref['path'])
    if len(guardian_raw) != guardian_ref['bytes'] or _sha(guardian_raw) != guardian_ref['sha256']:
        raise ValueError('remote guardian config differs from publisher binding')
    guardian = _json(guardian_raw, 'remote guardian config')
    if (guardian.get('control') != publisher['control_dir']
            or guardian.get('root') != publisher['generation_root']
            or not isinstance(guardian.get('source_sequence'), dict)):
        raise ValueError('guardian control, generation root, or source-sequence authority differs')
    source_sequence = guardian['source_sequence']

    control = publisher['control_dir']
    state_raw = read_file(f'{control}/state.json')
    state = _json(state_raw, 'guardian state')
    if state.get('schema') != 'natlang.teacher-generation-guardian/1':
        raise ValueError('active guardian state schema differs')
    config_binding = state.get('config')
    if config_binding != {'path': guardian_ref['path'], 'sha256': guardian_ref['sha256'],
                          'bytes': guardian_ref['bytes']}:
        raise ValueError('guardian state does not bind the publisher config revision')
    if state.get('primary_active_ahead')and guardian.get('primary_worker_capacity')is None:
        raise ValueError('publisher active tails lack an explicit worker envelope')
    batch, owner = state.get('active_batch'), state.get('owner')
    slot=None
    if batch_slot is not None:
        slot=_guardian_export_slot(state,batch_slot)
        if (guardian.get('primary_parallel_batches',1)!=2 or not isinstance(slot,dict)
                or type(batch_slot)is not int or slot.get('batch')!=batch_slot
                or type(batch_slot)is not int):
            raise ValueError('publisher requested an unowned next batch slot')
        batch,owner=batch_slot,slot.get('owner')
    elif batch is not None:
        slot=(state.get('primary_launches')or{}).get(str(batch))
    feeder=guardian['feeder'];phase_path=f"{publisher['generation_root']}/feed-evidence-phase.json"
    if slot is not None:
        original_ref=slot.get('guardian_config')
        if (not isinstance(original_ref,dict)or set(original_ref)!={'path','sha256','bytes'}
                or type(original_ref['bytes'])is not int or original_ref['bytes']<0
                or not isinstance(original_ref['sha256'],str)
                or re.fullmatch(r'[0-9a-f]{64}',original_ref['sha256'])is None
                or original_ref['path']!=f"{control}/config-history/{original_ref['sha256']}.json"):
            raise ValueError('batch slot lacks its immutable guardian configuration snapshot')
        original_raw=read_file(original_ref['path'])
        if len(original_raw)!=original_ref['bytes']or _sha(original_raw)!=original_ref['sha256']:
            raise ValueError('batch slot guardian configuration snapshot changed')
        original_guardian=_json(original_raw,'batch slot original guardian configuration')
        if (original_guardian.get('control')!=control or original_guardian.get('root')!=publisher['generation_root']
                or original_guardian.get('primary_parallel_batches')!=2
                or (original_guardian.get('teacher_identity')or{}).get('sha256')
                    !=guardian['teacher_identity'].get('sha256')
                or original_guardian.get('task_runtime')!=slot.get('task_runtime')):
            raise ValueError('batch slot guardian snapshot changed control, engine or runtime')
        context=slot.get('collection_context')
        if (not isinstance(context,dict)or set(context)!={'repo','queue_worker','node_bin'}
                or context.get('repo')!=original_guardian.get('collection_repo')
                or context.get('node_bin')!=original_guardian.get('node_bin')
                or not isinstance(context.get('queue_worker'),dict)
                or context['queue_worker'].get('path')!=original_guardian.get('queue_worker')
                or context['queue_worker']!=original_guardian.get('collection_worker')):
            raise ValueError('batch slot collection context changed from its original guardian')
        worker_ref=context['queue_worker'];worker_raw=read_file(worker_ref['path'])
        if len(worker_raw)!=worker_ref.get('bytes')or _sha(worker_raw)!=worker_ref.get('sha256'):
            raise ValueError('batch slot collection queue worker changed')
        if (slot.get('batch')!=batch or slot.get('feeder')!=original_guardian.get('overlap_feeder')
                or slot.get('phase_path')!=f"{publisher['generation_root']}/feeder-phases/batch-{batch:04d}.json"
                or slot.get('task_runtime')!=_publisher_batch_runtime(publisher,batch,read_file)):
            raise ValueError('batch slot changed feeder, phase or runtime authority')
        ref=slot['feeder'];raw=read_file(ref['path'])
        if len(raw)!=ref['bytes']or _sha(raw)!=ref['sha256']:
            raise ValueError('batch slot feeder bytes changed')
        sequence=state.get('source_sequence')or{}
        pending=(sequence.get('prefetch')if batch_slot is not None else sequence.get('pending'))
        if not pending or pending.get('batch')!=batch or pending.get('intent')!=slot.get('selection_intent'):
            # An already closed primary retains its immutable started intent.
            started=sequence.get('started',{}).get(str(batch))
            if not started:raise ValueError('batch slot lost its actual selected source')
            data=read_file(started['path'])
            if len(data)!=started['bytes']or _sha(data)!=started['sha256']or _json(data,'batch slot started')['selection_intent']!=slot.get('selection_intent'):
                raise ValueError('batch slot selection changed')
        intent_ref=slot['selection_intent'];raw=read_file(intent_ref['path'])
        if len(raw)!=intent_ref['bytes']or _sha(raw)!=intent_ref['sha256']:
            raise ValueError('batch slot selection intent changed')
        intent=_json(raw,'batch slot intent')
        if (intent.get('batch')!=batch or intent.get('new_configuration')!=slot.get('source_configuration')
                or intent.get('prefetch',{}).get('task_runtime')!=slot['task_runtime']):
            raise ValueError('batch slot immutable source authority differs')
        feeder=ref['path'];phase_path=slot['phase_path']
    closed_refs = {}; closed_receipts = {}
    sequence_state = state.get('source_sequence') or {}
    for key, closed_ref in (sequence_state.get('closed') or {}).items():
        if not isinstance(closed_ref, dict) or not {'path', 'sha256', 'bytes'} <= closed_ref.keys():
            raise ValueError('guardian closed-sequence history has an incomplete receipt binding')
        closed_raw = read_file(closed_ref['path'])
        if len(closed_raw) != closed_ref['bytes'] or _sha(closed_raw) != closed_ref['sha256']:
            raise ValueError('guardian closed-sequence history receipt changed')
        closed_receipt = _json(closed_raw, 'guardian closed-sequence receipt')
        boundary_ref = closed_receipt.get('boundary')
        if (str(closed_receipt.get('batch')) != key or not isinstance(boundary_ref, dict)
                or not {'path', 'sha256', 'bytes'} <= boundary_ref.keys()
                or boundary_ref['path'] != f"{publisher['control_dir']}/boundaries/batch-{int(key):04d}.json"):
            raise ValueError('guardian closed-sequence history does not bind its exact boundary file')
        closed_refs[key] = boundary_ref
        closed_receipts[key] = closed_receipt
    if (batch_slot is not None and batch != state.get('active_batch')
            and batch not in _guardian_lookahead_slots(state)):
        started = sequence_state.get('started', {}).get(str(batch))
        if (not started or closed_receipts.get(str(batch), {}).get('started') != started):
            raise ValueError('historical primary slot lost its exact source closure')
    latest_boundary = state.get('last_boundary')
    if latest_boundary is not None:
        if not isinstance(latest_boundary, dict) or not {'path', 'sha256', 'bytes'} <= latest_boundary.keys():
            raise ValueError('guardian latest boundary lacks exact file binding')
        name = Path(latest_boundary['path']).name
        match = re.fullmatch(r'batch-([0-9]{4})\.json', name)
        if (not match or latest_boundary['path'] != f"{publisher['control_dir']}/boundaries/{name}"):
            raise ValueError('guardian latest boundary path is outside its control directory')
        closed_refs[match.group(1)] = latest_boundary
    closed_boundaries = [closed_refs[key] for key in sorted(closed_refs, key=int)]
    if batch is None:
        if owner is not None:
            raise ValueError('guardian has an owner without an active batch')
        if not closed_boundaries:
            return None
        return {'closed_boundaries': closed_boundaries, 'guardian_state_sha256': _sha(state_raw),
                'guardian_state': state, 'sandbox_id': publisher['sandbox_id']}
    if type(batch) is not int or batch < publisher['anchor_batch'] or not isinstance(owner, dict):
        raise ValueError('guardian active batch/owner is outside the authorized source sequence')
    if (owner.get('boot_id') != publisher['sandbox_boot_id']
            or not isinstance(owner.get('pid'), int) or not str(owner.get('start_ticks', '')).isdigit()):
        raise ValueError('guardian feeder owner lacks exact process/boot identity')

    # Check PID/start-time/boot and feeder argv inside the sandbox rather than
    # trusting a stale state file. Values are validated integers/strings above.
    probe = """import pathlib,json,sys
p=int(sys.argv[1]);ticks=sys.argv[2];boot=sys.argv[3];feeder=sys.argv[4];batch=sys.argv[5]
actual_boot=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
try:
 r=pathlib.Path('/proc')/str(p);s=(r/'stat').read_text().rsplit(')',1)[1].split()
 a=(r/'cmdline').read_bytes().split(b'\\0')[:-1];env=(r/'environ').read_bytes().split(b'\\0')
 actual={'pid':p,'start_ticks':s[19],'boot_id':actual_boot,'argv':[x.decode(errors='replace') for x in a]}
 birth=actual['start_ticks']==ticks and actual_boot==boot
 ok=s[0]!='Z' and birth and feeder in actual['argv'] and ('NATLANG_TEACHER_START_BATCH='+batch).encode() in env
 print(json.dumps({'valid':ok,'identity':actual,'exited':s[0]=='Z' and birth}))
except (FileNotFoundError,ProcessLookupError):
 print(json.dumps({'valid':False,'missing':True,'boot_id':actual_boot}))
"""
    checked = sandbox.exec(['python', '-c', probe, str(owner['pid']), str(owner['start_ticks']),
                            owner['boot_id'], feeder, str(batch)], timeout_seconds=20).result(timeout=30)
    process = (_json(checked.stdout.encode(), 'live guardian feeder probe')
               if checked.returncode == 0 else {'probe_failed': True})
    if process.get('valid') is not True:
        if process.get('missing') and process.get('boot_id') != publisher['sandbox_boot_id']:
            raise ValueError('missing guardian feeder probe changed its boot identity')
        if not any(process.get(key) is True for key in ('missing', 'exited', 'probe_failed')):
            raise ValueError('guardian feeder PID/start/boot/argv/batch differs from active state')
        if batch_slot is not None:
            fresh_raw=read_file(f'{control}/state.json');fresh=_json(fresh_raw,'prefetch transition state')
            if fresh.get('config')!=state.get('config')or read_file(guardian_ref['path'])!=guardian_raw:
                raise ValueError('prefetch transition config changed')
            current=_guardian_export_slot(fresh,batch)
            if current!=slot:
                adopted=(fresh.get('primary_launches')or{}).get(str(batch))
                if adopted!=slot:raise ValueError('prefetch owner changed without its exact adoption')
            refs=list(closed_boundaries)
            boundary_ref=slot.get('boundary')
            if boundary_ref is not None:
                data=read_file(boundary_ref['path'])
                if (boundary_ref['path']!=f'{control}/boundaries/batch-{batch:04d}.json'
                        or len(data)!=boundary_ref['bytes']or _sha(data)!=boundary_ref['sha256']
                        or _json(data,'prefetch boundary').get('batch')!=batch):
                    raise ValueError('prefetch terminal boundary changed')
                refs.append(boundary_ref)
            else:
                phase=_json(read_file(phase_path),'prefetch draining phase')
                if phase.get('phase')!='batch_complete'or phase.get('batch')!=batch:
                    raise ValueError('prefetch owner exited without an authenticated drained phase')
            return {'draining':True,'batch':batch,'guardian_state':fresh,
                'guardian_state_sha256':_sha(fresh_raw),'sandbox_id':publisher['sandbox_id'],
                'closed_boundaries':refs,'verified_owner_transition':boundary_ref is not None}
        fresh, fresh_raw, boundary_ref = _closed_guardian_transition(
            publisher, guardian_raw, state, batch, read_file,
            owner_exited=process.get('missing') is True or process.get('exited') is True,phase_path=phase_path)
        refs = {reference['path']: reference for reference in closed_boundaries}
        if boundary_ref is not None and boundary_ref['path'] in refs and refs[boundary_ref['path']] != boundary_ref:
            raise ValueError('guardian transition changed an already observed boundary binding')
        if boundary_ref is not None:
            refs[boundary_ref['path']] = boundary_ref
        return {'draining': True, 'batch': batch, 'guardian_state': fresh,
                'guardian_state_sha256': _sha(fresh_raw), 'sandbox_id': publisher['sandbox_id'],
                'closed_boundaries': list(refs.values()), 'verified_owner_transition': boundary_ref is not None}

    try:phase_raw=read_file(phase_path)
    except (KeyError,FileNotFoundError)as error:
        if batch!=publisher['anchor_batch']and not(state.get('source_sequence')or{}).get('started',{}).get(str(batch)):
            raise ValueError('non-anchor batch lacks its guardian source-sequence started receipt')from error
        raise
    phase=_json(phase_raw,'live teacher feed phase')
    preparing=phase.get('phase')=='waiting_strong_server_ready'and phase.get('batch')is None
    previous_complete=(phase.get('phase')=='batch_complete'and phase.get('batch')==batch-1)
    current_complete=(phase.get('phase')=='batch_complete'and phase.get('batch')==batch)
    if preparing or previous_complete or current_complete:
        fresh_raw=read_file(f'{control}/state.json');fresh=_json(fresh_raw,'guardian phase transition state')
        if (read_file('/proc/sys/kernel/random/boot_id').decode().strip()!=publisher['sandbox_boot_id']
                or read_file(guardian_ref['path'])!=guardian_raw
                or fresh.get('schema')!=state['schema']or fresh.get('config')!=config_binding):
            raise ValueError('guardian preparation changed boot or config authority')
        if slot is not None:
            latest=(_guardian_export_slot(fresh,batch)if batch_slot is not None else(fresh.get('primary_launches')or{}).get(str(batch)))
            if latest!=slot:raise ValueError('batch slot changed during feeder preparation')
        elif fresh.get('owner')!=owner or fresh.get('active_batch')!=batch:
            fresh,fresh_raw,reference=_closed_guardian_transition(
                publisher,guardian_raw,state,batch,read_file,owner_exited=False,phase_path=phase_path)
            refs={r['path']:r for r in closed_boundaries}
            if reference is not None:refs[reference['path']]=reference
            return {'draining':True,'batch':batch,'guardian_state':fresh,
                'guardian_state_sha256':_sha(fresh_raw),'sandbox_id':publisher['sandbox_id'],
                'closed_boundaries':list(refs.values()),'verified_owner_transition':reference is not None}
        if previous_complete:
            ref=fresh.get('last_boundary')
            if (not isinstance(ref,dict)or ref.get('path')!=f'{control}/boundaries/batch-{batch-1:04d}.json'):
                raise ValueError('new feeder preparation lacks its actual predecessor boundary')
            data=read_file(ref['path'])
            if len(data)!=ref.get('bytes')or _sha(data)!=ref.get('sha256')or _json(data,'preparation predecessor boundary').get('batch')!=batch-1:
                raise ValueError('new feeder preparation predecessor changed')
        return {'draining':True,'preparing':not current_complete,'batch':batch,'guardian_state':fresh,
            'guardian_state_sha256':_sha(fresh_raw),'sandbox_id':publisher['sandbox_id'],
            'closed_boundaries':closed_boundaries,'verified_owner_transition':False}

    source_root = f"{publisher['generation_root']}/evidence-collections-v2/batch-{batch:04d}"
    cases_path = f'{source_root}/source-cases.jsonl'
    queue_path = f'{source_root}/queue.jsonl'
    teacher_path = f'{source_root}/teacher-source.json'
    started = None
    expected_sealed = None
    if batch == publisher['anchor_batch']:
        if source_sequence.get('anchor') != {'batch': batch, 'cases_sha256': publisher['anchor_cases_sha256']}:
            raise ValueError('protected anchor source differs from its exact packet binding')
    else:
        started = (state.get('source_sequence') or {}).get('started', {}).get(str(batch))
        if not isinstance(started, dict) or not {'path', 'sha256', 'bytes'} <= started.keys():
            raise ValueError('non-anchor batch lacks its guardian source-sequence started receipt')
        receipt_raw = read_file(started['path'])
        if len(receipt_raw) != started['bytes'] or _sha(receipt_raw) != started['sha256']:
            raise ValueError('source-sequence started receipt differs from state binding')
        receipt = _json(receipt_raw, 'source-sequence started receipt')
        if receipt.get('batch') != batch:
            raise ValueError('source-sequence started receipt is for a different batch')
        expected_sealed = receipt.get('sealed') or {}
        expected_paths = {'source-cases.jsonl': cases_path, 'queue.jsonl': queue_path,
                          'teacher-source.json': teacher_path}
        if set(expected_sealed) != set(expected_paths) or any(
                expected_sealed[name].get('path') != path for name, path in expected_paths.items()):
            raise ValueError('source-sequence receipt names unexpected batch files')
    cases_raw = read_file(cases_path)
    queue_raw = read_file(queue_path)
    teacher_raw = read_file(teacher_path)
    cases_sha, queue_sha, teacher_sha = _sha(cases_raw), _sha(queue_raw), _sha(teacher_raw)
    teacher = _teacher_source_binding(teacher_raw, batch=batch, source_sha256=cases_sha)
    if batch == publisher['anchor_batch']:
        if cases_sha != publisher['anchor_cases_sha256']:
            raise ValueError('protected anchor source differs from its exact packet binding')
    else:
        actual = {'source-cases.jsonl': {'path': cases_path, 'bytes': len(cases_raw), 'sha256': cases_sha},
                  'queue.jsonl': {'path': queue_path, 'bytes': len(queue_raw), 'sha256': queue_sha},
                  'teacher-source.json': {'path': teacher_path, 'bytes': len(teacher_raw), 'sha256': teacher_sha}}
        if expected_sealed != actual:
            raise ValueError('active batch source bytes differ from its immutable started receipt')

    entries = _jsonl_rows(queue_raw, 'active teacher queue')
    if not entries:
        raise ValueError('active teacher queue is empty')
    execution_paths = _execution_paths(entries, source_root, cases_path)
    execution_context = {'source-cases.jsonl': cases_raw, 'teacher-source.json': teacher_raw,
                         **{name: read_file(path) for name, path in execution_paths.items()}}
    source_case_count, _ = _execution_source(execution_context, entries)
    if (teacher.get('actual_teacher_execution_identity', {}).get('sha256')
            != guardian['teacher_identity'].get('sha256')):
        raise ValueError('teacher source execution identity differs from active guardian config')
    runtime = _publisher_batch_runtime(publisher, batch, read_file)
    if publisher.get('runtime_epochs') is None and runtime != guardian['task_runtime']:
        raise ValueError('publisher runtime differs from active guardian config')
    if teacher.get('actual_task_runtime') != runtime:
        raise ValueError('teacher source runtime differs from active guardian config')
    worker_count = phase.get('worker_count')
    if phase.get('batch') != batch or type(worker_count) is not int or worker_count < 1:
        raise ValueError('live feed phase does not bind active batch and worker count')
    worker_pids = phase.get('worker_pids')
    output_root = phase.get('out')
    source_model = (teacher.get('teacher') or {}).get('model')
    if (type(phase.get('cases')) is not int or phase.get('cases') != source_case_count
            or not isinstance(source_model, str) or phase.get('teacher') != source_model
            or output_root != source_root
            or not isinstance(worker_pids, list) or len(worker_pids) != worker_count
            or any(type(pid) is not int or pid <= 0 for pid in worker_pids)
            or len(set(worker_pids)) != len(worker_pids)):
        raise ValueError('live feed phase lacks exact case, teacher, output-root or worker identities')
    for row in entries:
        for field in ('output', 'jobs', 'log'):
            path = row.get(field)
            if path is not None and (not isinstance(path, str)
                    or not Path(os.path.normpath(path)).is_relative_to(os.path.normpath(output_root))):
                raise ValueError('queue worker output path escapes the exact feed output root')
    list_code = ("import pathlib,json,sys; p=pathlib.Path(sys.argv[1]); "
        "n=lambda x:int(x.rsplit('-',1)[1].split('.',1)[0]); "
        "q=sorted((x.name for x in p.glob('queue-worker-*.jsonl')),key=n); "
        "j=sorted((x.name for x in p.glob('journal-worker-*.jsonl')),key=n); "
        "print(json.dumps({'queues':q,'journals':j}))")
    listed = sandbox.exec(['python', '-c', list_code, source_root], timeout_seconds=20).result(timeout=30)
    if listed.returncode != 0:
        raise ValueError('could not list active worker queues and journals')
    workers = _json(listed.stdout.encode(), 'active worker file listing')
    expected_queue_names = [f'queue-worker-{index}.jsonl' for index in range(worker_count)]
    expected_journal_names = [f'journal-worker-{index}.jsonl' for index in range(worker_count)]
    if workers.get('queues') != expected_queue_names or not set(workers.get('journals', [])) <= set(expected_journal_names):
        raise ValueError('live worker queue/journal files differ from declared contiguous worker set')
    worker_queue_paths = [f'{source_root}/{name}' for name in expected_queue_names]
    worker_journal_paths = {f'journal-worker-{index}.jsonl':
        (f'{source_root}/journal-worker-{index}.jsonl' if f'journal-worker-{index}.jsonl' in workers['journals'] else None)
        for index in range(worker_count)}
    identity = {'sandbox_boot_id': publisher['sandbox_boot_id'],
        'teacher_source_sha256': teacher_sha, 'source_cases_sha256': cases_sha,
        'actor_sha256': teacher['actor']['sha256'],
        'execution_identity_sha256': teacher['actual_teacher_execution_identity']['sha256'],
        'teacher_config_sha256': teacher['teacher_configuration_source']['sha256'],
        'runtime_path': runtime}
    expected_identity = {**publisher['identity'], 'runtime_path': runtime}
    stable_fields = {'sandbox_boot_id', 'actor_sha256', 'execution_identity_sha256',
                     'teacher_config_sha256', 'runtime_path'}
    if (not isinstance(expected_identity, dict) or not stable_fields <= expected_identity.keys()
            or any(identity.get(key) != expected_identity[key] for key in stable_fields)):
        raise ValueError('active actor/config/runtime identity differs from publisher authority')
    return {'sandbox_id': publisher['sandbox_id'], 'batch': batch,
        'queue_path': queue_path, 'journal_path': f'{source_root}/journal.jsonl',
        'context_paths': {'teacher-source.json': teacher_path,
                          'source-cases.jsonl': cases_path, **execution_paths},
        'directory': f"{publisher['directory_root']}/batch-{batch:04d}",
        'role': publisher['role'], 'chunk_size': publisher['chunk_size'],
        'exporter_path': publisher['exporter_path'], 'identity': identity,
        'worker_queue_paths': worker_queue_paths, 'worker_journal_paths': worker_journal_paths,
        'queue_sha256': queue_sha, 'guardian_state_sha256': _sha(state_raw),
        'source_sequence_started_sha256': None if batch == publisher['anchor_batch'] else started['sha256'],
        'guardian_state': state, 'control_dir': control, 'closed_boundaries': closed_boundaries}


def export_guardian_closed_boundary(publisher: dict, active: dict, *, sandbox,
                                    read_file: Callable[[str], bytes], spool_root: Path) -> dict:
    """Seal and privately export the guardian's exact drained boundary.

    This preserves failed and unstarted work as repair/control data. It is
    deliberately separate from tranche completion and never grants admission.
    """
    reference = active.get('closed_boundary')
    state = active.get('guardian_state') or {}
    if not isinstance(reference, dict) or not {'path', 'sha256', 'bytes'} <= reference.keys():
        raise ValueError('guardian closed boundary lacks exact file binding')
    boundary_raw = read_file(reference['path'])
    if len(boundary_raw) != reference['bytes'] or _sha(boundary_raw) != reference['sha256']:
        raise ValueError('guardian closed boundary differs from its state binding')
    boundary = _json(boundary_raw, 'guardian closed boundary')
    batch = boundary.get('batch')
    if (type(batch) is not int or batch < publisher['anchor_batch']
            or reference['path'] != f"{publisher['control_dir']}/boundaries/batch-{batch:04d}.json"):
        raise ValueError('guardian boundary path or batch is outside the authorized sequence')
    completion = boundary.get('completion') or {}
    if completion.get('schema') != 'natlang.strong-teacher-batch-completion/1':
        raise ValueError('guardian boundary lacks exact batch completion accounting')
    started_receipt = None
    if batch > publisher['anchor_batch']:
        started = (state.get('source_sequence') or {}).get('started', {}).get(str(batch))
        if not isinstance(started, dict) or not {'path', 'sha256', 'bytes'} <= started.keys():
            raise ValueError('closed non-anchor batch lacks its started source-sequence receipt')
        started_raw = read_file(started['path'])
        if len(started_raw) != started['bytes'] or _sha(started_raw) != started['sha256']:
            raise ValueError('closed batch started receipt differs from guardian state')
        started_receipt = _json(started_raw, 'closed batch source sequence')
        if started_receipt.get('batch') != batch:
            raise ValueError('closed batch started receipt names another batch')

    source_root = f"{publisher['generation_root']}/evidence-collections-v2/batch-{batch:04d}"
    control = publisher['control_dir']
    queue_path = f'{source_root}/queue.jsonl'
    queue_raw = read_file(queue_path)
    source_cases = f'{source_root}/source-cases.jsonl'
    teacher_source = f'{source_root}/teacher-source.json'
    cases_raw, teacher_raw = read_file(source_cases), read_file(teacher_source)
    teacher = _teacher_source_binding(teacher_raw, batch=batch, source_sha256=_sha(cases_raw))
    authority = {**(publisher.get('identity') or {}),
                 'runtime_path': _publisher_batch_runtime(publisher, batch, read_file)}
    if (teacher['actor'].get('sha256') != authority.get('actor_sha256')
            or teacher['actual_teacher_execution_identity'].get('sha256') != authority.get('execution_identity_sha256')
            or teacher['teacher_configuration_source'].get('sha256') != authority.get('teacher_config_sha256')
            or teacher.get('actual_task_runtime') != authority.get('runtime_path')):
        raise ValueError('closed batch actor, execution, configuration or runtime differs from publisher authority')
    bindings = boundary.get('bindings') or {}
    for name, raw, path in (('queue.jsonl', queue_raw, queue_path),
                            ('teacher-source.json', teacher_raw, teacher_source)):
        binding = bindings.get(name)
        if not isinstance(binding, dict) or binding.get('path') != path or binding.get('bytes') != len(raw) or binding.get('sha256') != _sha(raw):
            raise ValueError(f'closed guardian boundary does not bind exact {name} source')
    if batch == publisher['anchor_batch']:
        if _sha(cases_raw) != publisher['anchor_cases_sha256']:
            raise ValueError('closed anchor source differs from the exact publisher packet')
    else:
        sealed = (started_receipt or {}).get('sealed') or {}
        case_binding = sealed.get('source-cases.jsonl')
        if (case_binding != {'path': source_cases, 'bytes': len(cases_raw), 'sha256': _sha(cases_raw)}
                or sealed.get('queue.jsonl') != {'path': queue_path, 'bytes': len(queue_raw), 'sha256': _sha(queue_raw)}
                or sealed.get('teacher-source.json') != {'path': teacher_source, 'bytes': len(teacher_raw), 'sha256': _sha(teacher_raw)}):
            raise ValueError('closed batch files differ from its exact source-sequence seal')

    journal_path = f'{source_root}/journal.jsonl'
    score_path = f'{source_root}/score.json'
    rewards_path = f'{source_root}/rewards.jsonl'
    repair_queue_path = f'{control}/repairs/batch-{batch:04d}/queue.jsonl'
    queue_rows = _jsonl_rows(queue_raw, 'closed guardian queue')
    execution_paths = _execution_paths(queue_rows, source_root, source_cases)
    _execution_source({'source-cases.jsonl': cases_raw, 'teacher-source.json': teacher_raw,
                       **{name: read_file(path) for name, path in execution_paths.items()}}, queue_rows)
    queue_keys = [row.get('key') for row in queue_rows]
    outcomes = boundary.get('individual_outcomes') or []
    outcome_by_key = {row.get('key'): row for row in outcomes if isinstance(row, dict)}
    if len(outcome_by_key) != len(outcomes) or set(outcome_by_key) != set(queue_keys):
        raise ValueError('closed guardian outcomes do not cover the exact queue')
    journal_raw = read_file(journal_path)
    journal_rows = _jsonl_rows(journal_raw, 'closed guardian journal')
    started_keys = {row.get('key') for row in journal_rows if row.get('event') == 'start'}
    complete_keys = set()
    for key, outcome in outcome_by_key.items():
        finish = outcome.get('finish')
        states = ((finish or {}).get('output_accounting') or {}).get('job_states') or []
        if (finish and (finish.get('output_accounting') or {}).get('complete') is True
                and states and all(row.get('state') in ('result', 'generation_held') for row in states)):
            complete_keys.add(key)
    unfinished = set(outcome_by_key) - {key for key, row in outcome_by_key.items() if row.get('finish') is not None}
    unstarted_keys = unfinished - started_keys
    reported_repairs = {row.get('key') for row in completion.get('repair_required') or []
                        if isinstance(row, dict) and isinstance(row.get('key'), str)}
    repair_keys = sorted(reported_repairs - unstarted_keys)
    exported_keys = [row['key'] for row in queue_rows if row['key'] in complete_keys]
    unstarted_ordered = [row['key'] for row in queue_rows if row['key'] in unstarted_keys]
    # The guardian creates this queue before publishing last_boundary. If it
    # has no repairs, an absent file is represented by the exact empty queue.
    exists = sandbox.exec(['test', '-f', repair_queue_path], timeout_seconds=20).result(timeout=30)
    if exists.returncode not in (0, 1):
        raise OSError('could not check exact guardian repair-queue snapshot')
    if exists.returncode == 0:
        repair_queue_raw = read_file(repair_queue_path)
    elif repair_keys:
        raise ValueError('guardian boundary has repairs but no durable repair queue')
    else:
        repair_queue_raw = b''
    files = {'teacher-source.json': teacher_source, 'source-cases.jsonl': source_cases,
        'queue.jsonl': queue_path, 'journal.jsonl': journal_path, 'score.json': score_path,
        'rewards.jsonl': rewards_path, 'completion.json': reference['path'],
        'repair-queue.jsonl': repair_queue_path, **execution_paths}
    for name, binding in _repair_payload_bindings(boundary).items():
        files[name] = binding['source_path']
    for name, binding in _budget_payload_bindings(boundary, queue_rows).items():
        files[name] = binding['source_path']
    # Keep the no-repair empty snapshot local to this call; read_file remains
    # the scoped remote reader for all source-bound payloads.
    def boundary_reader(path: str) -> bytes:
        if path == repair_queue_path and exists.returncode == 1:
            return repair_queue_raw
        return read_file(path)
    directory = Path(publisher['directory_root']) / f'batch-{batch:04d}' / 'final-boundary'
    batch_directory = Path(publisher['directory_root']) / f'batch-{batch:04d}'
    def export_file(tranche: dict, receipt_path: Path) -> dict:
        return export_private_tranche(tranche, exporter=Path(publisher['exporter_path']),
            role=publisher['role'], receipt_path=receipt_path)
    # A batch may finish with fewer than the normal tranche size remaining.
    # Flush those exact completed rows against the guardian's bound final
    # journal before sealing the repair/unstarted control boundary.
    export_finished_tranches(batch=batch, queue_path=queue_path, journal_path=journal_path,
        context_paths={'teacher-source.json': teacher_source, 'source-cases.jsonl': source_cases,
                       **execution_paths},
        directory=batch_directory, read_file=boundary_reader, export_file=export_file,
        chunk_size=publisher['chunk_size'], flush_partial=True,
        prune_after_private_verification=True, queue_snapshot=queue_raw,
        journal_snapshot=journal_raw)
    export_final_boundary(batch=batch, queue_path=queue_path, exported_keys=exported_keys,
        repair_keys=repair_keys, unstarted_keys=unstarted_ordered, file_paths=files,
        read_file=boundary_reader, directory=directory,
        exporter=Path(publisher['exporter_path']), role=publisher['role'])
    return {'batch': batch, 'directory': str(directory), 'boundary_sha256': reference['sha256'],
            'private': json.loads((directory/'boundary-private-export.json').read_text()).get('private_export'),
            'batch_can_advance': completion.get('can_advance') is True,
            'training_admission': False, 'exposure': False}


def run_incremental_service(config_path: Path, *, poll_seconds: float = 15,
                            max_polls: int | None = None,
                            allowed_spool_root: Path | None = None) -> dict:
    """Continuously follow an atomically replaced active-batch config.

    Each poll rebinds the sandbox boot, source packet, teacher actor/config and
    runtime before exporting. Transport errors are recorded and retried; a
    mismatch in any armed identity fails closed so a restarted/repointed
    teacher cannot inherit the prior batch's frontier.
    """
    if poll_seconds < 0 or (max_polls is not None and max_polls < 1):
        raise ValueError('poll interval and poll count must be nonnegative/positive')
    config_path = Path(config_path).resolve(strict=True)
    allowed_spool_root = (Path(allowed_spool_root).resolve(strict=True) if allowed_spool_root is not None
                          else _external_data_root())
    prior_config_sha = None
    completed_polls = 0
    last = None
    last_completed_sha = None
    last_completed_change = time.monotonic()
    last_batch_identity = None
    while max_polls is None or completed_polls < max_polls:
        config_raw = config_path.read_bytes()
        config = _json(config_raw, 'continuous teacher poll config')
        if not isinstance(config, dict):
            raise ValueError('continuous teacher poll config must be an object')
        required = {'sandbox_id', 'batch', 'queue_path', 'journal_path', 'context_paths',
                    'directory', 'role', 'chunk_size', 'exporter_path', 'identity'}
        if not required <= set(config):
            raise ValueError('continuous poll config lacks active-batch export bindings')
        directory = Path(config['directory']).resolve()
        if not directory.is_relative_to(allowed_spool_root):
            raise ValueError('continuous teacher spool must live under the external data volume')
        state_path = directory / 'service-state.json'
        config_sha = _sha(config_raw)
        try:
            cwctl = _cwctl_module()
            cwctl.load_env_file()
            sandbox, read_file = scoped_sdk_session(config['sandbox_id'])
            identity = _validate_live_identity(config, sandbox, read_file)
            batch_identity = (config['sandbox_id'], config['batch'], identity['teacher_source_sha256'])
            queue_snapshot = read_file(config['queue_path'])
            journal_snapshot = read_file(config['journal_path'])
            journal_sha = _sha(journal_snapshot)
            completed_sha = _completed_outputs_sha256(queue_snapshot, journal_snapshot)
            now = time.monotonic()
            if batch_identity != last_batch_identity or completed_sha != last_completed_sha:
                last_completed_change = now
            last_batch_identity, last_completed_sha = batch_identity, completed_sha
            idle_after = float(config.get('idle_flush_after_seconds', 300))
            if idle_after < 0:
                raise ValueError('idle flush interval must be nonnegative')
            idle_flush = now - last_completed_change >= idle_after

            def export_file(tranche: dict, receipt_path: Path) -> dict:
                return export_private_tranche(tranche, exporter=Path(config['exporter_path']),
                    role=config['role'], receipt_path=receipt_path)

            result = export_finished_tranches(batch=config['batch'], queue_path=config['queue_path'],
                journal_path=config['journal_path'], context_paths=config['context_paths'],
                directory=directory, read_file=read_file, export_file=export_file,
                chunk_size=config['chunk_size'],
                flush_partial=bool(config.get('flush_partial', False)) or idle_flush,
                prune_after_private_verification=True, queue_snapshot=queue_snapshot,
                journal_snapshot=journal_snapshot)
            last = {'schema': 'natlang.teacher-incremental-owner-poller/1',
                    'at': time.time(), 'config_sha256': config_sha,
                    'config_changed': config_sha != prior_config_sha, 'identity': identity,
                    'journal_sha256': journal_sha, 'completed_outputs_sha256': completed_sha,
                    'completed_outputs_idle_seconds': now - last_completed_change,
                    'result': result, 'last_transport_error': None,
                    'training_admission': False, 'exposure': False}
        except (ValueError, KeyError, TypeError) as error:
            last = {'schema': 'natlang.teacher-incremental-owner-poller/1',
                    'at': time.time(), 'config_sha256': config_sha,
                    'config_changed': config_sha != prior_config_sha,
                    'identity_error': str(error), 'training_admission': False, 'exposure': False}
            _write(state_path, (json.dumps(last, sort_keys=True, indent=2) + '\n').encode())
            raise
        except Exception as error:
            last = {'schema': 'natlang.teacher-incremental-owner-poller/1',
                    'at': time.time(), 'config_sha256': config_sha,
                    'config_changed': config_sha != prior_config_sha,
                    'last_transport_error': f'{type(error).__name__}: {error}',
                    'training_admission': False, 'exposure': False}
        _write(state_path, (json.dumps(last, sort_keys=True, indent=2) + '\n').encode())
        prior_config_sha = config_sha
        completed_polls += 1
        if max_polls is None or completed_polls < max_polls:
            time.sleep(poll_seconds)
    return last or {}


def run_incremental_once(config_path: Path) -> dict:
    """Owner-host poll entrypoint; config has paths, never credentials."""
    config_path = Path(config_path).resolve(strict=True)
    config = json.loads(config_path.read_text())
    required = {'sandbox_id', 'batch', 'queue_path', 'journal_path', 'context_paths',
                'directory', 'role', 'chunk_size', 'exporter_path'}
    if not required <= set(config):
        raise ValueError('owner-host export config lacks required explicit paths/identity')
    cwctl = _cwctl_module()
    cwctl.load_env_file()
    reader = scoped_sdk_reader(config['sandbox_id'])

    def export_file(tranche: dict, receipt_path: Path) -> dict:
        return export_private_tranche(tranche, exporter=Path(config['exporter_path']),
                                      role=config['role'], receipt_path=receipt_path)

    return export_finished_tranches(batch=config['batch'], queue_path=config['queue_path'],
        journal_path=config['journal_path'], context_paths=config['context_paths'],
        directory=Path(config['directory']), read_file=reader, export_file=export_file,
        chunk_size=config['chunk_size'], flush_partial=bool(config.get('flush_partial', False)))


def _export_discovered_guardian_batch(active,prior,publisher,publisher_raw,sandbox,reader,export_unseen_closed):
    # Re-read all identity-bound bytes through the ordinary poll
    # validator immediately before queue/journal/output collection.
    identity = _validate_live_identity(active, sandbox, reader)
    directory = Path(active['directory']).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    active_identity = {'sandbox_id': publisher['sandbox_id'], 'batch': active['batch'],
                       'teacher_source_sha256': identity['teacher_source_sha256']}
    queue_snapshot = reader(active['queue_path'])
    if _sha(queue_snapshot) != active['queue_sha256']:
        raise ValueError('active queue changed after guardian discovery')
    if active['worker_queue_paths']:
        snapshots = _read_tranche_files(reader, [*active['worker_queue_paths'],
            *(path for path in active['worker_journal_paths'].values() if path is not None)])
        worker_queues = [(Path(path).name, snapshots[path]) for path in active['worker_queue_paths']]
        worker_journals = [(name, snapshots[path] if path is not None else None)
            for name, path in active['worker_journal_paths'].items()]
        journal_snapshot = aggregate_worker_journals(queue_snapshot, worker_queues, worker_journals)
    else:
        journal_snapshot = reader(active['journal_path'])
    journal_sha = _sha(journal_snapshot)
    prior_identity = prior.get('active_identity')
    if isinstance(prior_identity, list) and len(prior_identity) == 3:
        # Compatibility with the earlier tuple serialized by JSON.
        prior_identity = {'sandbox_id': prior_identity[0], 'batch': prior_identity[1],
                          'teacher_source_sha256': prior_identity[2]}
    completed_sha = _completed_outputs_sha256(queue_snapshot, journal_snapshot)
    changed = active_identity != prior_identity or completed_sha != prior.get('completed_outputs_sha256')
    changed_at = time.time() if changed else float(prior.get('completed_outputs_changed_at', time.time()))
    if (active_identity == prior_identity and 'completed_outputs_sha256' not in prior
            and journal_sha == prior.get('journal_sha256')):
        changed_at = float(prior.get('journal_changed_at', time.time()))
    journal_changed_at = (time.time() if journal_sha != prior.get('journal_sha256')
                          else float(prior.get('journal_changed_at', time.time())))
    idle_flush = time.time() - changed_at >= float(publisher.get('idle_flush_after_seconds', 300))

    def export_file(tranche: dict, receipt_path: Path) -> dict:
        return export_private_tranche(tranche, exporter=Path(publisher['exporter_path']),
            role=publisher['role'], receipt_path=receipt_path)

    result = export_finished_tranches(batch=active['batch'],
        queue_path=active['queue_path'], journal_path=active['journal_path'],
        context_paths=active['context_paths'], directory=directory,
        read_file=reader, export_file=export_file, chunk_size=active['chunk_size'],
        flush_partial=bool(publisher.get('flush_partial', False)) or idle_flush,
        prune_after_private_verification=True, queue_snapshot=queue_snapshot,
        journal_snapshot=journal_snapshot)
    closed = export_unseen_closed(active.get('closed_boundaries'), active.get('guardian_state') or {},
                                  active_batch=active['batch'])
    last = {'schema': 'natlang.teacher-incremental-export-publisher-state/1',
        'at': time.time(), 'publisher_sha256': _sha(publisher_raw),
        'active': True, 'active_identity': active_identity,
        'guardian_state_sha256': active['guardian_state_sha256'],
        'source_sequence_started_sha256': active['source_sequence_started_sha256'],
        'journal_sha256': journal_sha, 'journal_changed_at': journal_changed_at,
        'completed_outputs_sha256': completed_sha, 'completed_outputs_changed_at': changed_at,
        'identity': identity, 'result': result,
        'closed_exports_this_poll': closed,
        'training_admission': False, 'exposure': False}
    return last


def run_guardian_export_service(publisher_path: Path, *, poll_seconds: float = 15,
                                max_polls: int | None = None,
                                allowed_spool_root: Path | None = None) -> dict:
    """Poll the current guardian owner and follow its sealed active batch.

    This is the automatic publisher path: it does not require an operator to
    rewrite a batch config after every guardian transition. It only exports
    live finished tranches; batch close/accounting remains a separate receipt
    and never implies training admission.
    """
    if poll_seconds < 0 or (max_polls is not None and max_polls < 1):
        raise ValueError('poll interval and poll count must be nonnegative/positive')
    publisher_path = Path(publisher_path).resolve(strict=True)
    publisher_raw = publisher_path.read_bytes()
    publisher = _json(publisher_raw, 'teacher export publisher')
    if not isinstance(publisher, dict):
        raise ValueError('teacher export publisher must be a JSON object')
    spool_root = (Path(allowed_spool_root).resolve(strict=True) if allowed_spool_root is not None
                  else _external_data_root())
    state_path = Path(publisher['directory_root']).resolve() / 'publisher-state.json'
    root = Path(publisher['directory_root']).resolve()
    if not root.is_relative_to(spool_root):
        raise ValueError('teacher export publisher spool must live under external data root')
    polls, last = 0, None
    while max_polls is None or polls < max_polls:
        try:
            cwctl = _cwctl_module()
            cwctl.load_env_file()
            sandbox, reader = scoped_sdk_session(publisher['sandbox_id'])
            active = discover_guardian_batch(publisher, sandbox, reader)
            prior = json.loads(state_path.read_text()) if state_path.exists() else {}
            closed_exports = dict(prior.get('closed_boundaries') or {})
            if any(not str(key).isdigit() or not isinstance(value, dict)
                   or not isinstance(value.get('sha256'), str)
                   for key, value in closed_exports.items()):
                raise ValueError('publisher durable closed-boundary map is malformed')

            def export_unseen_closed(boundary_refs, guardian_state, *, active_batch=None):
                exported = []
                for reference in boundary_refs or []:
                    path = reference.get('path', '') if isinstance(reference, dict) else ''
                    match = re.fullmatch(r'batch-([0-9]{4})\.json', Path(path).name)
                    if not match:
                        raise ValueError('closed-boundary backlog contains an unexpected path')
                    batch_key = match.group(1)
                    if active_batch is not None and int(batch_key) >= active_batch:
                        continue
                    previous = closed_exports.get(batch_key)
                    if previous is not None:
                        if previous.get('sha256') != reference.get('sha256'):
                            raise ValueError('guardian rewrote a previously exported closed boundary')
                        continue
                    closed = export_guardian_closed_boundary(publisher,
                        {'closed_boundary': reference, 'guardian_state': guardian_state},
                        sandbox=sandbox, read_file=reader, spool_root=spool_root)
                    closed_exports[batch_key] = {'sha256': reference['sha256'], 'result': closed}
                    exported.append(closed)
                return exported

            if active is None or active.get('batch') is None:
                closed = export_unseen_closed(
                    active.get('closed_boundaries') if active else [],
                    active.get('guardian_state', {}) if active else {})
                last = {'schema': 'natlang.teacher-incremental-export-publisher-state/1',
                    'at': time.time(), 'publisher_sha256': _sha(publisher_raw),
                    'active': False, 'reason': 'guardian has no active feeder/batch',
                    'closed_exports_this_poll': closed,
                    'training_admission': False, 'exposure': False}
            elif active.get('draining') is True:
                closed = export_unseen_closed(active.get('closed_boundaries'), active.get('guardian_state') or {},
                                              active_batch=active['batch'])
                last = {'schema': 'natlang.teacher-incremental-export-publisher-state/1',
                    'at': time.time(), 'publisher_sha256': _sha(publisher_raw),
                    'active': False, 'draining_batch': active['batch'],
                    'guardian_state_sha256': active['guardian_state_sha256'],
                    'reason': ('authenticated feeder preparation; no new export custody'if active.get('preparing')else'feeder boundary complete; waiting for guardian last_boundary publication'),
                    'closed_exports_this_poll': closed,
                    'training_admission': False, 'exposure': False}
            else:
                batch_prior=(prior.get('overlap_batches')or{}).get(str(active['batch']),prior)
                last=_export_discovered_guardian_batch(active,batch_prior,publisher,publisher_raw,sandbox,reader,export_unseen_closed)
            overlap_results={}
            state=active.get('guardian_state',{})if active else{}
            for slot in _guardian_lookahead_slots(state).values():
                if not(isinstance(slot.get('owner'),dict)
                        and str(slot.get('batch'))in(state.get('source_sequence')or{}).get('started',{})):
                    continue
                extra=discover_guardian_batch(publisher,sandbox,reader,batch_slot=slot['batch'])
                extra_prior=(prior.get('overlap_batches')or{}).get(str(slot['batch']),{})
                if extra.get('draining'):
                    exported=export_unseen_closed(extra.get('closed_boundaries'),extra['guardian_state'])
                    extra_result={'active':False,'draining_batch':slot['batch'],'closed_exports_this_poll':exported,
                        'guardian_state_sha256':extra['guardian_state_sha256'],'training_admission':False,'exposure':False}
                else:
                    extra_result=_export_discovered_guardian_batch(extra,extra_prior,publisher,publisher_raw,sandbox,reader,export_unseen_closed)
                overlap_results[str(slot['batch'])]=extra_result
            if overlap_results:last['overlap_batches']=overlap_results
            # Persist every verified boundary, not just the most recent one.
            # This makes restarts and slow polls recover all completed tails.
            last['closed_boundaries'] = closed_exports
        except (ValueError, KeyError, TypeError) as error:
            last = {'schema': 'natlang.teacher-incremental-export-publisher-state/1',
                'at': time.time(), 'publisher_sha256': _sha(publisher_raw),
                'identity_error': f'{type(error).__name__}: {error}',
                'closed_boundaries': locals().get('closed_exports', {}),
                'training_admission': False, 'exposure': False}
            _write(state_path, (json.dumps(last, sort_keys=True, indent=2) + '\n').encode())
            raise
        except Exception as error:
            last = {'schema': 'natlang.teacher-incremental-export-publisher-state/1',
                'at': time.time(), 'publisher_sha256': _sha(publisher_raw),
                'transport_error': f'{type(error).__name__}: {error}',
                'closed_boundaries': locals().get('closed_exports', {}),
                'training_admission': False, 'exposure': False}
        _write(state_path, (json.dumps(last, sort_keys=True, indent=2) + '\n').encode())
        polls += 1
        if max_polls is None or polls < max_polls:
            time.sleep(poll_seconds)
    return last or {}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path,
                        help='owner-host JSON with the exact sandbox, batch, context and spool paths')
    parser.add_argument('--continuous', action='store_true',
                        help='reload the active-batch config each poll and require exact live identity bindings')
    parser.add_argument('--follow-guardian', action='store_true',
                        help='discover and follow only the exact live guardian batch from a publisher authority file')
    parser.add_argument('--poll-seconds', type=float, default=15)
    args = parser.parse_args()
    if args.continuous and args.follow_guardian:
        parser.error('choose --continuous or --follow-guardian')
    result = (run_guardian_export_service(args.config, poll_seconds=args.poll_seconds)
              if args.follow_guardian else run_incremental_service(args.config, poll_seconds=args.poll_seconds)
              if args.continuous else run_incremental_once(args.config))
    print(json.dumps(result, sort_keys=True, indent=2))
