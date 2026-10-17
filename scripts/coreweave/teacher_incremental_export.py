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
        if not isinstance(key, str) or not re.fullmatch(r'[A-Za-z0-9._-]+', key):
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
                or manifest.get('source_sha256') != input_identity['source-cases.jsonl']['sha256']
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
            context = {'teacher-source.json': read_file(context_paths['teacher-source.json']),
                       'queue.jsonl': queue_raw,
                       'source-cases.jsonl': read_file(context_paths['source-cases.jsonl'])}
            for name, path in context_paths.items():
                if name in ('teacher-source.json', 'source-cases.jsonl', 'queue.jsonl'):
                    continue
                context[name] = read_file(path)
            outputs = {}
            for row in selected:
                outputs[row['key']] = (read_file(row['output']), read_file(row['output'] + '.manifest.json'))
            held = {}
            for job in row['finish_event']['output_accounting']['job_states']:
                if job.get('state') == 'generation_held':
                    path = f"{row['jobs'].rstrip('/')}/{job['index']:06d}.error.json"
                    held[f"held-errors/{job['index']:06d}.error.json"] = read_file(path)
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
    groups = [exported_keys, repair_keys, unstarted_keys]
    flat = [key for group in groups for key in group]
    if len(flat) != len(set(flat)) or set(flat) != set(keys):
        raise ValueError('durable, repair and unstarted keys do not partition the original queue')
    if not {'teacher-source.json', 'source-cases.jsonl', 'queue.jsonl', 'journal.jsonl', 'score.json',
            'rewards.jsonl', 'completion.json', 'repair-queue.jsonl'} <= set(files):
        raise ValueError('final boundary is missing canonical source, accounting or repair control')
    if files['queue.jsonl'] != queue_raw:
        raise ValueError('final boundary queue bytes differ from original queue binding')
    completion = _json(files['completion.json'], 'completion')
    _teacher_source_binding(files['teacher-source.json'], batch=batch,
                            source_sha256=_sha(files['source-cases.jsonl']))
    if completion.get('schema') != 'natlang.teacher-generation-boundary/1' or completion.get('batch') != batch:
        raise ValueError('final boundary does not bind the actual teacher boundary schema and batch')
    verified = completion.get('completion') or {}
    if verified.get('schema') != 'natlang.strong-teacher-batch-completion/1':
        raise ValueError('final boundary lacks the shared batch-completion proof')
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
        else:
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
    if (set(exported_keys) != complete_keys or set(repair_keys) != required_repairs
            or repair_actual != required_repairs or set(unstarted_keys) != unstarted_expected):
        raise ValueError('final boundary key partitions differ from actual per-key batch accounting')
    repair_payloads = _repair_payload_bindings(completion)
    base_files = {'teacher-source.json', 'source-cases.jsonl', 'queue.jsonl', 'journal.jsonl',
                  'score.json', 'rewards.jsonl', 'completion.json', 'repair-queue.jsonl'}
    if set(files) - base_files != set(repair_payloads):
        raise ValueError('final boundary must retain every exact terminal-error payload needed for repairs')
    for name, binding in repair_payloads.items():
        if _sha(files[name]) != binding['sha256']:
            raise ValueError('final boundary repair payload bytes differ from collector binding')
    folder = Path(directory); folder.mkdir(parents=True, exist_ok=True)
    manifest = {'schema': 'natlang.teacher-incremental-export-boundary/1', 'batch': batch,
        'queue_sha256': _sha(queue_raw), 'queue_count': len(keys), 'exported_keys': exported_keys,
        'repair_keys': repair_keys, 'unstarted_keys': unstarted_keys,
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


def scoped_sdk_session(sandbox_id: str, *, timeout: int = 180):
    """Bind one scoped provider session and its read-only file reader."""
    cwctl = _cwctl_module()
    sandbox = cwctl.read_sandbox(sandbox_id, patience=60)

    def read(path: str) -> bytes:
        result = sandbox.read_file(path).result(timeout=timeout)
        return result if isinstance(result, bytes) else bytes(result)
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
    boot = sandbox.exec(['cat', '/proc/sys/kernel/random/boot_id'], timeout_seconds=30).result(timeout=45)
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
    last_journal_sha = None
    last_journal_change = time.monotonic()
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
            now = time.monotonic()
            if batch_identity != last_batch_identity or journal_sha != last_journal_sha:
                last_journal_change = now
            last_batch_identity, last_journal_sha = batch_identity, journal_sha
            idle_after = float(config.get('idle_flush_after_seconds', 300))
            if idle_after < 0:
                raise ValueError('idle flush interval must be nonnegative')
            idle_flush = now - last_journal_change >= idle_after

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


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path,
                        help='owner-host JSON with the exact sandbox, batch, context and spool paths')
    parser.add_argument('--continuous', action='store_true',
                        help='reload the active-batch config each poll and require exact live identity bindings')
    parser.add_argument('--poll-seconds', type=float, default=15)
    args = parser.parse_args()
    result = (run_incremental_service(args.config, poll_seconds=args.poll_seconds)
              if args.continuous else run_incremental_once(args.config))
    print(json.dumps(result, sort_keys=True, indent=2))
