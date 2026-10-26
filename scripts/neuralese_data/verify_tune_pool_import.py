#!/usr/bin/env python3
"""Verify Luna tune-pool result custody before the tuner imports any turns.

This reuses the reviewed dispatcher's plan, source-inventory and append-only
claim-ledger validators. It binds each imported result to its exact source
case, closed claim, collector output manifest, actor and file bytes. It does
not decide whether a result is useful training data.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

REPOSITORY_ROOT = str(Path(__file__).resolve().parents[2])
SCRIPT_ROOT = str(Path(__file__).resolve().parents[1])
TRAINING_PACKAGE_ROOT = str(Path(REPOSITORY_ROOT) / 'training/neuralese')
for entry in (REPOSITORY_ROOT, SCRIPT_ROOT, TRAINING_PACKAGE_ROOT):
    if entry not in sys.path:
        sys.path.insert(0, entry)


SCHEMA = 'natlang.tune-pool-import-custody/2'


def _sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_sha(value) -> str:
    return _sha_bytes(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8'))


def _binding(path: Path, payload: bytes | None = None) -> dict:
    resolved = path.resolve(strict=True)
    data = resolved.read_bytes() if payload is None else payload
    return {'path': str(resolved), 'bytes': len(data), 'sha256': _sha_bytes(data)}


def _stat_observation(path: Path) -> dict:
    value = path.stat()
    return {'device': value.st_dev, 'inode': value.st_ino, 'bytes': value.st_size,
            'mtime_ns': value.st_mtime_ns, 'ctime_ns': value.st_ctime_ns}


def _stable_read(path: Path, label: str) -> tuple[bytes, dict]:
    path = path.resolve(strict=True)
    for _ in range(4):
        before = _stat_observation(path)
        data = path.read_bytes()
        after = _stat_observation(path)
        if before == after and len(data) == after['bytes']:
            return data, {'path': str(path), **after}
    raise ValueError(f'{label} changed while it was being observed; retry the import')


def _capture_snapshot(snapshot_dir: Path, name: str, live_path: Path,
                      payload: bytes, observation: dict) -> dict:
    """Durably preserve the exact bytes used for validation under a content-addressed name."""
    digest = _sha_bytes(payload)
    snapshot_dir = snapshot_dir.expanduser().resolve()
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    target = snapshot_dir / f'{name}-{digest}.snapshot'
    if target.is_symlink():
        raise ValueError(f'{name} snapshot path must not be a symlink')
    if target.exists():
        existing = target.read_bytes()
        if existing != payload:
            raise ValueError(f'{name} snapshot path is occupied by different bytes')
    else:
        fd, temporary_name = tempfile.mkstemp(prefix=f'.{name}-', dir=snapshot_dir)
        try:
            with os.fdopen(fd, 'wb') as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(temporary_name, 0o444)
            try:
                os.link(temporary_name, target)
            except FileExistsError:
                if target.read_bytes() != payload:
                    raise ValueError(f'{name} snapshot raced with different bytes')
            directory_fd = os.open(snapshot_dir, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
    binding = _binding(target, payload)
    return {'observed': {'path': str(live_path.resolve(strict=True)), 'bytes': len(payload),
                         'sha256': digest, 'stat': observation,
                         'prefix': {'bytes': len(payload), 'sha256': digest}},
            'snapshot': binding}


def _input_binding(path: Path, selected_files: list[Path] | None = None) -> dict:
    resolved = path.resolve(strict=True)
    if resolved.is_file():
        return {'kind': 'file', **_binding(resolved)}
    if not resolved.is_dir():
        raise ValueError('import input must be a JSONL file or a directory')
    if selected_files is None:
        raise ValueError('directory custody must name its exact closed-case result-file cohort')
    files = sorted({item.resolve(strict=True) for item in selected_files})
    return {'kind': 'directory', 'path': str(resolved), 'files': [_binding(item) for item in files]}


def _jsonl(payload: bytes, label: str) -> list[tuple[int, bytes, dict]]:
    rows = []
    for line_no, raw in enumerate(payload.splitlines(keepends=True), 1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except ValueError as error:
            raise ValueError(f'{label}:{line_no}: invalid JSONL row') from error
        if not isinstance(value, dict):
            raise ValueError(f'{label}:{line_no}: result row must be an object')
        rows.append((line_no, raw, value))
    return rows


def _campaign_root(input_path: Path, explicit_root: Path | None) -> Path:
    if explicit_root is not None:
        root = explicit_root.resolve(strict=True)
        if not (root / 'plan.json').is_file():
            raise ValueError('the explicit import campaign has no plan.json')
        return root
    cursor = input_path if input_path.is_dir() else input_path.parent
    for candidate in (cursor, *cursor.parents):
        if (candidate / 'plan.json').is_file():
            return candidate.resolve(strict=True)
    raise ValueError('result files outside their reviewed campaign require --import-campaign')


def _load_bound_dispatcher(plan_path: Path):
    """Load the plan's real canonical dispatcher only when candidate bytes match its pin.

    The validator intentionally requires its own canonical checkout. Running this
    reader from a private worktree must not bypass that guard or make the private
    copy appear canonical. Instead, bind both the actual canonical file and this
    reader's candidate dependency to the dispatcher's exact reviewed digest.
    """
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    canonical_root = Path(plan.get('cwd', '')).expanduser().resolve(strict=True)
    canonical_path = (canonical_root / 'scripts/run_reviewed_luna_dispatcher.py').resolve(strict=True)
    expected = plan.get('dispatcher_sha256')
    pins = plan.get('artifact_hashes')
    if not isinstance(expected, str) or not isinstance(pins, dict) or pins.get(str(canonical_path)) != expected:
        raise ValueError('reviewed plan does not bind its canonical dispatcher path and digest')
    candidate_path = Path(REPOSITORY_ROOT) / 'scripts/run_reviewed_luna_dispatcher.py'
    if _binding(candidate_path)['sha256'] != expected:
        raise ValueError('private dispatcher candidate differs from the reviewed canonical dispatcher digest')
    if _binding(canonical_path)['sha256'] != expected:
        raise ValueError('canonical dispatcher bytes differ from the reviewed dispatcher digest')

    scripts_root = str(canonical_root / 'scripts')
    if scripts_root not in sys.path:
        sys.path.insert(0, scripts_root)
    name = f'_natlang_reviewed_dispatcher_{expected}'
    spec = importlib.util.spec_from_file_location(name, canonical_path)
    if spec is None or spec.loader is None:
        raise ValueError('cannot load the authenticated canonical dispatcher')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if (Path(module.__file__).resolve() != canonical_path or Path(module.ROOT).resolve() != canonical_root
            or Path(module.DISPATCHER).resolve() != canonical_path):
        raise ValueError('loaded dispatcher identity differs from the authenticated canonical checkout')
    return module


def _selected_result_files(input_path: Path, campaign_root: Path) -> list[Path] | None:
    """Return in-campaign result paths, or None for an explicitly bound detached JSONL file."""
    resolved = input_path.resolve(strict=True)
    try:
        resolved.relative_to(campaign_root)
    except ValueError:
        if not resolved.is_file():
            raise ValueError('a detached import must be one JSONL file')
        return None
    if resolved.is_file():
        return [resolved]
    return sorted(path.resolve(strict=True) for path in resolved.rglob('*.results.jsonl') if path.is_file())


def _source_rows(source_path: Path) -> dict[int, tuple[bytes, dict]]:
    rows = {}
    for index, raw, row in _jsonl(source_path.read_bytes(), str(source_path)):
        source_index = index - 1
        if source_index in rows:
            raise ValueError('tune case source repeats a physical row index')
        rows[source_index] = (raw, row)
    return rows


def verify_import(input_path: Path, *, campaign_root: Path | None = None,
                  snapshot_dir: Path) -> dict:
    input_path = input_path.resolve(strict=True)
    root = _campaign_root(input_path, campaign_root)
    plan_path = root / 'plan.json'
    plan_bytes = plan_path.read_bytes()
    plan_sha = _sha_bytes(plan_bytes)
    dispatcher = _load_bound_dispatcher(plan_path)
    plan, identity = dispatcher.validate_plan(plan_path, plan_sha)
    identity['identity_sha256'] = hashlib.sha256(json.dumps(
        identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if Path(identity['campaign_root']).resolve(strict=True) != root:
        raise ValueError('reviewed dispatcher identity names another campaign root')

    launch_path = Path(identity['launch_record']).resolve(strict=True)
    launch_bytes, launch_observation = _stable_read(launch_path, 'dispatcher launch record')
    launch_binding = _capture_snapshot(snapshot_dir, 'launch', launch_path, launch_bytes, launch_observation)
    launch = json.loads(launch_bytes)
    if launch.get('status') not in {'running', 'draining', 'finished', 'incomplete', 'drained', 'operator_stopped'}:
        raise ValueError('tune imports require a recognized reviewed dispatcher launch state')
    if launch.get('identity') != identity:
        raise ValueError('dispatcher launch identity differs from the validated reviewed plan')

    ledger_path = Path(identity['ledger']).resolve(strict=True)
    ledger_bytes, ledger_observation = _stable_read(ledger_path, 'dispatcher claim ledger')
    ledger_binding = _capture_snapshot(snapshot_dir, 'claim-ledger', ledger_path, ledger_bytes, ledger_observation)
    events = []
    for line_no, line in enumerate(ledger_bytes.splitlines(), 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except ValueError as error:
            raise ValueError(f'claim ledger is corrupt at line {line_no}') from error
        if not isinstance(event, dict):
            raise ValueError(f'claim ledger event {line_no} is not an object')
        events.append(event)
    dispatcher._verify_ledger_identity(events, identity)
    dispatcher._verify_ledger_events(events, identity)

    reviewed_artifacts = []
    for reference, expected_sha in sorted(plan.get('artifact_hashes', {}).items()):
        artifact_path = Path(reference).expanduser().resolve(strict=True)
        binding = _binding(artifact_path)
        if binding['sha256'] != expected_sha:
            raise ValueError(f'reviewed dispatcher artifact changed after plan validation: {artifact_path}')
        reviewed_artifacts.append(binding)

    source_path = Path(identity['source']).resolve(strict=True)
    source_bytes = source_path.read_bytes()
    if _sha_bytes(source_bytes) != identity['source_sha256']:
        raise ValueError('reviewed tune source changed after plan validation')
    source_rows = _source_rows(source_path)
    cases = {case['index']: case for case in identity['cases']}
    if len(cases) != len(identity['cases']):
        raise ValueError('reviewed tune plan repeats source case indices')

    claims = {event['claim_id']: event for event in events if event.get('event') == 'claim'}
    terminals = {event['claim_id']: event for event in events if event.get('event') == 'terminal'}
    selected_files = _selected_result_files(input_path, root)
    detached_rows = None
    if selected_files is None:
        detached_bytes = input_path.read_bytes()
        detached_rows = _jsonl(detached_bytes, str(input_path))
        selected_hashes = {}
        for line_no, _, row in detached_rows:
            row_hash = _canonical_sha(row)
            if row_hash in selected_hashes:
                raise ValueError('detached import repeats an identical result row; source occurrence is ambiguous')
            selected_hashes[row_hash] = line_no
    else:
        selected_paths = {str(path) for path in selected_files}
        selected_hashes = None

    verified = []
    seen_cases = set()
    for index, case in sorted(cases.items()):
        claim_id = f"{identity['campaign_id']}-case-{index:06d}"
        claim = claims.get(claim_id)
        terminal = terminals.get(claim_id)
        if claim is None or terminal is None or terminal.get('exit_code') != 0:
            continue
        accounting = terminal.get('output_accounting') or {}
        if (accounting.get('complete') is not True or accounting.get('manifest_valid') is not True
                or accounting.get('exact_result_rows') != 1 or accounting.get('output_rows') != 1
                or accounting.get('output_error') is not None or accounting.get('output_row_errors')):
            raise ValueError(f'closed tune claim {claim_id} has incomplete collector output accounting')
        slot = claim.get('slot')
        if type(slot) is not int or not 1 <= slot <= identity['slots']:
            raise ValueError(f'closed tune claim {claim_id} has an invalid reviewed slot')
        queue_path = Path(claim['queue']).resolve(strict=True)
        try:
            queue_path.relative_to(root)
        except ValueError as error:
            raise ValueError(f'closed tune claim {claim_id} escaped the campaign root') from error
        queue_bytes = queue_path.read_bytes()
        if _sha_bytes(queue_bytes) != claim.get('queue_sha256'):
            raise ValueError(f'closed tune claim {claim_id} queue bytes changed')
        queue_rows = _jsonl(queue_bytes, str(queue_path))
        if len(queue_rows) != 1:
            raise ValueError(f'closed tune claim {claim_id} must own one case queue row')
        _, _, queue = queue_rows[0]
        expected_case = cases[index]
        if (queue.get('key') != claim_id or queue.get('index') != index or queue.get('seed') != expected_case['seed']
                or queue.get('count') != 1 or Path(queue.get('source', '')).resolve() != source_path
                or queue.get('source_row_sha256') != expected_case.get('source_row_sha256')):
            raise ValueError(f'closed tune claim {claim_id} changed its reviewed source binding')
        output_path = Path(queue.get('output', '')).resolve(strict=True)
        expected_output = root / f'slot-{slot:02d}' / 'results' / f'case-{index:06d}.results.jsonl'
        if output_path != expected_output.resolve():
            raise ValueError(f'closed tune claim {claim_id} output path differs from dispatcher convention')
        output_bytes = output_path.read_bytes()
        output_binding = _binding(output_path, output_bytes)
        if output_binding['sha256'] != accounting.get('output_sha256') or accounting.get('output_rows') != 1:
            raise ValueError(f'closed tune claim {claim_id} result bytes differ from dispatcher accounting')
        result_rows = _jsonl(output_bytes, str(output_path))
        if len(result_rows) != 1:
            raise ValueError(f'closed tune claim {claim_id} must own one exact collector result')
        line_no, raw_result, row = result_rows[0]
        source_item = source_rows.get(index)
        if source_item is None:
            raise ValueError(f'closed tune claim {claim_id} names no source row')
        source_raw, source_row = source_item
        if _sha_bytes(source_raw) != expected_case.get('source_row_sha256'):
            raise ValueError(f'closed tune claim {claim_id} source row differs from the reviewed inventory')
        program_ir = (row.get('task') or {}).get('program_ir')
        if program_ir != source_row:
            raise ValueError(f'closed tune claim {claim_id} result changed its complete tune-bound task')
        if (row.get('version') != 'natlang.teacher_trajectory.native/1'
                or not isinstance(row.get('trajectory'), list) or not row['trajectory']):
            raise ValueError(f'closed tune claim {claim_id} has no complete native trajectory')
        provenance = row.get('provenance') or {}
        execution_identity = row.get('execution_identity') or {}
        if (provenance.get('program_ir_sha256') != _canonical_sha(program_ir)
                or provenance.get('model') != plan['model_id'] or provenance.get('provider') != plan['provider']
                or 'execution_plans' not in provenance
                or execution_identity.get('schema') != 'natlang.teacher_execution_identity/1'
                or not isinstance(execution_identity.get('execution_run_id'), str)
                or not execution_identity['execution_run_id']):
            raise ValueError(f'closed tune claim {claim_id} lacks exact reviewed Luna actor provenance')
        manifest_path = Path(str(output_path) + '.manifest.json').resolve(strict=True)
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
        if (manifest.get('version') != 'natlang.teacher_batch.native/1'
                or manifest.get('range') != {'start': index, 'count': 1}
                or manifest.get('completed') != 1 or manifest.get('missing')
                or manifest.get('output_sha256') != output_binding['sha256']
                or manifest.get('source_sha256') != identity['source_sha256']
                or Path(manifest.get('source', '')).resolve() != source_path):
            raise ValueError(f'closed tune claim {claim_id} output manifest is not bound to the exact source/result')

        if selected_files is not None:
            if str(output_path) not in selected_paths:
                continue
        else:
            row_hash = _canonical_sha(row)
            if row_hash not in selected_hashes:
                continue
            detached_line = selected_hashes.pop(row_hash)
        if index in seen_cases:
            raise ValueError('import input repeats a reviewed tune case')
        seen_cases.add(index)
        verified.append({
            'case_index': index, 'case': expected_case,
            'input': {'path': str(input_path) if selected_files is None else str(output_path),
                      'line': detached_line if selected_files is None else line_no},
            'result_row_sha256': _canonical_sha(row),
            'result_line_sha256': _sha_bytes(raw_result),
            'result_file': output_binding,
            'result_manifest': _binding(manifest_path, manifest_bytes),
            'queue': _binding(queue_path, queue_bytes),
            'source_row_sha256': _sha_bytes(source_raw),
            'provenance_sha256': _canonical_sha(provenance),
            'execution_identity': execution_identity,
            'actor': {'provider': plan['provider'], 'model': plan['model_id'],
                      'dispatch_plan_sha256': plan_sha,
                      'runtime_receipt_sha256': identity['runtime_receipt_sha256'],
                      'source_inventory_sha256': (identity.get('source_inventory') or {}).get('sha256')},
        })
    if selected_files is not None:
        verified_paths = {entry['result_file']['path'] for entry in verified}
        pending_or_failed_paths = set()
        for claim_id, claim in claims.items():
            terminal = terminals.get(claim_id)
            if terminal is not None and terminal.get('exit_code') == 0:
                continue
            slot, index = claim.get('slot'), claim.get('index')
            if type(slot) is int and type(index) is int:
                pending_or_failed_paths.add(str((root / f'slot-{slot:02d}' / 'results' /
                    f'case-{index:06d}.results.jsonl').resolve()))
        unverified = [str(path) for path in selected_files if str(path) not in verified_paths
                      and str(path) not in pending_or_failed_paths]
        if unverified:
            raise ValueError(f'import directory contains unclaimed or unverified result files: {unverified[:3]}')
    elif selected_hashes:
        raise ValueError('detached import contains a row not present in its reviewed campaign results')
    if not verified:
        raise ValueError('import input contains no exact closed tune-case results')

    return {
        'schema': SCHEMA,
        'campaign_root': str(root),
        'plan': _binding(plan_path, plan_bytes),
        'source': _binding(source_path, source_bytes),
        'claim_ledger': ledger_binding,
        'launch_record': launch_binding,
        'dispatcher_identity_sha256': _canonical_sha(identity),
        'reviewed_artifacts': reviewed_artifacts,
        'runtime': {'root': identity['runtime'], 'receipt_sha256': identity['runtime_receipt_sha256'],
                    'receipt': _binding(Path(identity['runtime']) / 'frozen-runtime.json')},
        'source_inventory': identity.get('source_inventory'),
        'input': (_input_binding(input_path) if input_path.is_file() else
                  _input_binding(input_path, [Path(row['result_file']['path']) for row in verified])),
        'selected_rows': verified,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--campaign', type=Path,
                        help='required for detached single/merged JSONL files; in-campaign paths infer their root')
    parser.add_argument('--snapshot-dir', required=True, type=Path,
                        help='durable importing-project directory for immutable ledger and launch snapshots')
    args = parser.parse_args(argv)
    receipt = verify_import(args.input, campaign_root=args.campaign, snapshot_dir=args.snapshot_dir)
    print(json.dumps(receipt, sort_keys=True, ensure_ascii=False))


if __name__ == '__main__':
    main()
