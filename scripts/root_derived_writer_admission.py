"""Shared validation for exact root-admitted derived text-writer rows."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


SINGLE = 'natlang.root-derived-observed-text-writer-admission/1'
INDEX = 'natlang.root-derived-body-admission-index/1'


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _read_pin(root: Path, relative: str, pin: dict) -> Path:
    path = Path(relative)
    if path.is_absolute() or not isinstance(pin, dict):
        raise ValueError('derived-writer input pin is malformed')
    path = (root / path).resolve()
    if (not path.is_relative_to(root) or not path.is_file() or _sha(path) != pin.get('sha256') or
            ('bytes' in pin and path.stat().st_size != pin.get('bytes'))):
        raise ValueError(f'derived-writer input pin is missing or mismatched: {relative}')
    return path


def _validate_receipt(receipt: dict, receipt_path: Path, *, root: Path) -> dict:
    if (receipt.get('schema') != SINGLE or
            receipt.get('decision') != 'admit-derived-observed-text-writer-body' or
            receipt.get('training_admission') is not True or
            receipt.get('original_action_admission') is not False or
            receipt.get('limits', {}).get('whole_trajectory_admission') is not False or
            receipt.get('limits', {}).get('runtime_gradient_qualification') is not False or
            receipt.get('limits', {}).get('active_training_inputs_changed') is not False):
        raise ValueError('root receipt does not admit only the derived writer body')
    pins = receipt.get('input_pins')
    if not isinstance(pins, dict) or not pins:
        raise ValueError('root derived-writer admission lacks input pins')
    for relative, pin in pins.items():
        _read_pin(root, relative, pin)
    proposal_id = receipt.get('proposal_id')
    target_sha = receipt.get('derived_target_sha256')
    source_id = receipt.get('source_native_record_id')
    group, split, body_sha = (receipt.get('source_group'), receipt.get('split'),
                              receipt.get('exact_body_sha256'))
    if not all(isinstance(item, str) and item for item in
               (proposal_id, target_sha, source_id, group, split, body_sha)):
        raise ValueError('root derived-writer admission lacks exact target/source fields')
    return {'native_id': proposal_id, 'target_sha256': target_sha, 'source_native_record_id': source_id,
            'source_group': group, 'split': split, 'exact_body_sha256': body_sha,
            'receipt_path': receipt_path.relative_to(root).as_posix(),
            'receipt_sha256': _sha(receipt_path)}


def admitted_root_derived_writer_rows(approval: dict, approval_path: Path, *,
                                      delta_records: list[dict], root: Path) -> list[dict]:
    """Resolve a single receipt or pinned index; require exact delta ID/target/group/split bijection."""
    root = root.resolve()
    approval_path = approval_path.resolve()
    if not approval_path.is_relative_to(root) or not approval_path.is_file():
        raise ValueError('root derived-writer approval must be a file inside the repository')
    index_sha = _sha(approval_path)
    if approval.get('schema') == SINGLE:
        receipt_rows = [_validate_receipt(approval, approval_path, root=root)]
    elif approval.get('schema') == INDEX:
        entries = approval.get('admissions')
        if not isinstance(entries, list) or not entries or approval.get('count') != len(entries):
            raise ValueError('root derived-writer index count does not match its admission list')
        receipt_rows = []
        paths = set()
        for entry in entries:
            relative, expected = entry.get('path'), entry.get('sha256')
            if not isinstance(relative, str) or not isinstance(expected, str) or relative in paths:
                raise ValueError('root derived-writer index has malformed or duplicate receipt paths')
            paths.add(relative)
            receipt_path = _read_pin(root, relative, {'sha256': expected})
            receipt = json.loads(receipt_path.read_text())
            row = _validate_receipt(receipt, receipt_path, root=root)
            if entry.get('source_group') != row['source_group']:
                raise ValueError(f'index source group disagrees with child receipt: {relative}')
            receipt_rows.append(row)
    else:
        raise ValueError(f"unsupported derived-writer approval schema: {approval.get('schema')!r}")

    approved = {row['native_id']: row for row in receipt_rows}
    if len(approved) != len(receipt_rows):
        raise ValueError('root derived-writer approvals contain duplicate proposal IDs')
    ids = [row.get('id') for row in delta_records]
    if len(ids) != len(set(ids)) or set(ids) != set(approved) or len(ids) != len(approved):
        raise ValueError('derived-writer delta IDs do not equal root admissions exactly')
    for record in delta_records:
        admission = approved[record['id']]
        derived = record.get('derived_target') or {}
        target = record.get('target')
        target_sha = hashlib.sha256(json.dumps(target, ensure_ascii=False,
            separators=(',', ':')).encode()).hexdigest()
        if (target_sha != admission['target_sha256'] or record.get('split') != admission['split'] or
                admission['source_group'] not in (record.get('source_groups') or []) or
                derived.get('schema') != 'natlang.root-admitted-derived-text-writer-target/1' or
                derived.get('original_native_record_id') != admission['source_native_record_id'] or
                derived.get('exact_body_sha256') != admission['exact_body_sha256'] or
                derived.get('target_sha256') != admission['target_sha256']):
            raise ValueError(f'root derived-writer target/source/split/group mismatch: {record["id"]}')
        admission['admission_index_sha256'] = index_sha
    return [approved[ident] for ident in ids]
