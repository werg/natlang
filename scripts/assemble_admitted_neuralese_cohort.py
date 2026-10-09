#!/usr/bin/env python3
"""Assemble a hash-bound approved native delta onto immutable native/R prefixes.

This proposal builder has no corpus-specific IDs or expected row totals. The approval
receipt supplies the exact row IDs; prefixes are copied byte-for-byte and deltas are
appended only after source hashes, split/group isolation, piece closure and recurrence
structure have been checked. It writes a proposal directory and does not publish it.
"""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import shutil
import errno
import os

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / 'scripts'))
from root_integration_adoption import root_integration_adoption_bindings
from root_derived_writer_admission import admitted_root_derived_writer_rows
from root_admission_scope import no_new_world_credit


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def rows(path: Path, key='id'):
    seen = set()
    with path.open('rb') as f:
        for line in f:
            if not line.strip(): continue
            row = json.loads(line)
            ident = row.get(key)
            if not isinstance(ident, str) or not ident or ident in seen:
                raise ValueError(f'missing or duplicate {key} in {path}: {ident!r}')
            seen.add(ident)
            yield row


def line(row):
    return json.dumps(row, ensure_ascii=False, separators=(',', ':')).encode() + b'\n'

def target_digest(row):
    value = row.get('target')
    if not isinstance(value, dict):
        raise ValueError(f"native row has no structured target: {row.get('id')}")
    # Admission target digests use canonical key ordering so they do not depend on
    # the insertion order used by the source converter.
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def json_digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()

def validate_root_recurrence_facet_approval(approval, approval_path, delta_rows, *, root=CODE_ROOT):
    """Validate explicit recurrence-only admission for exact held R records."""
    if approval.get('schema') != 'natlang.root-neuralese-record-facet-admission/1':
        raise ValueError('unsupported root record-facet admission schema')
    if (approval.get('decision') != 'admit-exact-counterfactual-recurrence-only'
            or approval.get('facet') != 'recurrence'
            or approval.get('training_recurrence_facet') is not True
            or approval.get('native_sft') is not False
            or approval.get('ordinary_text') is not False
            or approval.get('task_or_trajectory_admission') is not False
            or approval.get('runtime_qualification') is not False
            or approval.get('active_gpu_inputs_changed') is not False
            or not no_new_world_credit(approval.get('new_world_credit'))):
        raise ValueError('root record-facet receipt exceeds recurrence-only scope')
    for path_field, hash_field in (('review_path', 'review_sha256'), ('proof_path', 'proof_sha256')):
        rel, expected = approval.get(path_field), approval.get(hash_field)
        path = Path(rel) if isinstance(rel, str) else Path('/')
        if (not isinstance(rel, str) or path.is_absolute() or not isinstance(expected, str)
                or len(expected) != 64):
            raise ValueError(f'root recurrence receipt lacks a repository-relative {path_field} pin')
        resolved = (root / path).resolve()
        if (not resolved.is_relative_to(root.resolve()) or not resolved.is_file()
                or sha(resolved) != expected):
            raise ValueError(f'root recurrence {path_field} is missing or mismatched')
    pins = approval.get('input_pins')
    if not isinstance(pins, dict) or not pins:
        raise ValueError('root recurrence receipt lacks input pins')
    for rel, pin in pins.items():
        path = Path(rel)
        if path.is_absolute() or not isinstance(pin, dict):
            raise ValueError('root recurrence input pin is malformed')
        resolved = (root / path).resolve()
        if (not resolved.is_relative_to(root.resolve()) or not resolved.is_file()
                or sha(resolved) != pin.get('sha256')
                or resolved.stat().st_size != pin.get('bytes')):
            raise ValueError(f'root recurrence input is missing or mismatched: {rel}')
    if approval.get('review_path') not in pins or approval.get('proof_path') not in pins:
        raise ValueError('root recurrence review and proof must also be included in input_pins')
    seen, admitted = set(), {}
    for item in approval.get('rows', []):
        ident = item.get('id')
        if not isinstance(ident, str) or not ident or ident in seen:
            raise ValueError(f'root recurrence receipt has missing/duplicate record ID: {ident!r}')
        seen.add(ident)
        if (item.get('decision') != 'admit-counterfactual-recurrence-record'
                or item.get('facet') != 'recurrence'
                or item.get('training_admission') is not True
                or item.get('native_sft') is not False
                or item.get('ordinary_text') is not False
                or item.get('task_or_trajectory_admission') is not False
                or item.get('runtime_qualification') is not False
                or item.get('active_gpu_inputs_changed') is not False):
            raise ValueError(f'root recurrence row exceeds its facet scope: {ident}')
        if (item.get('kind') not in {'derived-body-as-proposed-learned-writer',
                                     'observed-reader-with-proposed-learned-writer-input'}
                or item.get('split') not in {'train', 'test'}
                or not isinstance(item.get('source_groups'), list) or not item['source_groups']
                or not all(isinstance(group, str) and group for group in item['source_groups'])):
            raise ValueError(f'root recurrence row lacks exact source/split/kind: {ident}')
        admitted[ident] = item
    if approval.get('admitted_recurrence_count') != len(admitted):
        raise ValueError('root recurrence admitted count conflicts with exact row decisions')
    ids = [row.get('id') for row in delta_rows]
    if (not admitted or len(ids) != len(set(ids)) or set(ids) != set(admitted)):
        raise ValueError('recurrence delta IDs do not equal root-admitted facet IDs exactly')
    for row in delta_rows:
        ident = row['id']
        decision = admitted[ident]
        view = row.get('counterfactual_recurrence_view') or {}
        if (decision['kind'] != view.get('role')
                or decision['target_sha256'] != target_digest(row)
                or decision['messages_sha256'] != json_digest(row.get('messages') or [])
                or decision['split'] != row.get('split')
                or decision['source_groups'] != row.get('source_groups')):
            raise ValueError(f'root recurrence target/context/source mismatch: {ident}')
    return admitted

def validate_root_per_action_approval(approval, delta_rows, *, root=CODE_ROOT):
    """Validate a mixed root review and return only its admitted native decisions."""
    if approval.get('schema') != 'natlang.root-per-action-training-admission/1':
        raise ValueError('unsupported root per-action admission schema')
    review_rel, review_hash = approval.get('review_path'), approval.get('review_sha256')
    if not isinstance(review_rel, str) or Path(review_rel).is_absolute() or not isinstance(review_hash, str):
        raise ValueError('root per-action approval lacks a repository-relative review pin')
    review_path = (root / review_rel).resolve()
    if not review_path.is_relative_to(root.resolve()) or not review_path.is_file() or sha(review_path) != review_hash:
        raise ValueError('root per-action review is missing or mismatched')
    pins = approval.get('input_pins')
    if not isinstance(pins, dict) or not pins:
        raise ValueError('root per-action approval lacks pinned review inputs')
    for rel, pin in pins.items():
        path = Path(rel)
        if path.is_absolute() or not isinstance(pin, dict):
            raise ValueError('root per-action input pin is malformed')
        path = (root / path).resolve()
        if (not path.is_relative_to(root.resolve()) or not path.is_file()
                or sha(path) != pin.get('sha256') or path.stat().st_size != pin.get('bytes')):
            raise ValueError(f'root per-action input is missing or mismatched: {rel}')
    dispositions = {
        'admit-ordinary-native-action': 'admitted_native_count',
        'hold-source-required-neuralese-reader-contract': 'held_source_contract_final_count',
        'hold-ambiguous-source-read-scope': 'held_ambiguous_source_read_scope_count',
        'hold-unrecorded-action': 'held_unrecorded_action_count',
        'reject-action-failed': 'failed_count',
        'exclude-already-admitted-case01-duplicate': 'already_adopted_count',
    }
    seen, counts, admitted = set(), collections.Counter(), {}
    for row in approval.get('rows', []):
        ident, decision = row.get('native_id'), row.get('decision')
        if not isinstance(ident, str) or not ident or ident in seen:
            raise ValueError(f'root per-action approval has missing/duplicate ID: {ident!r}')
        seen.add(ident)
        if decision not in dispositions:
            raise ValueError(f'unknown root per-action disposition: {decision!r}')
        is_admitted = decision == 'admit-ordinary-native-action'
        if not isinstance(row.get('training_admission'), bool) or row['training_admission'] != is_admitted:
            raise ValueError(f'root per-action flag conflicts with decision: {ident}')
        counts[decision] += 1
        if is_admitted:
            if row.get('split') != 'train':
                raise ValueError(f'root per-action admission is not train-only: {ident}')
            admitted[ident] = row
    if not seen:
        raise ValueError('root per-action approval has no decision rows')
    if approval.get('admitted_native_count') != len(admitted):
        raise ValueError('root per-action admitted count conflicts with rows')
    for decision, field in dispositions.items():
        if approval.get(field, 0) != counts[decision]:
            raise ValueError(f'root per-action disposition count conflicts: {field}')
    if (approval.get('whole_trajectory_admission') is not False
            or approval.get('runtime_qualification') is not False
            or approval.get('active_gpu_inputs_changed') is not False
            or not no_new_world_credit(approval.get('new_world_credit'))):
        raise ValueError('root per-action approval includes an unsupported admission facet')
    ids = [row.get('id') for row in delta_rows]
    if len(ids) != len(set(ids)) or set(ids) != set(admitted):
        raise ValueError('delta native IDs do not equal root-admitted per-action IDs exactly')
    return admitted


def merge_root_approval_rows(by_id, hashes_by_id, current_by_id, receipt_hash):
    """Union independently validated root decisions, rejecting duplicate IDs even if identical."""
    overlap = set(by_id) & set(current_by_id)
    if overlap:
        for ident in overlap:
            if by_id[ident] != current_by_id[ident] or hashes_by_id[ident] != receipt_hash:
                raise ValueError(f'conflicting duplicate approval ID: {ident}')
        raise ValueError(f'duplicate approval IDs across receipts: {sorted(overlap)}')
    return {**by_id, **current_by_id}, {**hashes_by_id, **{ident: receipt_hash for ident in current_by_id}}

def receipt_artifact(receipt, name):
    for entry in receipt.get('files', {}).values():
        if entry.get('path') and entry.get('sha256') and entry.get('path').endswith(name):
            return entry
    return None

def prefix_sha(path: Path, length: int) -> str:
    h=hashlib.sha256(); remaining=length
    with path.open('rb') as f:
        while remaining:
            block=f.read(min(1<<20, remaining))
            if not block: raise ValueError('unexpected EOF during prefix verification')
            h.update(block); remaining-=len(block)
    return h.hexdigest()


def referenced_soft(value):
    found = set()
    if isinstance(value, dict):
        if value.get('type') == 'soft' and isinstance(value.get('name'), str): found.add(value['name'])
        for child in value.values(): found.update(referenced_soft(child))
    elif isinstance(value, list):
        for child in value: found.update(referenced_soft(child))
    return found


def write_append(prefix: Path, output: Path, additions):
    with output.open('xb') as out:
        with prefix.open('rb') as src:
            last=b''
            for block in iter(lambda: src.read(1<<20), b''):
                out.write(block); last=block[-1:]
        if prefix.stat().st_size and last != b'\n': out.write(b'\n')
        for row in additions: out.write(line(row))
    if prefix_sha(output, prefix.stat().st_size) != sha(prefix):
        raise ValueError(f'prefix changed while appending: {prefix}')


def link_or_copy_unchanged(source: Path, output: Path):
    """Expose an immutable unchanged artifact without duplicating same-device bytes."""
    if output.exists(): raise ValueError(f'output already exists: {output}')
    if source.stat().st_dev == output.parent.stat().st_dev:
        try:
            os.link(source, output)
            method = 'hardlink'
        except OSError as exc:
            if exc.errno != errno.EXDEV: raise
            shutil.copyfile(source, output)
            method = 'copy-cross-device'
    else:
        shutil.copyfile(source, output)
        method = 'copy-cross-device'
    if sha(source) != sha(output):
        raise ValueError(f'unchanged artifact hash mismatch after {method}: {source}')
    return {'method': method, 'sha256': sha(output), 'bytes': output.stat().st_size,
            'same_inode': os.path.samefile(source, output)}


def merge_pieces(prefix: Path, delta: Path, output: Path):
    prior = {p['name']: p for p in rows(prefix, 'name')}
    additions, duplicates = [], []
    for piece in rows(delta, 'name'):
        name = piece.get('name')
        if not isinstance(name, str) or not name: raise ValueError('piece has no name')
        if name not in prior:
            prior[name] = piece; additions.append(piece)
        elif prior[name] == piece: duplicates.append(name)
        else: raise ValueError(f'piece name conflict: {name}')
    with output.open('xb') as out:
        with prefix.open('rb') as src:
            last=b''
            for block in iter(lambda: src.read(1<<20), b''):
                out.write(block); last=block[-1:]
        if prefix.stat().st_size and last != b'\n': out.write(b'\n')
        for piece in additions: out.write(line(piece))
    if prefix_sha(output, prefix.stat().st_size) != sha(prefix): raise ValueError('piece prefix changed')
    return {'added': len(additions), 'exact_duplicates': sorted(duplicates), 'total': len(prior)}


def assemble_recurrence_facet_proposal(*, args, paths, approval_path, approval, repo_root, audit_script):
    """Append an explicitly recurrence-only approved delta; keep native bytes untouched."""
    if not args.compact_only:
        raise ValueError('recurrence-facet admission currently requires --compact-only')
    if args.delta_recurrence is None or args.delta_recurrence_pieces is None:
        raise ValueError('recurrence-facet admission requires explicit recurrence record and piece deltas')
    if len(args.approval) != 1:
        raise ValueError('recurrence-facet admission requires exactly one record-facet receipt')
    if args.delta_native.stat().st_size or args.delta_pieces.stat().st_size:
        raise ValueError('recurrence-facet admission requires byte-empty native and native-piece deltas')
    if approval.get('schema') != 'natlang.root-neuralese-record-facet-admission/1':
        raise ValueError('recurrence-facet admission requires a record-facet receipt')
    base_native_rows = list(rows(paths['base-native']))
    base_recurrence_rows = list(rows(paths['base-recurrence']))
    base_recurrence_ids = {row['id'] for row in base_recurrence_rows}
    delta_recurrence = list(rows(args.delta_recurrence))
    admitted = validate_root_recurrence_facet_approval(
        approval, approval_path, delta_recurrence, root=repo_root)
    pins = approval.get('input_pins') or {}
    for path in (args.delta_recurrence, args.delta_recurrence_pieces):
        try:
            rel = path.resolve().relative_to(repo_root).as_posix()
        except ValueError as exc:
            raise ValueError('recurrence delta inputs must be inside the canonical repository root') from exc
        pin = pins.get(rel)
        if (not pin or pin.get('sha256') != sha(path) or pin.get('bytes') != path.stat().st_size):
            raise ValueError(f'root recurrence receipt does not pin exact delta input: {rel}')
    new_ids = [row['id'] for row in delta_recurrence]
    if set(new_ids) & base_recurrence_ids:
        raise ValueError('recurrence delta ID already occurs in the base prefix')

    base_piece_rows = list(rows(paths['base-recurrence-pieces'], 'name'))
    delta_piece_rows = list(rows(args.delta_recurrence_pieces, 'name'))
    base_pieces = {row['name']: row for row in base_piece_rows}
    delta_pieces = {row['name']: row for row in delta_piece_rows}
    wanted = set().union(*(referenced_soft(row.get('messages')) |
                           referenced_soft(row.get('target')) for row in delta_recurrence))
    for name, piece in delta_pieces.items():
        prior = base_pieces.get(name)
        if prior is not None and prior != piece:
            raise ValueError(f'recurrence piece conflicts with base piece: {name}')
    available = set(base_pieces) | set(delta_pieces)
    if wanted - available:
        raise ValueError(f'missing recurrence pieces: {sorted(wanted-available)}')
    if set(delta_pieces) - wanted:
        raise ValueError(f'unreferenced recurrence pieces: {sorted(set(delta_pieces)-wanted)}')

    receipt_sha = sha(approval_path)
    admitted_rows = []
    for source_row in delta_recurrence:
        row = dict(source_row)
        decision = admitted[row['id']]
        row['training_admission'] = {
            'approved': True, 'kind': 'root-counterfactual-recurrence-only',
            'facet': 'recurrence', 'root_admission_sha256': receipt_sha,
            'decision': decision['decision'],
        }
        row['recurrence_admission'] = {
            'approved': True, 'kind': 'root-counterfactual-recurrence-only',
            'root_admission_sha256': receipt_sha,
        }
        row['native_sft_admission'] = {'approved': False}
        row['ordinary_text_admission'] = {'approved': False}
        row['task_or_trajectory_admission'] = {'approved': False}
        row['runtime_qualification'] = False
        row['decision'] = {**(row.get('decision') or {}),
                           'training_approved': False,
                           'training_admission_facet': 'recurrence'}
        view = dict(row.get('counterfactual_recurrence_view') or {})
        view['training_admission'] = True
        view['recurrence_admission'] = True
        view['native_sft_admission'] = False
        view['root_recurrence_facet_admission_sha256'] = receipt_sha
        row['counterfactual_recurrence_view'] = view
        admitted_rows.append(row)

    out = args.out.resolve()
    if any(out.iterdir()):
        raise ValueError(f'output directory is not empty: {out}')
    names = {'delta_recurrence': 'delta-recurrence-records.jsonl',
             'delta_recurrence_pieces': 'delta-recurrence-pieces.jsonl',
             'audit': 'recurrence-audit.json'}
    delta_output = out / names['delta_recurrence']
    with delta_output.open('xb') as stream:
        for row in admitted_rows:
            stream.write(line(row))
    shutil.copyfile(args.delta_recurrence_pieces, out / names['delta_recurrence_pieces'])
    audit_path = out / names['audit']
    subprocess.run([sys.executable, str(audit_script),
                    str(paths['base-recurrence']), str(delta_output),
                    '--out', str(audit_path)], cwd=repo_root, check=True)
    recurrence_audit = json.loads(audit_path.read_text())
    if recurrence_audit.get('structurally_closed') is not True:
        raise ValueError('base plus recurrence-facet delta is not structurally closed')
    virtual_ids = base_recurrence_ids | set(new_ids)
    if len(virtual_ids) != len(base_recurrence_rows) + len(new_ids):
        raise ValueError('recurrence IDs are not unique across the virtual union')
    output_manifest = {}
    for name in names.values():
        output_path = out / name
        output_manifest[name] = {'sha256': sha(output_path), 'bytes': output_path.stat().st_size}
        if name.endswith('.jsonl'):
            output_manifest[name]['rows'] = sum(1 for _ in rows(
                output_path, 'name' if name == 'delta-recurrence-pieces.jsonl' else 'id'))
    manifest = {
      'schema': 'natlang.approved-neuralese-compact-recurrence-delta-proposal/1',
      'status': 'compact recurrence-only delta; publication and training activation require separate review',
      'compact_only': True, 'repo_root': str(repo_root),
      'approval': {'path': str(approval_path.resolve()), 'sha256': receipt_sha,
                   'approved_ids': new_ids, 'facet': 'recurrence'},
      'argv': sys.argv,
      'invocation': {'canonical_repo_root': str(repo_root),
                     'auditor_path': str(audit_script),
                     'admission_facet': 'recurrence', 'compact_only': True},
      'admitted_facets': {'native': False, 'ordinary_text': False, 'recurrence': True,
                          'task_or_trajectory': False, 'runtime_qualification': False},
      'inputs': {key: {'path': str(path.resolve()), 'sha256': sha(path), 'bytes': path.stat().st_size}
                 for key, path in {**paths, 'delta-recurrence': args.delta_recurrence,
                                   'delta-recurrence-pieces': args.delta_recurrence_pieces,
                                   'approval': approval_path}.items() if key != 'out'},
      'unchanged_native_prefix': {'path': str(paths['base-native'].resolve()),
          'sha256': sha(paths['base-native']), 'bytes': paths['base-native'].stat().st_size,
          'rows': len(base_native_rows), 'delta_rows': 0, 'output_copy_made': False},
      'virtual_recurrence': {'base_rows': len(base_recurrence_rows),
          'added_rows': len(admitted_rows), 'total_rows': len(base_recurrence_rows)+len(admitted_rows),
          'ids_unique': True, 'base_sha256': sha(paths['base-recurrence'])},
      'piece_merge': {'base_pieces': len(base_piece_rows), 'delta_pieces': len(delta_piece_rows),
                      'virtual_piece_count': len(base_pieces | delta_pieces)},
      'recurrence_audit': recurrence_audit,
      'outputs': output_manifest,
      'assembler': {'path': str(Path(__file__).relative_to(CODE_ROOT)), 'sha256': sha(Path(__file__))},
      'auditor': {'path': str(audit_script), 'sha256': sha(audit_script)},
      'limits': {'native_delta_rows': 0, 'new_native_action_credit': False,
                 'new_text_document_credit': False, 'new_world_credit': False,
                 'whole_trajectory_admission': False, 'active_gpu_inputs_changed': False,
                 'publication': False, 'training_launch': False}}
    manifest_path = out / 'proposal-manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({'status': manifest['status'], 'native_rows_unchanged': len(base_native_rows),
                      'recurrence_rows': manifest['virtual_recurrence'],
                      'proposal_manifest_sha256': sha(manifest_path)}, indent=2))


def audit_splits(path: Path):
    return audit_split_rows(rows(path))


def audit_split_rows(source_rows):
    counts = collections.Counter()
    group_splits = collections.defaultdict(set)
    source_splits = {}
    ids = set()
    for row in source_rows:
        ids.add(row['id'])
        split = row.get('split')
        if split not in ('train', 'test'): raise ValueError(f'invalid split on {row["id"]}')
        counts[split] += 1
        groups = row.get('source_groups') or []
        if not groups: raise ValueError(f'no source_groups on {row["id"]}')
        for group in groups: group_splits[group].add(split)
        for sid in row.get('source_ids') or []:
            prior = source_splits.setdefault(sid, split)
            if prior != split: raise ValueError(f'source ID crosses split: {sid}')
    crossing = {g: sorted(s) for g, s in group_splits.items() if len(s) > 1}
    if crossing: raise ValueError(f'source groups cross splits: {crossing}')
    return {'rows': len(ids), 'train': counts['train'], 'test': counts['test'],
            'groups': len(group_splits), 'source_ids': len(source_splits), 'crossing_groups': crossing}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('base-native', 'base-recurrence', 'base-receipt', 'delta-native', 'delta-pieces',
                 'base-native-pieces', 'base-recurrence-pieces', 'out'):
        p.add_argument('--' + name, required=True, type=Path)
    p.add_argument('--approval', required=True, action='append', type=Path,
                   help='root approval receipt; repeat to combine independently validated per-action receipts')
    p.add_argument('--delta-recurrence', type=Path,
                   help='defaults to --delta-native when the approved records feed both streams')
    p.add_argument('--delta-recurrence-pieces', type=Path,
                   help='optional distinct soft-piece delta for recurrence-only admission')
    p.add_argument('--admission-facet', choices=('native', 'recurrence'), default='native',
                   help='admission stream to change; recurrence mode preserves the native prefix and adds no native rows')
    p.add_argument('--approval-id-field', default='approved_row_ids')
    p.add_argument('--admission-kind', default='exact-native-runtime-oracle')
    p.add_argument('--repo-root', type=Path, default=CODE_ROOT,
                   help='canonical data/receipt root; code may be loaded from a frozen snapshot')
    p.add_argument('--audit-script', type=Path, default=CODE_ROOT / 'scripts/audit_neuralese_recurrence.py',
                   help='pinned recurrence auditor executable')
    p.add_argument('--compact-only', action='store_true',
                   help='write only the admitted delta and compact audit; do not copy or assemble full base prefixes')
    args = p.parse_args()
    repo_root = args.repo_root.resolve()
    audit_script = args.audit_script.resolve()
    if not audit_script.is_file():
        raise ValueError(f'recurrence auditor is missing: {audit_script}')
    paths = {k: getattr(args, k.replace('-', '_')) for k in ('base-native','base-recurrence','base-receipt','delta-native',
        'delta-pieces','base-native-pieces','base-recurrence-pieces','out')}
    delta_r = args.delta_recurrence or args.delta_native
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()): raise ValueError(f'output directory is not empty: {out}')
    out.mkdir(parents=True, exist_ok=True)

    approvals = [(path, json.loads(path.read_text())) for path in args.approval]
    if not approvals:
        raise ValueError('at least one root approval receipt is required')
    approved_schemas = {'natlang.root-per-action-training-admission/1',
                        'natlang.root-selected-action-admission/1',
                        'natlang.root-derived-body-admission-index/1'}
    if len(approvals) > 1 and any(approval.get('schema') not in approved_schemas for _, approval in approvals):
        raise ValueError('multiple approvals require independently validated root action admission receipts')
    base_receipt = json.loads(paths['base-receipt'].read_text())
    root_corpus_receipt = base_receipt.get('schema') == 'natlang.root-corpus-admission/1'
    prefix_binding = base_receipt.get('schema') == 'natlang.corpus-prefix-binding/1'
    adoption_bindings = root_integration_adoption_bindings(base_receipt, root=repo_root)
    if prefix_binding:
        if base_receipt.get('status') != 'verified-exact-prefix' or base_receipt.get('training_admission') is not False:
            raise ValueError('base prefix binding must be verification-only, with no training admission')
        for name, path in (('native-records.jsonl', paths['base-native']),
                           ('recurrence-records.jsonl', paths['base-recurrence']),
                           ('native-pieces.jsonl', paths['base-native-pieces']),
                           ('recurrence-pieces.jsonl', paths['base-recurrence-pieces'])):
            binding = (base_receipt.get('files') or {}).get(name)
            if not binding or binding.get('sha256') != sha(path):
                raise ValueError(f'base prefix binding does not bind the exact {name}')
    elif adoption_bindings is not None:
        for name, artifact_key in (('base-native', 'native'), ('base-recurrence', 'recurrence'),
                                   ('base-native-pieces', 'native_pieces'),
                                   ('base-recurrence-pieces', 'recurrence_pieces')):
            binding = adoption_bindings['artifacts'][artifact_key]
            if binding['sha256'] != sha(paths[name]):
                raise ValueError(f'root integration adoption does not bind the exact {name} prefix')
    elif root_corpus_receipt:
        if (not str(base_receipt.get('status', '')).startswith('admitted-')
                or base_receipt.get('admission', {}).get('native_sft') is not True):
            raise ValueError('base root receipt does not grant native SFT admission')
        native_binding = receipt_artifact(base_receipt, 'native-records.jsonl')
        recurrence_binding = receipt_artifact(base_receipt, 'recurrence-records.jsonl')
        native_piece_binding = receipt_artifact(base_receipt, 'native-pieces.jsonl')
        recurrence_piece_binding = receipt_artifact(base_receipt, 'recurrence-pieces.jsonl')
        for binding, path, label in ((native_binding, paths['base-native'], 'native'),
                                     (recurrence_binding, paths['base-recurrence'], 'recurrence'),
                                     (native_piece_binding, paths['base-native-pieces'], 'native pieces'),
                                     (recurrence_piece_binding, paths['base-recurrence-pieces'], 'recurrence pieces')):
            if not binding or binding['sha256'] != sha(path):
                raise ValueError(f'base root receipt does not bind the exact {label} prefix')
    else:
        if base_receipt.get('approved') is not True:
            raise ValueError('base root receipt does not approve the native prefix')
        if base_receipt.get('records_sha256') != sha(paths['base-native']):
            raise ValueError('base root receipt does not bind the native prefix hash')
        if base_receipt.get('pieces_sha256') != sha(paths['base-native-pieces']):
            raise ValueError('base root receipt does not bind the native piece prefix hash')
    if args.admission_facet == 'recurrence':
        if len(approvals) != 1:
            raise ValueError('recurrence-facet mode requires exactly one root receipt')
        assemble_recurrence_facet_proposal(
            args=args, paths=paths, approval_path=approvals[0][0], approval=approvals[0][1],
            repo_root=repo_root, audit_script=audit_script)
        return
    approval_by_id = {}
    approval_sha_by_id = {}
    approval_schema_by_id = {}
    approval_ids_by_receipt = []
    root_action_admission = False
    root_per_action_admission = False
    root_derived_writer_admission = False
    for approval_path, approval in approvals:
        receipt_hash = sha(approval_path)
        this_root_action = approval.get('schema') == 'natlang.root-selected-action-admission/1'
        this_root_per_action = approval.get('schema') == 'natlang.root-per-action-training-admission/1'
        this_root_derived = approval.get('schema') in {
            'natlang.root-derived-observed-text-writer-admission/1',
            'natlang.root-derived-body-admission-index/1'}
        if this_root_action:
            root_action_admission = True
            if args.approval_id_field != 'approved_row_ids':
                raise ValueError('root action admission uses its fixed rows schema')
            if approval.get('qualifications', {}).get('recurrence') is True:
                raise ValueError('native-only selected-action receipt unexpectedly claims recurrence qualification')
            approval_rows = approval.get('rows')
            if not isinstance(approval_rows, list) or not approval_rows:
                raise ValueError('root action admission has no selected rows')
            approved = [entry.get('native_id') for entry in approval_rows]
            if any(not isinstance(ident, str) or not ident for ident in approved) or len(set(approved)) != len(approved):
                raise ValueError('root action admission has missing or duplicate selected IDs')
            if any(entry.get('decision') != 'admit-exact-selected-native-action-SFT-only' for entry in approval_rows):
                raise ValueError('root action admission contains an unsupported decision')
            counts = approval.get('counts') or {}
            old_counts = (counts.get('native_SFT_train_actions') == len(approval_rows)
                          and counts.get('whole_trajectories') == 0)
            selected_counts = (counts.get('selected_native_actions') == len(approval_rows)
                               and counts.get('train_actions') == len(approval_rows)
                               and counts.get('test_actions') == 0
                               and counts.get('whole_trajectories') == 0
                               and counts.get('new_worlds') == 0)
            qualifications = approval.get('qualifications', {})
            if (not (old_counts or selected_counts)
                    or qualifications.get('learned_writer') is not False
                    or qualifications.get('recurrence') is not False
                    or approval.get('integration', {}).get('active_GPU_inputs_changed') is True
                    or (selected_counts and
                        approval.get('integration', {}).get('active_GPU_inputs_changed') is not False)):
                raise ValueError('root action receipt does not describe exact native SFT-only admission')
            current_by_id = {entry['native_id']: entry for entry in approval_rows}
        elif this_root_per_action:
            root_per_action_admission = True
            all_delta_rows = list(rows(paths['delta-native']))
            approval_row_ids = {row.get('native_id') for row in approval.get('rows', [])
                                if row.get('decision') == 'admit-ordinary-native-action'
                                and row.get('training_admission') is True}
            current_delta_rows = [row for row in all_delta_rows if row.get('id') in approval_row_ids]
            current_by_id = validate_root_per_action_approval(approval, current_delta_rows, root=repo_root)
        elif this_root_derived:
            root_derived_writer_admission = True
            current_by_id = {row['native_id']: row for row in admitted_root_derived_writer_rows(
                approval, approval_path, delta_records=list(rows(paths['delta-native'])), root=repo_root)}
        else:
            current_ids = approval.get(args.approval_id_field)
            if not isinstance(current_ids, list) or not current_ids or any(not isinstance(x, str) for x in current_ids):
                raise ValueError(f'approval receipt lacks nonempty {args.approval_id_field!r}')
            if len(set(current_ids)) != len(current_ids):
                raise ValueError('duplicate approved IDs')
            current_by_id = {ident: {} for ident in current_ids}
        current_ids = list(current_by_id)
        if not current_ids or len(set(current_ids)) != len(current_ids):
            raise ValueError('approval receipt contains no IDs or duplicate IDs')
        approval_by_id, approval_sha_by_id = merge_root_approval_rows(
            approval_by_id, approval_sha_by_id, current_by_id, receipt_hash)
        approval_ids_by_receipt.append({'path': str(approval_path.resolve()), 'sha256': receipt_hash,
                                        'approved_ids': current_ids})
        approval_schema_by_id.update({ident: approval.get('schema') for ident in current_ids})
        # If the receipt provides an artifacts mapping, enforce hashes for every named input it binds.
        artifact_hashes = approval.get('artifact_hashes', {})
        for raw, expected in artifact_hashes.items():
            bound = (repo_root / raw).resolve()
            if not bound.is_file() or sha(bound) != expected:
                raise ValueError(f'approval artifact missing/hash mismatch: {raw}')
    approved = list(approval_by_id)
    approval = approvals[0][1]
    approval_summary = ({'path': str(approvals[0][0].resolve()), 'sha256': sha(approvals[0][0]),
                         'approved_ids': approved}
                        if len(approvals) == 1 else
                        {'receipts': approval_ids_by_receipt,
                         'id_field': 'union of independently validated root per-action admitted IDs'})
    approval_inputs = ({'approval': approvals[0][0]} if len(approvals) == 1 else
                       {f'approval-{i+1}': path for i, (path, _) in enumerate(approvals)})
    if not approved:
        raise ValueError('approval receipts contain no approved IDs')
    native_only = root_action_admission or root_per_action_admission or root_derived_writer_admission
    base_n = list(rows(paths['base-native'])); base_r = list(rows(paths['base-recurrence']))
    delta_n = list(rows(paths['delta-native']))
    delta_recs = [] if native_only and args.delta_recurrence is None else list(rows(delta_r))
    ids = [r['id'] for r in delta_n]
    if set(ids) != set(approved) or len(ids) != len(approved):
        raise ValueError('delta native IDs do not equal approved IDs exactly')
    rids = [r['id'] for r in delta_recs]
    if native_only:
        if delta_recs:
            raise ValueError('native-only receipt must not supply recurrence additions')
    elif set(rids) != set(approved) or len(rids) != len(approved):
        raise ValueError('delta recurrence IDs do not equal approved IDs exactly')
    if set(ids) & {r['id'] for r in base_n} or set(rids) & {r['id'] for r in base_r}:
        raise ValueError('delta ID already occurs in base prefix')
    admitted_delta = []
    for row in delta_n:
        row = dict(row)
        if root_derived_writer_admission:
            decision = approval_by_id[row['id']]
            target_sha = hashlib.sha256(json.dumps(row.get('target'), ensure_ascii=False,
                separators=(',', ':')).encode()).hexdigest()
            if (target_sha != decision.get('target_sha256') or row.get('split') != decision.get('split') or
                    decision.get('source_group') not in (row.get('source_groups') or [])):
                raise ValueError(f'root derived-writer source/target/split/group mismatch: {row["id"]}')
            row['training_admission'] = {
                **(row.get('training_admission') or {}),
                'approved': True,
                'kind': 'root-derived-text-writer-body-admission',
                'root_admission_sha256': decision['receipt_sha256'],
                'root_admission_index_sha256': decision['admission_index_sha256'],
                'source_group': decision['source_group'], 'split': decision['split']}
            row['derived_target'] = {**(row.get('derived_target') or {}),
                'root_admission_index_sha256': decision['admission_index_sha256'],
                'root_admission_sha256': decision['receipt_sha256']}
        elif root_action_admission or root_per_action_admission:
            decision = approval_by_id[row['id']]
            row_schema = approval_schema_by_id[row['id']]
            row_is_root_action = row_schema == 'natlang.root-selected-action-admission/1'
            if target_digest(row) != decision.get('target_sha256'):
                raise ValueError(f'root admission target digest mismatch: {row["id"]}')
            decision_groups = decision.get('source_groups')
            if decision_groups is None and isinstance(decision.get('source_group'), str):
                decision_groups = [decision['source_group']]
            groups_match = (isinstance(decision_groups, list) and bool(decision_groups)
                            and (decision_groups == (row.get('source_groups') or [])
                                 if row_is_root_action else
                                 set(decision_groups).issubset(set(row.get('source_groups') or []))))
            if row.get('split') != decision.get('split') or not groups_match:
                raise ValueError(f'root admission source split/group mismatch: {row["id"]}')
            row['training_admission'] = {
                'approved': True,
                'kind': ('root-per-action-native-action-sft-only' if not row_is_root_action
                         else 'root-selected-native-action-sft-only'),
                'status': 'admitted-ordinary-native-action' if not row_is_root_action
                          else 'admitted-exact-selected-native-action',
                'root_admission_sha256': approval_sha_by_id[row['id']],
                'decision': decision['decision'], 'role': decision.get('role')}
            row['decision'] = {**(row.get('decision') or {}), 'training_approved': True,
                               'root_action_admission_sha256': approval_sha_by_id[row['id']]}
        if row.get('training_admission', {}).get('approved') is not True:
            raise ValueError(f'delta row lacks approved admission metadata: {row["id"]}')
        if row.get('decision', {}).get('failed_action') is True:
            raise ValueError(f'failed action in approved delta: {row["id"]}')
        admitted_delta.append(row)
    delta_n = admitted_delta
    delta_piece_names = {piece['name'] for piece in rows(paths['delta-pieces'], 'name')}
    wanted = set().union(*(referenced_soft(r.get('messages')) | referenced_soft(r.get('target')) for r in delta_n))
    if delta_piece_names != wanted:
        raise ValueError(f'delta piece names do not exactly match delta soft references; missing={sorted(wanted-delta_piece_names)}, extra={sorted(delta_piece_names-wanted)}')

    if args.compact_only:
        # Validate the virtual prefix+delta in memory while publishing only compact additions.
        # This keeps the already adopted prefix as a hash-bound dependency and avoids a redundant
        # hundreds-of-megabytes copy during proposal construction.
        merged_pieces = {piece['name']: piece for piece in rows(paths['base-native-pieces'], 'name')}
        exact_piece_duplicates = []
        for piece in rows(paths['delta-pieces'], 'name'):
            prior = merged_pieces.get(piece['name'])
            if prior is not None:
                if prior != piece: raise ValueError(f'piece name conflict: {piece["name"]}')
                exact_piece_duplicates.append(piece['name'])
            else: merged_pieces[piece['name']] = piece
        missing = set().union(*(referenced_soft(r.get('messages')) | referenced_soft(r.get('target')) for r in delta_n)) - set(merged_pieces)
        if missing: raise ValueError(f'missing pieces in virtual native prefix for selected delta: {sorted(missing)}')
        split_audit = audit_split_rows([*base_n, *delta_n])
        names = {'delta': 'delta-native-records.jsonl', 'delta_pieces': 'delta-native-pieces.jsonl',
                 'audit': 'recurrence-audit.json'}
        with (out / names['delta']).open('xb') as stream:
            for row in delta_n: stream.write(line(row))
        shutil.copyfile(paths['delta-pieces'], out / names['delta_pieces'])
        audit_path = out / names['audit']
        subprocess.run([sys.executable, str(audit_script),
                        str(paths['base-recurrence']), '--out', str(audit_path)], cwd=repo_root, check=True)
        recurrence_audit = json.loads(audit_path.read_text())
        if recurrence_audit.get('structurally_closed') is not True:
            raise ValueError('unchanged recurrence prefix is not structurally closed')
        manifest = {
          'schema': 'natlang.approved-neuralese-compact-native-delta-proposal/1',
          'status': 'compact delta only; separate root integration review required',
          'compact_only': True,
          'repo_root': str(repo_root),
          'approval': {**approval_summary,
                       'id_field': ('rows.native_id under receipt-specific strict decision/training_admission predicates'
                                    if root_action_admission or root_per_action_admission else args.approval_id_field)},
          'admitted_facets': {'native': True, 'recurrence': False, 'native_only_receipt': native_only},
          'inputs': {k: {'path': str(v.resolve()), 'sha256': sha(v), 'bytes': v.stat().st_size}
                     for k,v in {**paths, **approval_inputs, 'delta-recurrence': delta_r}.items() if k != 'out'},
          'virtual_prefix': {
            'native': {'path': str(paths['base-native'].resolve()), 'sha256': sha(paths['base-native']),
                       'bytes': paths['base-native'].stat().st_size,
                       'rows': len(base_n), 'train': sum(r.get('split') == 'train' for r in base_n),
                       'test': sum(r.get('split') == 'test' for r in base_n)},
            'recurrence': {'path': str(paths['base-recurrence'].resolve()), 'sha256': sha(paths['base-recurrence']),
                           'bytes': paths['base-recurrence'].stat().st_size, 'rows': len(base_r)},
            'native_pieces': {'path': str(paths['base-native-pieces'].resolve()),
                              'sha256': sha(paths['base-native-pieces']), 'rows': len(merged_pieces)-len(delta_piece_names)+len(exact_piece_duplicates)}},
          'native_split_group_audit_virtual': split_audit,
          'piece_merge': {'added': len(delta_piece_names)-len(exact_piece_duplicates),
                          'exact_duplicates': sorted(exact_piece_duplicates), 'virtual_total': len(merged_pieces)},
          'recurrence_audit': recurrence_audit,
          'outputs': {name: {'sha256': sha(out/name), 'bytes': (out/name).stat().st_size}
                      for name in names.values()},
          'assembler': {'path': str(Path(__file__).relative_to(CODE_ROOT)), 'sha256': sha(Path(__file__))},
          'auditor': {'path': str(audit_script), 'sha256': sha(audit_script)},
          'limits': {'full_base_assembly_performed': False, 'task_or_trajectory_admission': False,
                     'model_qualification': False, 'publication': False, 'training_launch': False}}
        (out/'proposal-manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+'\n')
        print(json.dumps({'status': manifest['status'], 'native': split_audit,
                          'delta_rows': len(delta_n), 'recurrence_unchanged_rows': len(base_r),
                          'proposal_manifest_sha256': sha(out/'proposal-manifest.json')}, indent=2))
        return

    names = {'native': 'native-records.jsonl', 'recurrence': 'recurrence-records.jsonl',
             'native_pieces': 'native-pieces.jsonl', 'recurrence_pieces': 'recurrence-pieces.jsonl',
             'delta': 'delta-native-records.jsonl', 'audit': 'recurrence-audit.json'}
    write_append(paths['base-native'], out / names['native'], delta_n)
    unchanged_transport = {}
    if native_only:
        unchanged_transport['recurrence'] = link_or_copy_unchanged(
            paths['base-recurrence'], out / names['recurrence'])
    else:
        write_append(paths['base-recurrence'], out / names['recurrence'], delta_recs)
    np = merge_pieces(paths['base-native-pieces'], paths['delta-pieces'], out / names['native_pieces'])
    if native_only:
        unchanged_transport['recurrence_pieces'] = link_or_copy_unchanged(
            paths['base-recurrence-pieces'], out / names['recurrence_pieces'])
        rp = {'added': 0, 'exact_duplicates': [], 'total': len(list(rows(paths['base-recurrence-pieces'], 'name')))}
    else:
        rp = merge_pieces(paths['base-recurrence-pieces'], paths['delta-pieces'], out / names['recurrence_pieces'])
    available = {piece['name'] for piece in rows(out / names['native_pieces'], 'name')}
    for row in delta_n:
        missing = referenced_soft(row.get('messages')) - available
        if missing: raise ValueError(f'missing merged pieces for {row["id"]}: {sorted(missing)}')
    with (out / names['delta']).open('xb') as f:
        for row in delta_n: f.write(line(row))
    split_audit = audit_splits(out / names['native'])
    audit_path = out / names['audit']
    subprocess.run([sys.executable, str(audit_script),
                    str(out/names['recurrence']), '--out', str(audit_path)], cwd=repo_root, check=True)
    recurrence_audit = json.loads(audit_path.read_text())
    if recurrence_audit.get('structurally_closed') is not True:
        raise ValueError('combined recurrence records are not structurally closed')
    manifest = {
      'schema': 'natlang.approved-neuralese-cohort-assembly-proposal/1',
      'status': 'proposal-only; separate root publication/admission review required',
      'approval': {**approval_summary,
                   'id_field': ('rows.native_id under receipt-specific strict decision/training_admission predicates'
                                if root_action_admission or root_per_action_admission else args.approval_id_field)},
      'admitted_facets': {'native': True, 'recurrence': not native_only,
                          'native_only_receipt': native_only},
      'repo_root': str(repo_root),
      'inputs': {k: {'path': str(v.resolve()), 'sha256': sha(v), 'bytes': v.stat().st_size}
                 for k,v in {**paths, **approval_inputs, 'delta-recurrence': delta_r}.items() if k != 'out'},
      'prefixes_byte_exact': {
        'native': prefix_sha(out/names['native'], paths['base-native'].stat().st_size) == sha(paths['base-native']),
        'recurrence': prefix_sha(out/names['recurrence'], paths['base-recurrence'].stat().st_size) == sha(paths['base-recurrence']),
        'native_pieces': prefix_sha(out/names['native_pieces'], paths['base-native-pieces'].stat().st_size) == sha(paths['base-native-pieces']),
        'recurrence_pieces': prefix_sha(out/names['recurrence_pieces'], paths['base-recurrence-pieces'].stat().st_size) == sha(paths['base-recurrence-pieces'])},
      'native_split_group_audit': split_audit, 'piece_merge': {'native': np, 'recurrence': rp},
      'unchanged_artifact_transport': unchanged_transport,
      'recurrence_audit': recurrence_audit,
      'outputs': {name: {'sha256': sha(out/name), 'bytes': (out/name).stat().st_size}
                  for name in names.values()},
      'assembler': {'path': str(Path(__file__).relative_to(CODE_ROOT)), 'sha256': sha(Path(__file__))},
      'auditor': {'path': str(audit_script), 'sha256': sha(audit_script)},
      'limits': {'task_or_trajectory_admission': False, 'model_qualification': False,
                 'publication': False, 'training_launch': False}}
    (out/'proposal-manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({'status': manifest['status'], 'native': split_audit,
                      'recurrence_structurally_closed': recurrence_audit['structurally_closed'],
                      'proposal_manifest_sha256': sha(out/'proposal-manifest.json')}, indent=2))

if __name__ == '__main__': main()
