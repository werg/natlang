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

ROOT = Path(__file__).resolve().parents[1]


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


def audit_splits(path: Path):
    counts = collections.Counter()
    group_splits = collections.defaultdict(set)
    source_splits = {}
    ids = set()
    for row in rows(path):
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
                 'base-native-pieces', 'base-recurrence-pieces', 'approval', 'out'):
        p.add_argument('--' + name, required=True, type=Path)
    p.add_argument('--delta-recurrence', type=Path,
                   help='defaults to --delta-native when the approved records feed both streams')
    p.add_argument('--approval-id-field', default='approved_row_ids')
    p.add_argument('--admission-kind', default='exact-native-runtime-oracle')
    args = p.parse_args()
    paths = {k: getattr(args, k.replace('-', '_')) for k in ('base-native','base-recurrence','base-receipt','delta-native',
        'delta-pieces','base-native-pieces','base-recurrence-pieces','approval','out')}
    delta_r = args.delta_recurrence or args.delta_native
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()): raise ValueError(f'output directory is not empty: {out}')
    out.mkdir(parents=True, exist_ok=True)

    approval = json.loads(paths['approval'].read_text())
    base_receipt = json.loads(paths['base-receipt'].read_text())
    root_corpus_receipt = base_receipt.get('schema') == 'natlang.root-corpus-admission/1'
    prefix_binding = base_receipt.get('schema') == 'natlang.corpus-prefix-binding/1'
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
    root_action_admission = approval.get('schema') == 'natlang.root-selected-action-admission/1'
    if root_action_admission:
        if args.approval_id_field != 'approved_row_ids':
            raise ValueError('root action admission uses its fixed rows schema')
        if approval.get('qualifications', {}).get('recurrence') is True:
            raise ValueError('native-only selected-action receipt unexpectedly claims recurrence qualification')
        approval_rows = approval.get('rows')
        if not isinstance(approval_rows, list) or not approval_rows:
            raise ValueError('root action admission has no selected rows')
        approved = [entry.get('native_id') for entry in approval_rows]
        if any(entry.get('decision') != 'admit-exact-selected-native-action-SFT-only' for entry in approval_rows):
            raise ValueError('root action admission contains an unsupported decision')
        counts = approval.get('counts') or {}
        if (counts.get('native_SFT_train_actions') != len(approval_rows)
                or counts.get('whole_trajectories') != 0
                or approval.get('qualifications', {}).get('learned_writer') is not False):
            raise ValueError('root action receipt does not describe exact native SFT-only admission')
        approval_by_id = {entry['native_id']: entry for entry in approval_rows}
    else:
        approved = approval.get(args.approval_id_field)
        approval_by_id = {}
    if not isinstance(approved, list) or not approved or any(not isinstance(x, str) for x in approved):
        raise ValueError(f'approval receipt lacks nonempty {args.approval_id_field!r}')
    if len(set(approved)) != len(approved): raise ValueError('duplicate approved IDs')
    # If the receipt provides an artifacts mapping, enforce hashes for every named input it binds.
    artifact_hashes = approval.get('artifact_hashes', {})
    for raw, expected in artifact_hashes.items():
        bound = (ROOT / raw).resolve()
        if not bound.is_file() or sha(bound) != expected: raise ValueError(f'approval artifact missing/hash mismatch: {raw}')
    base_n = list(rows(paths['base-native'])); base_r = list(rows(paths['base-recurrence']))
    delta_n = list(rows(paths['delta-native']))
    delta_recs = [] if root_action_admission and args.delta_recurrence is None else list(rows(delta_r))
    ids = [r['id'] for r in delta_n]
    if set(ids) != set(approved) or len(ids) != len(approved):
        raise ValueError('delta native IDs do not equal approved IDs exactly')
    native_only = root_action_admission
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
        if root_action_admission:
            decision = approval_by_id[row['id']]
            if target_digest(row) != decision.get('target_sha256'):
                raise ValueError(f'root admission target digest mismatch: {row["id"]}')
            decision_groups = decision.get('source_groups')
            if decision_groups is None and isinstance(decision.get('source_group'), str):
                decision_groups = [decision['source_group']]
            if (row.get('split') != decision.get('split') or not isinstance(decision_groups, list) or
                    not decision_groups or decision_groups != (row.get('source_groups') or [])):
                raise ValueError(f'root admission source split/group mismatch: {row["id"]}')
            row['training_admission'] = {
                'approved': True, 'kind': 'root-selected-native-action-sft-only',
                'status': 'admitted-exact-selected-native-action',
                'root_admission_sha256': sha(paths['approval']),
                'decision': decision['decision'], 'role': decision.get('role')}
            row['decision'] = {**(row.get('decision') or {}), 'training_approved': True,
                               'root_action_admission_sha256': sha(paths['approval'])}
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

    names = {'native': 'native-records.jsonl', 'recurrence': 'recurrence-records.jsonl',
             'native_pieces': 'native-pieces.jsonl', 'recurrence_pieces': 'recurrence-pieces.jsonl',
             'delta': 'delta-native-records.jsonl', 'audit': 'recurrence-audit.json'}
    write_append(paths['base-native'], out / names['native'], delta_n)
    if native_only:
        shutil.copyfile(paths['base-recurrence'], out / names['recurrence'])
    else:
        write_append(paths['base-recurrence'], out / names['recurrence'], delta_recs)
    np = merge_pieces(paths['base-native-pieces'], paths['delta-pieces'], out / names['native_pieces'])
    if native_only:
        shutil.copyfile(paths['base-recurrence-pieces'], out / names['recurrence_pieces'])
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
    subprocess.run([sys.executable, str(ROOT/'scripts/audit_neuralese_recurrence.py'),
                    str(out/names['recurrence']), '--out', str(audit_path)], cwd=ROOT, check=True)
    recurrence_audit = json.loads(audit_path.read_text())
    if recurrence_audit.get('structurally_closed') is not True:
        raise ValueError('combined recurrence records are not structurally closed')
    manifest = {
      'schema': 'natlang.approved-neuralese-cohort-assembly-proposal/1',
      'status': 'proposal-only; separate root publication/admission review required',
      'approval': {'path': str(paths['approval'].resolve()), 'sha256': sha(paths['approval']),
                   'id_field': args.approval_id_field, 'approved_ids': approved},
      'admitted_facets': {'native': True, 'recurrence': not native_only,
                          'native_only_receipt': native_only},
      'inputs': {k: {'path': str(v.resolve()), 'sha256': sha(v), 'bytes': v.stat().st_size}
                 for k,v in {**paths, 'delta-recurrence': delta_r}.items() if k != 'out'},
      'prefixes_byte_exact': {
        'native': prefix_sha(out/names['native'], paths['base-native'].stat().st_size) == sha(paths['base-native']),
        'recurrence': prefix_sha(out/names['recurrence'], paths['base-recurrence'].stat().st_size) == sha(paths['base-recurrence']),
        'native_pieces': prefix_sha(out/names['native_pieces'], paths['base-native-pieces'].stat().st_size) == sha(paths['base-native-pieces']),
        'recurrence_pieces': prefix_sha(out/names['recurrence_pieces'], paths['base-recurrence-pieces'].stat().st_size) == sha(paths['base-recurrence-pieces'])},
      'native_split_group_audit': split_audit, 'piece_merge': {'native': np, 'recurrence': rp},
      'recurrence_audit': recurrence_audit,
      'outputs': {name: {'sha256': sha(out/name), 'bytes': (out/name).stat().st_size}
                  for name in names.values()},
      'assembler': {'path': str(Path(__file__).relative_to(ROOT)), 'sha256': sha(Path(__file__))},
      'limits': {'task_or_trajectory_admission': False, 'model_qualification': False,
                 'publication': False, 'training_launch': False}}
    (out/'proposal-manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({'status': manifest['status'], 'native': split_audit,
                      'recurrence_structurally_closed': recurrence_audit['structurally_closed'],
                      'proposal_manifest_sha256': sha(out/'proposal-manifest.json')}, indent=2))

if __name__ == '__main__': main()
