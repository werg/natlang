"""Promote one exact completed admission checkpoint across a code-only binding change.

This utility never edits the old checkpoint.  The narrowly scoped caller must
prove the old/new verifier transition and the outer source/model identity
before invoking ``promote_checkpoint_file``.
"""
from __future__ import annotations

import json
import argparse
import hashlib
from pathlib import Path
import subprocess

from natlang_neuralese.common.hashing import canonical_json_sha256_hex, sha256_file_hex_cached
from natlang_neuralese.common.jsonio import canonical_json_bytes, write_json_durable
from scripts.neuralese_data.admission_resume import SCHEMA, save_model_checkpoint


SHARD_SCHEMA = 'natlang.acquired-admission-shard-checkpoint-identity/1'
REPO = Path(__file__).resolve().parents[2]
MODULE_RELATIVE = 'scripts/neuralese_data/admission_resume.py'


def verify_artifact_closure(identity):
    """Strongly verify the identity's complete artifact inventory once."""
    model_identity = identity.get('model_identity', {})
    artifacts = {}
    for item in [*(identity.get('artifacts') or []), *(model_identity.get('artifacts') or [])]:
        if not isinstance(item, dict) or not isinstance(item.get('path'), str):
            raise ValueError('checkpoint has a malformed artifact inventory')
        path = Path(item['path']).resolve(strict=True)
        expected = (item.get('sha256'), item.get('bytes'))
        previous = artifacts.get(path)
        if previous is not None and previous != expected:
            raise ValueError(f'checkpoint repeats an artifact with conflicting bindings: {path}')
        artifacts[path] = expected
    for path, (digest, size) in artifacts.items():
        if path.stat().st_size != size or sha256_file_hex_cached(
                path, trust_archive_receipt=False) != digest:
            raise ValueError(f'bound checkpoint artifact changed: {path}')
    return len(artifacts)


def promote_checkpoint_file(old_path, output_dir, *, old_binding, new_binding):
    """Write a new-key checkpoint after changing exactly one verifier binding."""
    old_path = Path(old_path).resolve(strict=True)
    payload = json.loads(old_path.read_text(encoding='utf-8'))
    if payload.get('schema') != SCHEMA:
        raise ValueError('old admission checkpoint schema differs')
    identity = payload.get('identity')
    result = payload.get('result')
    if not isinstance(identity, dict) or identity.get('schema') != SHARD_SCHEMA:
        raise ValueError('checkpoint is not a completed acquired-admission shard')
    artifact_count = verify_artifact_closure(identity)
    return _promote_checkpoint_payload(old_path, identity, result, output_dir,
        old_binding=old_binding, new_binding=new_binding,
        verified_artifact_count=artifact_count)


def _promote_checkpoint_payload(old_path, identity, result, output_dir, *, old_binding,
                                new_binding, verified_artifact_count):
    """Create the new content-addressed copy from an already verified payload."""
    result_sha = canonical_json_sha256_hex(result)
    old_key = canonical_json_sha256_hex(identity)
    if old_path.name != f'{old_key}-{result_sha}.json':
        raise ValueError('old admission checkpoint filename is not content-addressed')

    model_identity = identity.get('model_identity')
    verifier_files = model_identity.get('verifier_files') if isinstance(model_identity, dict) else None
    if not isinstance(verifier_files, list):
        raise ValueError('shard checkpoint lacks its model verifier inventory')
    matches = [i for i, item in enumerate(verifier_files)
               if isinstance(item, dict) and item.get('path') == old_binding.get('path')]
    if len(matches) != 1 or verifier_files[matches[0]] != old_binding:
        raise ValueError('old verifier binding is absent, duplicated, or differs from approved bytes')

    promoted = json.loads(canonical_json_bytes(identity))
    promoted_files = promoted['model_identity']['verifier_files']
    promoted_matches = [i for i, item in enumerate(promoted_files)
                        if item.get('path') == old_binding['path']]
    if len(promoted_matches) != 1:
        raise ValueError('promoted identity lost its unique verifier binding')
    promoted_files[promoted_matches[0]] = dict(new_binding)
    # The entire identity must be equal after undoing only this single change.
    reverted = json.loads(canonical_json_bytes(promoted))
    reverted['model_identity']['verifier_files'][promoted_matches[0]] = dict(old_binding)
    if canonical_json_bytes(reverted) != canonical_json_bytes(identity):
        raise ValueError('promotion changes identity fields beyond the exact verifier binding')

    new_path = save_model_checkpoint(output_dir, promoted, result)
    return {
        'old_checkpoint': {'path': str(old_path), 'bytes': old_path.stat().st_size,
                           'sha256': sha256_file_hex_cached(old_path, trust_archive_receipt=False)},
        'new_checkpoint': {'path': str(new_path), 'bytes': new_path.stat().st_size,
                           'sha256': sha256_file_hex_cached(new_path, trust_archive_receipt=False)},
        'old_identity_sha256': old_key,
        'new_identity_sha256': canonical_json_sha256_hex(promoted),
        'result_sha256': result_sha,
        'range': identity.get('range'),
        'verified_artifact_count': verified_artifact_count,
        'training_admission': False,
    }


def _verify_bound_checkpoint(path, *, binding, export_path, model_name, expected_range,
                            normalized_source_sha256):
    path = Path(path).resolve(strict=True)
    payload = json.loads(path.read_text(encoding='utf-8'))
    identity, result = payload.get('identity'), payload.get('result')
    if payload.get('schema') != SCHEMA or not isinstance(identity, dict):
        raise ValueError(f'checkpoint payload is malformed: {path}')
    model_identity = identity.get('model_identity')
    files = model_identity.get('verifier_files') if isinstance(model_identity, dict) else None
    if not isinstance(files, list) or files.count(binding) != 1:
        raise ValueError(f'checkpoint does not carry the exact expected verifier binding: {path}')
    source = model_identity.get('source', {})
    model = model_identity.get('model', {})
    interval = identity.get('range', {})
    actual_range = (interval.get('start'), interval.get('stop'))
    if (identity.get('schema') != SHARD_SCHEMA or model.get('name') != model_name
            or source.get('export_path') != str(Path(export_path).resolve(strict=True))
            or source.get('source_output_sha256') != normalized_source_sha256
            or actual_range != expected_range):
        raise ValueError(f'checkpoint is outside the requested source/model/range: {path}')
    result_sha = canonical_json_sha256_hex(result)
    identity_sha = canonical_json_sha256_hex(identity)
    if path.name != f'{identity_sha}-{result_sha}.json':
        raise ValueError(f'checkpoint filename is not content-addressed: {path}')
    # Validate each copy's complete artifact closure, even when it is already
    # promoted.  This makes a rerun after interruption an authenticated resume.
    verified_artifact_count = verify_artifact_closure(identity)
    return {'identity': identity, 'result': result, 'identity_sha256': identity_sha,
            'result_sha256': result_sha, 'path': path,
            'verified_artifact_count': verified_artifact_count}


def _promotion_record(old, new, source_range):
    return {
        'old_checkpoint': None if old is None else {
            'path': str(old['path']),
            'sha256': sha256_file_hex_cached(old['path'], trust_archive_receipt=False)},
        'new_checkpoint': {
            'path': str(new['path']),
            'sha256': sha256_file_hex_cached(new['path'], trust_archive_receipt=False)},
        'old_identity_sha256': None if old is None else old['identity_sha256'],
        'new_identity_sha256': new['identity_sha256'],
        'result_sha256': new['result_sha256'],
        'range': {'start': source_range[0], 'stop': source_range[1]},
        'verified_artifact_count': new['verified_artifact_count'],
        'training_admission': False,
    }


def promote_checkpoint_set(*, checkpoint_root, export_path, model_name, old_binding,
                           new_binding, expected_ranges, normalized_source_sha256):
    """Promote/resume a complete shard set, authenticating old and new copies."""
    checkpoint_root = Path(checkpoint_root).resolve(strict=True)
    export_path = Path(export_path).resolve(strict=True)
    shards = checkpoint_root / 'shards'
    paths = sorted(shards.glob('*/*.json'))
    by_range = {}
    for path in paths:
        # Ignore unrelated checkpoint files only when their shard schema is not
        # this operation's scope; once a source/model shard is found, validate
        # all of its identity and artifact bindings strictly below.
        payload = json.loads(path.read_text(encoding='utf-8'))
        identity = payload.get('identity', {})
        if identity.get('schema') != SHARD_SCHEMA:
            continue
        model_identity = identity.get('model_identity', {})
        model = model_identity.get('model', {})
        source = model_identity.get('source', {})
        interval = identity.get('range', {})
        current_range = (interval.get('start'), interval.get('stop'))
        if (model.get('name') != model_name
                or source.get('export_path') != str(export_path)
                or current_range not in expected_ranges):
            continue
        files = model_identity.get('verifier_files', [])
        if old_binding not in files and new_binding not in files:
            continue
        kind = 'old' if old_binding in files else 'new'
        verified = _verify_bound_checkpoint(path, binding=old_binding if kind == 'old' else new_binding,
                                            export_path=export_path, model_name=model_name,
                                            expected_range=current_range,
                                            normalized_source_sha256=normalized_source_sha256)
        entries = by_range.setdefault(current_range, {})
        if kind in entries:
            raise ValueError(f'duplicate {kind} checkpoint for source range {current_range}')
        entries[kind] = verified

    if set(by_range) != set(expected_ranges):
        missing = sorted(set(expected_ranges) - set(by_range))
        raise ValueError(f'shard checkpoint ranges are incomplete; missing={missing}')

    promoted_rows = []
    for source_range in sorted(expected_ranges):
        entries = by_range[source_range]
        old = entries.get('old')
        existing_new = entries.get('new')
        if existing_new is not None and old is None:
            raise ValueError(f'promoted checkpoint lacks its immutable old parent: {source_range}')
        if old is not None and existing_new is not None:
            changed = json.loads(canonical_json_bytes(existing_new['identity']))
            old_identity = json.loads(canonical_json_bytes(old['identity']))
            new_files = changed['model_identity']['verifier_files']
            old_files = old_identity['model_identity']['verifier_files']
            old_matches = [i for i, item in enumerate(old_files) if item == old_binding]
            new_matches = [i for i, item in enumerate(new_files) if item == new_binding]
            if len(old_matches) != 1 or new_matches != old_matches:
                raise ValueError(f'old/new checkpoint binding transition differs for {source_range}')
            old_files[old_matches[0]] = dict(new_binding)
            if canonical_json_bytes(old_identity) != canonical_json_bytes(changed):
                raise ValueError(f'old/new checkpoint identity changed outside verifier binding: {source_range}')
            if canonical_json_bytes(old['result']) != canonical_json_bytes(existing_new['result']):
                raise ValueError(f'old/new checkpoint results differ for {source_range}')
            promoted = _promotion_record(old, existing_new, source_range)
        else:
            created = _promote_checkpoint_payload(
                old['path'], old['identity'], old['result'], old['path'].parent,
                old_binding=old_binding, new_binding=new_binding,
                verified_artifact_count=old['verified_artifact_count'])
            new = _verify_bound_checkpoint(
                created['new_checkpoint']['path'], binding=new_binding,
                export_path=export_path, model_name=model_name,
                expected_range=source_range,
                normalized_source_sha256=normalized_source_sha256)
            promoted = _promotion_record(old, new, source_range)
        promoted_rows.append(promoted)
    return promoted_rows


def verify_pinned_verifier_revisions(repo_root, *, old_revision, new_revision,
                                     old_sha256, old_bytes, new_sha256, new_bytes):
    """Pin exact verifier blobs while allowing later unrelated HEAD advances."""
    repo = Path(repo_root).resolve(strict=True)

    def git(*args):
        return subprocess.run(['git', '-C', str(repo), *args], check=True,
                              stdout=subprocess.PIPE).stdout

    current_revision = git('rev-parse', 'HEAD').decode().strip()
    ancestry = subprocess.run(['git', '-C', str(repo), 'merge-base', '--is-ancestor',
                               new_revision, current_revision],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if ancestry.returncode != 0:
        raise ValueError('approved new verifier revision is not an ancestor of checkout HEAD')
    old_module = git('show', f'{old_revision}:{MODULE_RELATIVE}')
    approved_module = git('show', f'{new_revision}:{MODULE_RELATIVE}')
    if (hashlib.sha256(old_module).hexdigest(), len(old_module)) != (old_sha256, old_bytes):
        raise ValueError('old revision does not contain the exactly pinned verifier bytes')
    if (hashlib.sha256(approved_module).hexdigest(), len(approved_module)) != (new_sha256, new_bytes):
        raise ValueError('new revision does not contain the exactly pinned verifier bytes')
    module_path = repo / MODULE_RELATIVE
    if module_path.read_bytes() != approved_module:
        raise ValueError('checkout verifier bytes differ from the pinned reviewed revision')
    return ({'path': str(module_path), 'sha256': old_sha256, 'bytes': old_bytes},
            {'path': str(module_path), 'sha256': new_sha256, 'bytes': new_bytes},
            current_revision)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=REPO)
    parser.add_argument('--checkpoint-root', type=Path, required=True)
    parser.add_argument('--export-path', type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--old-revision', required=True)
    parser.add_argument('--new-revision', required=True)
    parser.add_argument('--old-sha256', required=True)
    parser.add_argument('--old-bytes', type=int, required=True)
    parser.add_argument('--new-sha256', required=True)
    parser.add_argument('--new-bytes', type=int, required=True)
    parser.add_argument('--range-stop', type=int, required=True)
    parser.add_argument('--range-size', type=int, default=50)
    parser.add_argument('--export-sha256', required=True)
    parser.add_argument('--normalized-source-sha256', required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo_root.resolve(strict=True)
    old_binding, new_binding, current_revision = verify_pinned_verifier_revisions(
        repo, old_revision=args.old_revision, new_revision=args.new_revision,
        old_sha256=args.old_sha256, old_bytes=args.old_bytes,
        new_sha256=args.new_sha256, new_bytes=args.new_bytes)
    if args.range_size <= 0 or args.range_stop <= 0 or args.range_stop % args.range_size:
        raise ValueError('range stop must be a positive multiple of range size')
    expected_ranges = {(start, start + args.range_size)
                       for start in range(0, args.range_stop, args.range_size)}
    export_path = args.export_path.resolve(strict=True)
    export_sha = sha256_file_hex_cached(export_path, trust_archive_receipt=False)
    if export_sha != args.export_sha256:
        raise ValueError('source export bytes differ from the pinned export SHA-256')
    promoted_rows = promote_checkpoint_set(
        checkpoint_root=args.checkpoint_root, export_path=export_path,
        model_name=args.model, old_binding=old_binding, new_binding=new_binding,
        expected_ranges=expected_ranges,
        normalized_source_sha256=args.normalized_source_sha256)
    receipt = {
        'schema': 'natlang.acquired-admission-checkpoint-code-promotion/1',
        'source': {'export': str(export_path), 'sha256': export_sha,
                   'normalized_source_sha256': args.normalized_source_sha256},
        'model': args.model,
        'ranges': [row['range'] for row in promoted_rows],
        'transition': {'old_revision': args.old_revision, 'old_binding': old_binding,
                       'new_revision': args.new_revision, 'new_binding': new_binding},
        'promoted_checkpoints': promoted_rows,
        'training_admission': False,
        'training_exposure': 0,
    }
    if args.receipt.exists():
        if canonical_json_bytes(json.loads(args.receipt.read_text())) != canonical_json_bytes(receipt):
            raise ValueError('an existing checkpoint promotion receipt has different content')
    else:
        write_json_durable(args.receipt, receipt)
    print(json.dumps({'receipt': str(args.receipt), 'promoted': len(promoted_rows),
                      'ranges': receipt['ranges'], 'training_exposure': 0}, sort_keys=True))


if __name__ == '__main__':
    main()
