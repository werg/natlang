"""Fail-closed resolution for catalogued, source-reviewed training-turn replacements."""
import hashlib
import json
from pathlib import Path
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'training' / 'neuralese'))
from natlang_neuralese.common.hashing import sha256_hex as _sha  # noqa: E402


def _validate_replacement(repo, original_relative, replacement_relative, decision):
    if decision.get('status') != 'source_review_approved':
        raise ValueError(f"source-reviewed training-turn replacement is not approved: {original_relative}")
    manifest_name = decision.get('review_manifest')
    manifest_sha = decision.get('review_manifest_sha256')
    if not manifest_name or not manifest_sha:
        raise ValueError(f"source-reviewed replacement lacks a catalog-pinned approved manifest: {original_relative}")
    manifest_path = repo / manifest_name
    original_path = repo / original_relative
    replacement_path = repo / replacement_relative
    for candidate in (manifest_path, original_path, replacement_path):
        if not candidate.resolve().is_relative_to(repo) or candidate.is_symlink():
            raise ValueError(f"source-reviewed replacement path escapes or aliases repository: {candidate}")
    if not manifest_path.is_file() or not original_path.is_file() or not replacement_path.is_file():
        raise ValueError(f"source-reviewed training-turn replacement evidence is missing: {original_relative}")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    original_bytes = original_path.read_bytes()
    replacement_bytes = replacement_path.read_bytes()
    if (_sha(manifest_bytes) != manifest_sha or
            manifest.get('status') != 'source_reviewed_replacement_approved' or
            manifest.get('original_artifact') != original_relative or
            manifest.get('original_sha256') != _sha(original_bytes) or
            manifest.get('replacement_artifact') != replacement_relative or
            manifest.get('replacement_sha256') != _sha(replacement_bytes)):
        raise ValueError(f"source-reviewed training-turn replacement identity/hash mismatch: {original_relative}")
    old_rows = original_bytes.splitlines()
    new_rows = replacement_bytes.splitlines()
    retained = manifest.get('retained_turns', [])
    excluded = manifest.get('excluded_turns', [])
    if (len(old_rows) != manifest.get('original_turn_count') or
            len(new_rows) != manifest.get('replacement_turn_count') or len(retained) != len(new_rows)):
        raise ValueError(f"source-reviewed training-turn replacement count mismatch: {original_relative}")
    retained_indexes, retained_bytes = set(), []
    for item in retained:
        index = item.get('original_line', 0) - 1
        if index < 0 or index >= len(old_rows) or index in retained_indexes:
            raise ValueError(f"source-reviewed retained-row index mismatch: {original_relative}")
        row = json.loads(old_rows[index])
        if (_sha(old_rows[index]) != item.get('original_line_sha256') or
                row.get('id') != item.get('id') or
                row.get('teacher_trajectory_digest') != item.get('trajectory_digest')):
            raise ValueError(f"source-reviewed retained-row evidence mismatch: {original_relative}")
        retained_indexes.add(index)
        retained_bytes.append(old_rows[index])
    excluded_indexes = set()
    for item in excluded:
        index = item.get('original_line', 0) - 1
        if index < 0 or index >= len(old_rows) or index in excluded_indexes:
            raise ValueError(f"source-reviewed excluded-row index mismatch: {original_relative}")
        row = json.loads(old_rows[index])
        if (_sha(old_rows[index]) != item.get('original_line_sha256') or
                row.get('id') != item.get('id') or
                row.get('teacher_trajectory_digest') != item.get('trajectory_digest')):
            raise ValueError(f"source-reviewed excluded-row evidence mismatch: {original_relative}")
        excluded_indexes.add(index)
    if (retained_indexes & excluded_indexes or
            retained_indexes | excluded_indexes != set(range(len(old_rows))) or
            retained_bytes != new_rows):
        raise ValueError(f"source-reviewed replacement does not exactly partition original rows: {original_relative}")
    return {'original': original_relative, 'replacement': replacement_relative,
            'review_manifest': manifest_name, 'review_manifest_sha256': manifest_sha,
            'excluded_turns': len(excluded), 'replacement_sha256': _sha(replacement_bytes)}


def resolve_reviewed_turn_inputs(repo, paths, *, allow_coalesce=False):
    """Resolve reviewed hold mappings and audit any repeated source aliases.

    Review replacements must have explicit approval and a manifest SHA pinned in
    training/data_sources.json. Default input assembly may coalesce an old path
    plus its approved replacement target; caller-supplied overrides remain strict.
    """
    repo = Path(repo).resolve()
    policy = json.loads((repo / 'training/data_sources.json').read_text())
    replacements = policy.get('replacements', {})
    decisions = policy.get('decisions', [])
    resolved, replacements_audit, seen = [], [], {}
    for raw_path in paths:
        path = Path(raw_path).resolve()
        try:
            relative = path.relative_to(repo).as_posix()
        except ValueError:
            relative = None
        if relative is None:
            target_path = path
        else:
            held_decision = next((d for d in decisions if d.get('glob') == relative and
                                  (d.get('status', '').startswith('source_review_hold') or d.get('status') == 'source_review_approved')), None)
            if held_decision:
                target_relative = replacements.get(relative)
                if not target_relative or target_relative != held_decision.get('replacement'):
                    raise ValueError(f"held training-turn artifact has no exact replacement: {relative}")
                replacements_audit.append(_validate_replacement(repo, relative, target_relative, held_decision))
                target_path = repo / target_relative
            else:
                target_path = path
                original = next((old for old, new in replacements.items() if new == relative), None)
                if original:
                    decision = next((d for d in decisions if d.get('glob') == original and
                                     (d.get('status', '').startswith('source_review_hold') or d.get('status') == 'source_review_approved')), None)
                    if not decision:
                        raise ValueError(f"reviewed target has no matching source-review decision: {relative}")
                    replacements_audit.append(_validate_replacement(repo, original, relative, decision))
                elif any(d.get('glob') == relative and (d.get('status', '').startswith('source_review_hold') or d.get('status') == 'source_review_approved')
                         for d in decisions):
                    raise ValueError(f"training-turn artifact is source-review held: {relative}")
        key = str(target_path.resolve())
        if key in seen:
            if not allow_coalesce:
                raise ValueError(f'duplicate training-turn input after replacement resolution: {key}')
            seen[key]['aliases'].append(str(path))
            continue
        seen[key] = {'path': key, 'aliases': [str(path)]}
        resolved.append(key)
    return resolved, replacements_audit, [entry for entry in seen.values() if len(entry['aliases']) > 1]
