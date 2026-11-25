"""Held identity of live catalogs, distinct from explicitly inventoried execution metadata."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath

from natlang_neuralese.common.jsonio import canonical_json_bytes

# Host collect-live-workflow-teacher.mjs list output and release entry enrichment.
# Everything else, including unknown future fields and non-goal inputs, remains identity.
EXECUTION_FIELDS = frozenset({'requiresReview', 'devices', 'requiresAndroidEmulator',
                              'requiresDeploymentPackage', 'workspaceBaseline'})
ENTRY_EXECUTION_FIELDS = frozenset({'program', 'callable', 'resultType', 'skill'})


def adapter_defaults(entry):
    """Validate explicit producer provenance without treating non-null evidence as metadata."""
    declared = entry.get('adapterDefaults')
    if 'adapterDefaults' not in entry:
        return {}
    if not isinstance(declared, dict) or set(declared) != {'values', 'source'}:
        raise ValueError('invalid entry adapter defaults')
    values, source = declared['values'], declared['source']
    if (not isinstance(values, dict) or not values or 'goal' in values or
            any(not isinstance(name, str) or not name.isidentifier() for name in values) or
            not isinstance(source, dict) or set(source) != {'path', 'sha256'}):
        raise ValueError('invalid entry adapter default provenance')
    path, digest = source['path'], source['sha256']
    if (not isinstance(path, str) or not path or PurePosixPath(path).is_absolute() or
            '..' in PurePosixPath(path).parts or path == '.' or ':' in path or '\\' in path or
            str(PurePosixPath(path)) != path or
            not isinstance(digest, str) or len(digest) != 64 or
            any(c not in '0123456789abcdef' for c in digest)):
        raise ValueError('invalid entry adapter default source')
    args, encoded = entry.get('arguments'), entry.get('argumentJson')
    names, inputs = entry.get('parameterNames'), entry.get('inputValues')
    if (not isinstance(args, dict) or not isinstance(encoded, dict) or
            not isinstance(names, list) or not isinstance(inputs, list) or
            len(names) != len(inputs) or any(not isinstance(name, str) for name in names) or
            len(set(names)) != len(names)):
        raise ValueError('adapter defaults require complete actual input bindings')
    for name, value in values.items():
        if name not in args or name not in encoded or name not in names:
            raise ValueError('adapter default is absent from actual input bindings')
        try:
            representations = [args[name], json.loads(encoded[name]), inputs[names.index(name)]]
        except (ValueError, TypeError, IndexError) as error:
            raise ValueError('invalid adapter default argument encoding') from error
        if any(canonical_json_bytes(actual) != canonical_json_bytes(value) for actual in representations):
            raise ValueError('adapter default differs from actual input bindings')
    return values


def catalog_rows(path, *, lines=None):
    from live_teacher_sources import open_goal_fingerprint
    rows, seen = [], set()
    for line in lines if lines is not None else Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError('catalog row must be an object')
        task = row.get('taskId')
        if not isinstance(task, str) or not task or task in seen:
            raise ValueError('invalid or duplicate catalog taskId')
        seen.add(task)
        for key in EXECUTION_FIELDS & row.keys():
            value = row[key]
            if key == 'devices':
                valid = isinstance(value, list) and all(isinstance(v, str) for v in value)
            elif key == 'workspaceBaseline':
                valid = isinstance(value, str) and bool(value)
            else:
                valid = type(value) is bool
            if not valid:
                raise ValueError(f'invalid execution metadata {key}: {task}')
        if row.get('interaction') == 'open-user-goal':
            groups, digest = row.get('sourceGroups'), row.get('goalSha256')
            if (not isinstance(groups, list) or not groups or
                    any(not isinstance(g, str) or not g.strip() for g in groups) or len(set(groups)) != len(groups) or
                    row.get('split') not in ('train', 'validation', 'test') or
                    not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest)):
                raise ValueError(f'invalid open-goal identity: {task}')
            goals = [row['goal']] if 'goal' in row else []
            entry = row.get('entry', {})
            if not isinstance(entry, dict):
                raise ValueError(f'invalid catalog entry: {task}')
            adapter_defaults(entry)
            for key in ENTRY_EXECUTION_FIELDS & entry.keys():
                if not isinstance(entry[key], str) or not entry[key]:
                    raise ValueError(f'invalid entry execution metadata {key}: {task}')
            for key in ('arguments', 'argumentJson'):
                if key in entry:
                    if not isinstance(entry[key], dict):
                        raise ValueError(f'invalid entry {key}: {task}')
                    if 'goal' in entry[key]:
                        value = entry[key]['goal']
                        goals.append(json.loads(value) if key == 'argumentJson' else value)
            if 'parameterNames' in entry or 'inputValues' in entry:
                names, values = entry.get('parameterNames'), entry.get('inputValues')
                if (not isinstance(names, list) or not isinstance(values, list) or len(names) != len(values) or
                        any(not isinstance(n, str) for n in names) or len(set(names)) != len(names)):
                    raise ValueError(f'invalid entry input bindings: {task}')
                goals.extend(v for n, v in zip(names, values) if n == 'goal')
            if any(not isinstance(g, str) or not g.strip() or open_goal_fingerprint(g)[1] != digest for g in goals):
                raise ValueError(f'actual goal differs from goalSha256: {task}')
        rows.append(row)
    if not rows:
        raise ValueError('empty live catalog')
    return rows


def identity(row, *, adapter_context=None):
    """Preserve every non-allowlisted field; redundant goal encodings are checked before projection."""
    result = deepcopy(row)
    for key in EXECUTION_FIELDS:
        result.pop(key, None)
    if result.get('interaction') == 'open-user-goal':
        result['sourceGroups'] = sorted(result['sourceGroups'])
        result.pop('goal', None)  # actual value has been verified against the retained digest
        entry = result.get('entry')
        if entry is not None:
            defaults = adapter_defaults(entry)
            if not defaults and adapter_context:
                # Older catalogs may already encode the same empty argument,
                # before the producer learned to distinguish adapter provenance.
                nulls = {name: value for name, value in adapter_context['values'].items() if value is None}
                present = any(name in entry.get('arguments', {}) or name in entry.get('argumentJson', {}) or
                              name in entry.get('parameterNames', []) for name in nulls)
                if present:
                    candidate = {**entry, 'adapterDefaults': {**adapter_context, 'values': nulls}}
                    defaults = adapter_defaults(candidate)
            if os.environ.get('NATLANG_CATALOG_ADAPTER_DEFAULTS', '1') == '0':
                defaults = {}
            # Null supplementary adapter state carries no authored task input.
            # Non-null research/evidence stays in all original bindings.
            entry.pop('adapterDefaults', None)
            for name, value in defaults.items():
                if value is not None:
                    continue
                entry['arguments'].pop(name)
                entry['argumentJson'].pop(name)
                index = entry['parameterNames'].index(name)
                del entry['parameterNames'][index], entry['inputValues'][index]
            for key in ENTRY_EXECUTION_FIELDS:
                entry.pop(key, None)
            for key in ('arguments', 'argumentJson'):
                if key in entry:
                    entry[key].pop('goal', None)
                    if not entry[key]:
                        del entry[key]
            if 'parameterNames' in entry:
                pairs = [(n, v) for n, v in zip(entry['parameterNames'], entry['inputValues']) if n != 'goal']
                if pairs:
                    entry['parameterNames'], entry['inputValues'] = map(list, zip(*pairs))
                else:
                    del entry['parameterNames'], entry['inputValues']
            if not entry:
                del result['entry']
    return result


def catalog_identity(path):
    return {row['taskId']: identity(row) for row in catalog_rows(path)}


def compatible_catalogs(snapshot, current):
    """Reject changed task/held identity; return a source-hashed, auditable compatibility receipt."""
    snapshot_bytes, current_bytes = Path(snapshot).read_bytes(), Path(current).read_bytes()
    before_rows = catalog_rows(snapshot, lines=snapshot_bytes.decode('utf-8').splitlines())
    after_rows = catalog_rows(current, lines=current_bytes.decode('utf-8').splitlines())
    declarations = {row['taskId']: row.get('entry', {}).get('adapterDefaults')
                    for row in [*before_rows, *after_rows] if row.get('entry', {}).get('adapterDefaults')}
    before = {row['taskId']: identity(row, adapter_context=declarations.get(row['taskId'])) for row in before_rows}
    after = {row['taskId']: identity(row, adapter_context=declarations.get(row['taskId'])) for row in after_rows}
    if canonical_json_bytes(before) != canonical_json_bytes(after):
        changed = sorted(task for task in before.keys() | after.keys()
                         if task not in before or task not in after or
                         canonical_json_bytes(before[task]) != canonical_json_bytes(after[task]))
        raise ValueError(f'live catalog task identity changed: {changed[:8]}')
    changes = []
    old = {row['taskId']: row for row in before_rows}
    for row in after_rows:
        previous = old[row['taskId']]
        fields = [key for key in sorted(EXECUTION_FIELDS) if (key in previous) != (key in row) or previous.get(key) != row.get(key)]
        for key in sorted(ENTRY_EXECUTION_FIELDS | {'adapterDefaults'}):
            a, b = previous.get('entry', {}), row.get('entry', {})
            if (key in a) != (key in b) or a.get(key) != b.get(key):
                fields.append(f'entry.{key}')
        if fields:
            changes.append({'taskId': row['taskId'], 'fields': fields})
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return {'schema': 'natlang.live-catalog-identity/1', 'snapshot': str(Path(snapshot).resolve()),
            'snapshot_sha256': hashlib.sha256(snapshot_bytes).hexdigest(), 'current': str(Path(current).resolve()),
            'current_sha256': hashlib.sha256(current_bytes).hexdigest(), 'identity_sha256': hashlib.sha256(canonical_json_bytes(before)).hexdigest(),
            'validator_sha256': digest(__file__), 'tasks': len(before), 'execution_metadata_changes': changes}
