"""Narrow source-bound syntax repair for intact search-query argument objects.

This module never repairs arbitrary JSON. It only adds the JSON quoting needed
to encode the exact bounded query slice in a declared browser.search call.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from copy import deepcopy
from pathlib import Path

from natlang_neuralese.common.hashing import canonical_json_sha256_hex, sha256_file_hex
from natlang_neuralese.common.jsonio import open_jsonl_text, write_json_durable, write_jsonl_text

REPAIR_RULE = 'browser-search-unquoted-query-v1'
REPAIR_CLASS = 'source-syntax-repair-best-effort'
_QUERY_HEAD = re.compile(r'\A\s*\{\s*"query"\s*:\s*')
_TOPN_DELIMITER = re.compile(r',\s*"topn"\s*:')


def _supported_schema(schema):
    if not isinstance(schema, dict) or schema.get('type') != 'object':
        return False
    properties = schema.get('properties')
    required = schema.get('required')
    if (not isinstance(properties, dict) or not isinstance(required, list)
            or 'query' not in required or set(required) != {'query'}):
        return False
    if set(properties) - {'query', 'topn', 'source'}:
        return False
    topn = properties.get('topn')
    source = properties.get('source')
    return (properties.get('query') == {'type': 'string'}
            and isinstance(topn, dict) and topn.get('type') == 'number'
            and isinstance(source, dict) and source.get('type') == 'string')


def repair_unquoted_query_argument(arguments, *, schema):
    """Return a syntax-only repair, or ``None`` when the source is ambiguous.

    The query is exactly the non-structural substring between the declared
    query field and its unique topn delimiter. No whitespace or text inside
    that substring is normalized. All remaining fields must form a complete
    JSON object that satisfies the pinned browser.search schema.
    """
    if not isinstance(arguments, str) or not _supported_schema(schema):
        return None
    head = _QUERY_HEAD.match(arguments)
    if head is None:
        return None
    # A query containing the same delimiter is ambiguous: no recovery guesses
    # which occurrence was the object boundary.
    delimiters = list(_TOPN_DELIMITER.finditer(arguments, head.end()))
    if len(delimiters) != 1:
        return None
    delimiter = delimiters[0]
    query = arguments[head.end():delimiter.start()]
    if not query.strip() or any(ord(char) < 0x20 for char in query):
        return None
    # Surrounding JSON whitespace belongs to the object syntax. Preserve all
    # characters within the literal query, while excluding only edge JSON
    # whitespace that the original parser would have treated as syntax.
    left = len(query) - len(query.lstrip(' \t\r\n'))
    right = len(query.rstrip(' \t\r\n'))
    literal = query[left:right]
    unescaped_quotes = [index for index, char in enumerate(literal)
                        if char == '"' and (index == 0 or literal[index - 1] != '\\')]
    # A single trailing quote may be part of the literal query as captured;
    # preserve it verbatim. Any interior unescaped quote makes the boundary
    # ambiguous and remains pending.
    if not literal or any(index != len(literal) - 1 for index in unescaped_quotes):
        return None
    replacement = json.dumps(literal, ensure_ascii=False, separators=(',', ':'))
    repaired = arguments[:head.end() + left] + replacement + arguments[head.end() + right:]
    try:
        decoded = json.loads(repaired)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(decoded, dict) or set(decoded) - set(schema['properties']):
        return None
    if set(schema['required']) - set(decoded) or decoded.get('query') != literal:
        return None
    if not isinstance(decoded.get('query'), str):
        return None
    if 'topn' in decoded and (isinstance(decoded['topn'], bool)
            or not isinstance(decoded['topn'], (int, float))
            or not math.isfinite(decoded['topn'])):
        return None
    if 'source' in decoded and not isinstance(decoded['source'], str):
        return None
    # Replacing just this slice must be the only semantic difference.
    try:
        original_tail = json.loads('{"query":"__query_slice__"' + arguments[delimiter.start():])
    except (json.JSONDecodeError, TypeError):
        return None
    expected = dict(original_tail)
    expected['query'] = literal
    if decoded != expected:
        return None
    return repaired


def argument_repair_record(*, source, source_row, session_id, message_index,
                           call_index, tool_name, schema, original, repaired):
    """Bind both exact argument strings and their declared source occurrence."""
    if repair_unquoted_query_argument(original, schema=schema) != repaired:
        raise ValueError('argument repair is not the exact approved syntax transform')
    schema_text = json.dumps(schema, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return {
        'repair_class': REPAIR_CLASS,
        'repair_rule': REPAIR_RULE,
        'source_file': source,
        'source_row': source_row,
        'session_id': session_id,
        'message_index': message_index,
        'call_index': call_index,
        'tool_name': tool_name,
        'tool_schema_sha256': hashlib.sha256(schema_text.encode('utf-8')).hexdigest(),
        'original_arguments': original,
        'original_arguments_sha256': hashlib.sha256(original.encode('utf-8')).hexdigest(),
        'repaired_arguments': repaired,
        'repaired_arguments_sha256': hashlib.sha256(repaired.encode('utf-8')).hexdigest(),
        'training_admission': False,
        'tool_execution_verified': False,
    }


def apply_repair_records(value, records, *, repair_view_sha256):
    """Apply exact bound call replacements to a copied source session."""
    original_sha = canonical_json_sha256_hex(value)
    result = deepcopy(value)
    seen = set()
    tools = {item.get('function', {}).get('name'): item.get('function', {})
             for item in value['trajectory'].get('tools', [])}
    for record in records:
        if (record.get('repair_class') != REPAIR_CLASS or record.get('repair_rule') != REPAIR_RULE
                or record.get('session_id') != value['trajectory']['id']
                or record.get('source_row') != value['source_row']
                or record.get('source_file') != value['source_file']):
            raise ValueError('repair record does not bind this exact source occurrence')
        if record.get('original_normalized_session_sha256') != original_sha:
            raise ValueError('repair record does not bind the exact original normalized session')
        message_index, call_index = record.get('message_index'), record.get('call_index')
        if type(message_index) is not int or type(call_index) is not int:
            raise ValueError('repair record has an invalid tool call location')
        key = (message_index, call_index)
        if key in seen:
            raise ValueError('repair records repeat a tool call location')
        seen.add(key)
        try:
            call = result['trajectory']['messages'][message_index]['tool_calls'][call_index]
            function = call.get('function', call)
            original = value['trajectory']['messages'][message_index]['tool_calls'][call_index]
            original_function = original.get('function', original)
        except (IndexError, KeyError, TypeError) as error:
            raise ValueError('repair record points outside the original tool call list') from error
        name = function.get('name')
        schema = tools.get(name, {}).get('parameters')
        if name != record.get('tool_name') or not _supported_schema(schema):
            raise ValueError('repair record tool name or declared tool schema changed')
        schema_text = json.dumps(schema, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
        if hashlib.sha256(schema_text.encode('utf-8')).hexdigest() != record.get('tool_schema_sha256'):
            raise ValueError('repair record tool schema binding changed')
        original_arguments = original_function.get('arguments')
        if (not isinstance(original_arguments, str)
                or original_arguments != record.get('original_arguments')
                or hashlib.sha256(original_arguments.encode('utf-8')).hexdigest() != record.get('original_arguments_sha256')):
            raise ValueError('repair record original argument bytes changed')
        repaired = record.get('repaired_arguments')
        if (not isinstance(repaired, str)
                or hashlib.sha256(repaired.encode('utf-8')).hexdigest() != record.get('repaired_arguments_sha256')
                or repair_unquoted_query_argument(original_arguments, schema=schema) != repaired):
            raise ValueError('repair record is not the deterministic approved syntax repair')
        function['arguments'] = repaired
    return result


def verify_derived_repair_export(export_path, export, source_path):
    """Prove a derived partial export is an exact subset/repair of its parent.

    The parent source remains the authority used by the global projection. This
    verifier binds each derived row back to that parent and permits only the
    source-bound argument replacements in the immutable repair manifest.
    """
    contract = export.get('contract', {})
    binding = contract.get('source_syntax_repair')
    if binding is None:
        return None
    if binding.get('schema') != 'natlang.acquired-session-syntax-repair-export/1':
        raise ValueError('derived source has an unsupported syntax-repair contract')
    parent_export = Path(binding['parent_export']['path']).resolve(strict=True)
    parent_source = Path(binding['parent_source']['path']).resolve(strict=True)
    manifest_path = Path(binding['repair_manifest']['path']).resolve(strict=True)
    if sha256_file_hex(parent_export) != binding['parent_export']['sha256']:
        raise ValueError('syntax-repair parent export bytes changed')
    if sha256_file_hex(parent_source) != binding['parent_source']['sha256']:
        raise ValueError('syntax-repair parent normalized source bytes changed')
    if (Path(source_path).resolve(strict=True)
            != (Path(export_path).resolve(strict=True).parent / 'full-trajectories.jsonl.zst').resolve(strict=True)):
        raise ValueError('syntax-repair derived source is outside its immutable export directory')
    if sha256_file_hex(manifest_path) != binding['repair_manifest']['sha256']:
        raise ValueError('syntax-repair manifest bytes changed')
    parent = json.loads(parent_export.read_text())
    if parent.get('schema') != 'natlang.normalized-agent-source-file/1':
        raise ValueError('syntax-repair parent is not a normalized source export')
    if parent.get('outputs', {}).get(parent_source.name) != binding['parent_source']['sha256']:
        raise ValueError('syntax-repair parent source differs from its export receipt')
    if export.get('contract', {}).get('source') != parent.get('contract', {}).get('source'):
        raise ValueError('syntax-repair source parquet identity changed')
    if export.get('contract', {}).get('revision') != parent.get('contract', {}).get('revision'):
        raise ValueError('syntax-repair source revision changed')
    manifest = json.loads(manifest_path.read_text())
    if export.get('outputs', {}).get(Path(source_path).name) != sha256_file_hex(source_path):
        raise ValueError('syntax-repair derived source bytes differ from its export receipt')
    if (not isinstance(contract.get('source_rows'), int)
            or isinstance(contract.get('source_rows'), bool)
            or contract['source_rows'] <= 0):
        raise ValueError('syntax-repair export has invalid selected source row accounting')
    for key, expected_schema in (
            ('candidate_inventory', 'natlang.source16-tool-repair-candidate-extract/1'),
            ('pending_audit', 'natlang.source-tool-pending-audit/1')):
        item = manifest.get(key)
        if not isinstance(item, dict):
            raise ValueError('syntax-repair manifest lost its exact ' + key)
        artifact = Path(item.get('path', '')).resolve(strict=True)
        if sha256_file_hex(artifact) != item.get('sha256'):
            raise ValueError('syntax-repair ' + key + ' bytes changed')
        value = json.loads(artifact.read_text())
        if value.get('schema') != expected_schema or value.get('source') != {
                'path': str(parent_source), 'sha256': binding['parent_source']['sha256']}:
            raise ValueError('syntax-repair ' + key + ' no longer binds the exact parent source')
    expected_manifest_binding = {'schema': 'natlang.acquired-session-syntax-repair-manifest/1',
        'parent_export': {'path': str(parent_export), 'sha256': binding['parent_export']['sha256']},
        'parent_source': {'path': str(parent_source), 'sha256': binding['parent_source']['sha256']}}
    if any(manifest.get(key) != value for key, value in expected_manifest_binding.items()):
        raise ValueError('syntax-repair manifest parent binding differs')
    parent_rows = {}
    with open_jsonl_text(parent_source) as stream:
        for row in stream:
            if not row.strip():
                continue
            value = json.loads(row)
            key = (value['trajectory']['id'], value['source_row'])
            if key in parent_rows:
                raise ValueError('syntax-repair parent repeats a source occurrence')
            parent_rows[key] = value
    manifest_rows = manifest.get('rows')
    if not isinstance(manifest_rows, list) or not manifest_rows:
        raise ValueError('syntax-repair manifest has no selected source rows')
    seen = set()
    with open_jsonl_text(source_path) as stream:
        derived_rows = [line for line in stream if line.strip()]
    if len(derived_rows) != len(manifest_rows) or len(derived_rows) != export.get('complete_sessions'):
        raise ValueError('syntax-repair derived row count differs from its manifest/export')
    for encoded, entry in zip(derived_rows, manifest_rows):
        derived = json.loads(encoded)
        key = (entry.get('session_id'), entry.get('source_row'))
        if key in seen or key not in parent_rows:
            raise ValueError('syntax-repair manifest repeats or loses a parent source occurrence')
        seen.add(key)
        original = parent_rows[key]
        if canonical_json_sha256_hex(original) != entry.get('original_normalized_session_sha256'):
            raise ValueError('syntax-repair original normalized session hash differs')
        records = entry.get('repairs')
        repair_view_sha = canonical_json_sha256_hex(records)
        expected = apply_repair_records(original, records, repair_view_sha256=repair_view_sha)
        expected['source_syntax_repair'] = {
            'schema': 'natlang.acquired-session-syntax-repair-view/1',
            'repair_class': REPAIR_CLASS,
            'repair_rule': REPAIR_RULE,
            'repair_view_sha256': repair_view_sha,
            'parent_export_sha256': binding['parent_export']['sha256'],
            'parent_source_sha256': binding['parent_source']['sha256'],
            'original_normalized_session_sha256': entry['original_normalized_session_sha256'],
            'repairs': records,
        }
        if (canonical_json_sha256_hex(derived) != entry.get('derived_normalized_session_sha256')
                or derived != expected):
            raise ValueError('derived source session is not the exact approved syntax repair')
    candidate_rows = json.loads(Path(manifest['candidate_inventory']['path']).read_text()).get('rows', [])
    if len({(row.get('id'), row.get('source_row')) for row in candidate_rows}) != len(candidate_rows):
        raise ValueError('syntax-repair candidate inventory repeats a source occurrence')
    if not set(seen).issubset({(row.get('id'), row.get('source_row')) for row in candidate_rows}):
        raise ValueError('syntax-repair derived rows escape the exact candidate inventory')
    return {'manifest_sha256': binding['repair_manifest']['sha256'],
        'parent_export_sha256': binding['parent_export']['sha256'],
        'parent_source_sha256': binding['parent_source']['sha256'],
        'selected_rows': len(manifest_rows), 'repair_calls': sum(len(row['repairs']) for row in manifest_rows)}


def build_repair_view(*, parent_export_path, candidate_audit_path, pending_audit_path, output_dir):
    """Create a new immutable normalized export containing fully repaired rows.

    Candidate and audit files select the exact source occurrences; this
    function verifies those rows against the parent before applying the narrow
    syntax transform. A session with any unrepaired invalid tool argument is
    left out of this child view.
    """
    parent_export_path = Path(parent_export_path).resolve(strict=True)
    parent_export = json.loads(parent_export_path.read_text())
    if parent_export.get('schema') != 'natlang.normalized-agent-source-file/1':
        raise ValueError('repair view requires a normalized source-file parent')
    parent_source = (parent_export_path.parent / 'full-trajectories.jsonl.zst').resolve(strict=True)
    parent_export_sha = sha256_file_hex(parent_export_path)
    parent_source_sha = sha256_file_hex(parent_source)
    if parent_export.get('outputs', {}).get(parent_source.name) != parent_source_sha:
        raise ValueError('parent source bytes differ from its normalized export receipt')
    candidates = json.loads(Path(candidate_audit_path).read_text())
    if candidates.get('schema') != 'natlang.source16-tool-repair-candidate-extract/1':
        raise ValueError('unsupported exact repair-candidate inventory')
    if candidates.get('source') != {'path': str(parent_source), 'sha256': parent_source_sha}:
        raise ValueError('candidate inventory does not bind the exact parent source')
    audit_path = Path(pending_audit_path).resolve(strict=True)
    audit = json.loads(audit_path.read_text())
    if audit.get('source') != candidates.get('source'):
        raise ValueError('pending audit and candidate inventory bind different source bytes')
    audit_by_id = {row['id']: row for row in audit.get('rows', [])}
    candidate_by_id = {row['id']: row for row in candidates.get('rows', [])}
    if set(audit_by_id) != set(candidate_by_id):
        raise ValueError('candidate inventory does not cover exactly the audited pending rows')
    parent_rows = {}
    with open_jsonl_text(parent_source) as stream:
        for line in stream:
            if not line.strip():
                continue
            value = json.loads(line)
            key = value['trajectory']['id']
            if key in parent_rows:
                raise ValueError('parent contains duplicate session IDs')
            parent_rows[key] = value
    selected = []
    unresolved = []
    for session_id, candidate in candidate_by_id.items():
        value = parent_rows.get(session_id)
        audit_row = audit_by_id[session_id]
        if (value is None or value.get('source_row') != candidate.get('source_row')
                or value.get('source_file') != candidate.get('source_file')
                or canonical_json_sha256_hex(value) != candidate.get('normalized_source_sha256')
                or audit_row.get('id') != session_id):
            raise ValueError('candidate is not the exact audited normalized source occurrence')
        invalid = audit_row.get('calls', [])
        candidate_calls = candidate.get('calls', [])
        if len(invalid) != len(candidate_calls):
            raise ValueError('candidate does not cover every malformed call in the source row')
        failures = audit_row.get('failures', {})
        if (not isinstance(failures, dict) or set(failures) != {'lfm2', 'mellum'}
                or any(not isinstance(error, str) or not error.startswith('tool-arguments:')
                       for error in failures.values())):
            raise ValueError('source-syntax repair is limited to rows independently pending on both templates')
        call_keys = [(call.get('message_index'), call.get('call_index')) for call in candidate_calls]
        if len(set(call_keys)) != len(call_keys) or set(call_keys) != {
                (call.get('message_index'), call.get('call_index')) for call in invalid}:
            raise ValueError('candidate call inventory is duplicated or incomplete')
        call_map = dict(zip(call_keys, candidate_calls))
        records = []
        failure = None
        for bad in invalid:
            key = (bad.get('message_index'), bad.get('call_index'))
            item = call_map.get(key)
            if item is None:
                failure = 'candidate-call-missing'
                break
            try:
                message = value['trajectory']['messages'][key[0]]
                call = message['tool_calls'][key[1]]
                function = call.get('function', call)
                tool_name = function.get('name')
                schema = item.get('schema')
                original = function.get('arguments')
                if (tool_name != item.get('name') or original != item.get('arguments')
                        or hashlib.sha256(original.encode('utf-8')).hexdigest() != item.get('arguments_utf8_sha256')):
                    failure = 'candidate-argument-binding-mismatch'
                    break
                repaired = repair_unquoted_query_argument(original, schema=schema)
                if repaired is None:
                    failure = 'strict-query-repair-rejected'
                    break
                record = argument_repair_record(source=value['source_file'], source_row=value['source_row'],
                    session_id=session_id, message_index=key[0], call_index=key[1], tool_name=tool_name,
                    schema=schema, original=original, repaired=repaired)
                record['original_normalized_session_sha256'] = candidate['normalized_source_sha256']
                records.append(record)
            except (IndexError, KeyError, TypeError, AttributeError):
                failure = 'candidate-call-location-invalid'
                break
        if failure is not None:
            unresolved.append({'session_id': session_id, 'source_row': candidate['source_row'],
                'reason': failure, 'pending_status': 'source-faithful-review-pending',
                'normalized_source_sha256': candidate['normalized_source_sha256']})
            continue
        repairs_by_position = {(record['message_index'], record['call_index']): record for record in records}
        repaired_calls = apply_repair_records(value, records,
            repair_view_sha256=canonical_json_sha256_hex(records))
        # No call omitted by the source audit may remain invalid after repair.
        from scripts.neuralese_data.build_acquired_full_session_graphs import prevalidate_session_tool_arguments
        try:
            for backbone in ('lfm2', 'mellum'):
                prevalidate_session_tool_arguments(repaired_calls, template_backbone=backbone)
        except Exception as error:
            unresolved.append({'session_id': session_id, 'source_row': candidate['source_row'],
                'reason': 'current-model-template-prevalidation-failed',
                'error_type': type(error).__name__, 'error': str(error),
                'pending_status': 'source-faithful-review-pending',
                'normalized_source_sha256': candidate['normalized_source_sha256']})
            continue
        repair_view_sha = canonical_json_sha256_hex(records)
        derived = apply_repair_records(value, records, repair_view_sha256=repair_view_sha)
        derived['source_syntax_repair'] = {
            'schema': 'natlang.acquired-session-syntax-repair-view/1',
            'repair_class': REPAIR_CLASS, 'repair_rule': REPAIR_RULE,
            'repair_view_sha256': repair_view_sha,
            'parent_export_sha256': parent_export_sha, 'parent_source_sha256': parent_source_sha,
            'original_normalized_session_sha256': candidate['normalized_source_sha256'],
            'repairs': records,
        }
        selected.append({'session_id': session_id, 'source_row': candidate['source_row'],
            'original_normalized_session_sha256': candidate['normalized_source_sha256'],
            'derived_normalized_session_sha256': canonical_json_sha256_hex(derived),
            'repairs': records, 'value': derived})
    if not selected:
        raise ValueError('no complete sessions have a fully verified source-syntax repair')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    repair_rows = [{key: value for key, value in row.items() if key != 'value'} for row in selected]
    manifest = {'schema': 'natlang.acquired-session-syntax-repair-manifest/1',
        'repair_class': REPAIR_CLASS, 'repair_rule': REPAIR_RULE,
        'parent_export': {'path': str(parent_export_path), 'sha256': parent_export_sha},
        'parent_source': {'path': str(parent_source), 'sha256': parent_source_sha},
        'candidate_inventory': {'path': str(Path(candidate_audit_path).resolve(strict=True)),
            'sha256': sha256_file_hex(candidate_audit_path)},
        'pending_audit': {'path': str(audit_path.resolve(strict=True)), 'sha256': sha256_file_hex(audit_path)},
        'rows': repair_rows, 'unresolved': unresolved, 'training_admission': False,
        'tool_execution_verified': False}
    manifest_path = output_dir / 'repair-manifest.json'
    write_json_durable(manifest_path, manifest)
    manifest_sha = sha256_file_hex(manifest_path)
    source_path = output_dir / 'full-trajectories.jsonl.zst'
    with write_jsonl_text(source_path) as stream:
        for row in selected:
            stream.write(json.dumps(row['value'], ensure_ascii=False, separators=(',', ':')) + '\n')
    source_sha = sha256_file_hex(source_path)
    omission_path = output_dir / 'omissions.jsonl'
    with write_jsonl_text(omission_path):
        pass
    omission_sha = sha256_file_hex(omission_path)
    child_contract = deepcopy(parent_export['contract'])
    child_contract['source_syntax_repair'] = {
        'schema': 'natlang.acquired-session-syntax-repair-export/1',
        'parent_export': {'path': str(parent_export_path), 'sha256': parent_export_sha},
        'parent_source': {'path': str(parent_source), 'sha256': parent_source_sha},
        'repair_manifest': {'path': str(manifest_path.resolve()), 'sha256': manifest_sha},
        'selected_sessions': len(selected), 'unresolved_sessions': len(unresolved),
        'repair_calls': sum(len(row['repairs']) for row in repair_rows)}
    child_export = {'schema': 'natlang.normalized-agent-source-file/1',
        'complete_sessions': len(selected), 'positive_candidates': len(selected),
        'contract': child_contract,
        'outputs': {source_path.name: source_sha, omission_path.name: omission_sha},
        'training_admission': False}
    export_path = output_dir / 'export.json'
    write_json_durable(export_path, child_export)
    complete_path = output_dir / 'complete.json'
    complete = {'schema': 'natlang.normalized-agent-source-complete/1', 'ref': 'D32-repair-v1',
        'source_revision': child_contract['revision'], 'training_admission': False,
        'files': [{'file': parent_export['contract']['source']['path'],
            'receipt': str(export_path.resolve()), 'ref': 'D32-repair-v1',
            'status': 'exported', 'exporter_sha256': child_contract['exporter_sha256']} ]}
    write_json_durable(complete_path, complete)
    return {'output_dir': str(output_dir.resolve()), 'selected_sessions': len(selected),
        'repair_calls': sum(len(row['repairs']) for row in repair_rows),
        'unresolved_sessions': len(unresolved), 'repair_manifest_sha256': manifest_sha,
        'export_sha256': sha256_file_hex(export_path), 'source_sha256': source_sha,
        'training_admission': False, 'tool_execution_verified': False}
