"""Discover exact source-correct typed children; this grants no admission.

The existing decision-review/materializer remains the target authority. This
module only constructs independently checkable evidence for that review.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter

from natlang_neuralese.common.hashing import canonical_json_sha256_hex


_HASH = re.compile(r"[0-9a-f]{64}\Z")
_UNION = re.compile(r'''\s*(["'])(negative|positive)\1\s*\|\s*(["'])(negative|positive)\3\s*\Z''')
_OPENING = re.compile(
    r'([A-Za-z_$][\w$]*): FileHandle = folder\.file\(("(?:[^"\\]|\\.)*")\)\n'
    r'  // File contents:\n("(?:[^"\\]|\\.)*")\n'
    r'Declared ([A-Za-z_$][\w$]*(?:, [A-Za-z_$][\w$]*)*) for the rest of this call\.\Z', re.S)


def _sentiment_contract(text):
    match = _UNION.fullmatch(text) if isinstance(text, str) else None
    return bool(match and {match[2], match[4]} == {'negative', 'positive'})


def _sha(value):
    return isinstance(value, str) and _HASH.fullmatch(value) is not None


def _terminal_value(turn):
    """Require the captured provider action itself, rather than host text alone."""
    assistant = turn.get('assistant') or {}
    calls = assistant.get('calls') or []
    raw_calls = (turn.get('model_response') or {}).get('raw_calls')
    if not isinstance(raw_calls, list):
        return None, 'missing-raw-provider-calls'
    if calls:
        if len(calls) != 1 or len(raw_calls) != 1:
            return None, 'not-one-terminal-provider-call'
        call = calls[0]
        function = raw_calls[0].get('function') or {}
        try:
            arguments = json.loads(function.get('arguments', ''))
        except (TypeError, ValueError):
            return None, 'invalid-raw-provider-arguments'
        if (call.get('source_tool') != 'return_result'
                or function.get('name') != 'return_result'
                or call.get('tool') != 'return_result'
                or call.get('arguments') != arguments
                or not isinstance(arguments, dict)
                or arguments.get('status', 'success') != 'success'
                or set(arguments) - {'status', 'value'}):
            return None, 'provider-terminal-call-does-not-match'
        value = arguments.get('value')
    else:
        if raw_calls:
            return None, 'unmatched-raw-provider-call'
        content = assistant.get('content')
        if not isinstance(content, str):
            return None, 'missing-direct-provider-answer'
        value = content.strip()
        if value not in ('negative', 'positive'):
            try:
                value = json.loads(value)
            except ValueError:
                return None, 'not-an-exact-direct-label'
    if value not in ('negative', 'positive'):
        return None, 'not-a-declared-label'
    return value, None


def discover_exact_sentiment_children(row, *, primary_records, parent_hold_reason):
    """Return review candidates and explicit unresolved invocations.

    ``primary_records`` is independently authenticated source evidence keyed by
    the full source record ID. Its declared label must agree with both the task
    reference and a nonempty exact primary phrase/score inventory. This function
    does not infer a label from a score or a model answer. Source/archive/current
    protection authentication remains required by the caller and final verifier.
    """
    if parent_hold_reason not in ('quality_pending_partial_files', 'quality_pending_partial_agreement'):
        raise ValueError('child discovery requires an explicitly classified partial-parent hold')
    program = (row.get('task') or {}).get('program_ir') or {}
    if (program.get('split') != 'train' or program.get('dataset') != 'sst2'
            or not program.get('source_groups') or row.get('outcome', {}).get('status') != 'done'):
        raise ValueError('child review requires a complete source-owned TRAIN SST program')
    contexts = (program.get('generation') or {}).get('source_context')
    files = (program.get('semantics') or {}).get('folder_files')
    if not isinstance(contexts, list) or not contexts or not isinstance(files, dict):
        raise ValueError('child review needs exact full-sentence file inputs')
    records = {}
    for context in contexts:
        identifier = context.get('record_id')
        independent = primary_records.get(identifier)
        if (not _sha(identifier)
                or context.get('schema') != 'natlang.sst-full-sentence-context/1'
                or context.get('record_label') not in ('negative', 'positive')
                or not isinstance(independent, dict)
                or independent.get('declared_label') != context['record_label']
                or independent.get('source_sentence') != [member['original_sentence'] for member in context['members']]
                or not independent.get('primary_phrase_scores')
                or any(not isinstance(score, dict) or not isinstance(score.get('phrase_id'), str)
                       or not score['phrase_id'].isdigit() or not isinstance(score.get('phrase'), str)
                       or type(score.get('score')) not in (int, float)
                       or not math.isfinite(score['score']) or not 0 <= score['score'] <= 1
                       for score in independent.get('primary_phrase_scores', []))
                or any(member.get('original_split') != 'train' for member in context['members'])):
            raise ValueError('child source label/full sentence lacks exact primary evidence')
        prefix = '---\nid: ' + identifier[:16] + '\n---\n'
        matching = [(path, body) for path, body in files.items()
                    if isinstance(body, str) and body.startswith(prefix)]
        if len(matching) != 1:
            raise ValueError('child source record lacks one exact input file')
        path, body = matching[0]
        key = (path.rsplit('/', 1)[-1], body)
        if key in records:
            raise ValueError('child file identity is ambiguous')
        records[key] = {'record_id': identifier, 'source_path': path, 'body': body,
                        'label': context['record_label'], 'source_context': context,
                        'primary': independent}
    references = ((program.get('curriculum') or {}).get('reference') or {}).get('children') or []
    for record in records.values():
        matches = [ref for ref in references if ref.get('match') == record['record_id'][:16]]
        if len(matches) != 1:
            raise ValueError('child source lacks one unchanged scripted reference')
        returns = [call[1].get('value') for call in matches[0].get('calls', [])
                   if isinstance(call, list) and len(call) == 2 and call[0] == 'return_result']
        if returns != [record['label']]:
            raise ValueError('child source reference conflicts with its original label')
    trajectory = row.get('trajectory') or []
    ledgers = row.get('outcome', {}).get('invocation_ledger') or []
    by_invocation = {}
    for index, turn in enumerate(trajectory):
        by_invocation.setdefault(turn.get('invocation_id'), []).append((index, turn))
    candidates, unresolved = [], []
    for ledger in ledgers:
        invocation = ledger.get('invocation_id')
        if not ledger.get('parent_invocation_id'):
            continue
        reason = None
        site = ledger.get('inline_instruction_site') or {}
        parameters = site.get('parameters') or []
        returns = site.get('returns') or {}
        turns = by_invocation.get(invocation) or []
        host = ledger.get('host_result') or {}
        if (len(parameters) != 1 or (parameters[0].get('type') or {}).get('text') != 'FileHandle'
                or not _sentiment_contract(returns.get('text')) or not turns
                or invocation == ledger['parent_invocation_id']
                or ledger['parent_invocation_id'] not in by_invocation
                or (site.get('origin') or {}).get('parentInvocationId') != ledger['parent_invocation_id']
                or not isinstance(site.get('realized_instruction'), str) or not site['realized_instruction'].strip()):
            reason = 'not-an-exact-file-sentiment-child'
        elif (ledger.get('completion_status') != 'done' or host.get('complete') is not True
                or host.get('call_id') != invocation
                or host.get('parent_call_id') != ledger['parent_invocation_id']
                or host.get('result_type') != returns.get('text')):
            reason = 'child-has-no-complete-typed-host-result'
        opening = []
        if reason is None:
            messages = turns[0][1].get('context') or []
            declared = {parameters[0].get('name'), *[capture.get('name') for capture in site.get('captures', [])]}
            bad_opening = False
            for position, message in enumerate(messages):
                content = message.get('content')
                match = _OPENING.fullmatch(content) if message.get('role') == 'tool' and isinstance(content, str) else None
                if match and match[1] == parameters[0].get('name'):
                    previous = messages[position - 1] if position else {}
                    setups = [call for call in previous.get('tool_calls', [])
                              if call.get('id') == message.get('tool_call_id')
                              and (call.get('function') or {}).get('name') == 'eval']
                    names = match[4].split(', ')
                    if (previous.get('role') != 'assistant' or previous.get('natlang_host_generated') is not True
                            or len(setups) != 1 or len(names) != len(set(names)) or set(names) != declared):
                        bad_opening = True
                    opening.append((json.loads(match[2]), json.loads(match[3])))
            if bad_opening or len(opening) != 1 or opening[0] not in records:
                reason = 'child-opening-does-not-bind-exact-source-file'
        if reason is not None:
            unresolved.append({'invocation_id': invocation, 'reason': reason})
            continue
        record = records[opening[0]]
        index, terminal = turns[-1]
        value, reason = _terminal_value(terminal)
        if reason is None and (value != host.get('value') or value != record['label']):
            reason = 'provider-host-source-label-disagreement'
        calls = (terminal.get('assistant') or {}).get('calls') or []
        if reason is None and calls:
            raw_call = terminal['model_response']['raw_calls'][0]
            actions = [action for action in row['outcome'].get('action_ledger', [])
                       if action.get('call_id') == invocation
                       and action.get('seq') == host.get('terminal_action_seq')]
            if (type(host.get('terminal_action_seq')) is not int
                    or not isinstance(raw_call.get('id'), str) or not raw_call['id']
                    or len(actions) != 1 or actions[0].get('kind') != 'action'
                    or actions[0].get('call_id') != invocation
                    or actions[0].get('name') != 'return_result'
                    or actions[0].get('tool_call_id') != raw_call['id']
                    or actions[0].get('outcome') not in ('ok', 'completed')
                    or actions[0].get('arguments') != calls[0]['arguments']):
                reason = 'terminal-provider-action-lacks-exact-host-execution'
        elif reason is None:
            # Discovery must expose this gap rather than guess the association
            # of an implicit final-text action with its provider response.
            reason = 'direct-label-needs-exact-terminal-host-action-link'
        if reason is None and not all(_sha(terminal.get(key)) for key in ('request_sha256', 'raw_response_sha256')):
            reason = 'missing-provider-request-response-capture'
        if reason is not None:
            unresolved.append({'invocation_id': invocation, 'decision_index': index,
                               'record_id': record['record_id'], 'reason': reason})
            continue
        candidates.append({'trajectory_id': row['id'], 'decision_index': index,
            'invocation_id': invocation, 'parent_invocation_id': ledger['parent_invocation_id'],
            'request_sha256': terminal['request_sha256'],
            'raw_response_sha256': terminal['raw_response_sha256'],
            'captured_terminal_sha256': canonical_json_sha256_hex(terminal),
            'record_id': record['record_id'], 'source_path': record['source_path'],
            'input_sha256': canonical_json_sha256_hex(record['body']),
            'expected_value': record['label'], 'provider_value': value, 'host_value': host['value'],
            'host_result_sha256': canonical_json_sha256_hex(host),
            'inline_site_sha256': canonical_json_sha256_hex(site),
            'instruction': site.get('realized_instruction'),
            'instruction_captures': site.get('captures', []),
            'source_context_sha256': canonical_json_sha256_hex(record['source_context']),
            'primary_record_sha256': canonical_json_sha256_hex(record['primary']),
            'parent_hold_reason': parent_hold_reason,
            'parent_task_passed': False, 'training_admission': False, 'trace_admission': False})
    return {'candidates': candidates, 'unresolved': unresolved,
            'unresolved_counts': dict(Counter(entry['reason'] for entry in unresolved)),
            'source_row_canonical_sha256': canonical_json_sha256_hex(row),
            'training_admission': False, 'trace_admission': False}
