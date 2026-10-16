"""Lossless external format boundaries shared by the newly granted agent sources."""
from __future__ import annotations

import json

from .common import Reject
from .trajectory import tool_call


def _json(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def complete_messages(raw):
    """Normalize actual OpenAI/Anthropic blocks without dropping source context.

    The caller retains the original message inventory in immutable provenance.
    Unknown blocks remain explicit JSON text; unknown roles reject the context.
    Block order is preserved, including user text around tool results.
    """
    if not isinstance(raw, list) or not raw:
        raise Reject('missing complete original message inventory')
    out = []
    calls = {}
    for original in raw:
        if not isinstance(original, dict):
            raise Reject('original message is not an object')
        role = original.get('role')
        role = {'developer': 'system', 'function': 'tool'}.get(role, role)
        if role not in ('system', 'user', 'assistant', 'tool'):
            raise Reject('unsupported original message role: '+str(role))
        content = original.get('content')
        if not isinstance(content, (str, list, type(None))):
            content = _json(content)
        if not isinstance(content, list):
            message = {key: original[key] for key in ('name', 'tool_call_id') if original.get(key) is not None}
            message.update(role=role, content=content)
            reasoning = original.get('reasoning_content', original.get('reasoning'))
            if reasoning is not None:
                message['reasoning'] = reasoning if isinstance(reasoning, str) else _json(reasoning)
            declared = original.get('tool_calls')
            legacy = original.get('function_call')
            if declared and legacy:
                raise Reject('conflicting original tool call formats')
            if legacy is not None:
                declared = [{'function': legacy}]
            if isinstance(declared, str):
                declared = json.loads(declared)
            if declared:
                if role != 'assistant' or not isinstance(declared, list):
                    raise Reject('tool dispatch outside original assistant message')
                normalized = []
                for call in declared:
                    fn = call.get('function', call)
                    if not isinstance(fn.get('name'), str) or not fn['name']:
                        raise Reject('original tool call lacks name')
                    identifier = call.get('id')
                    if identifier:
                        if identifier in calls:
                            raise Reject('duplicate original tool call identity')
                        calls[identifier] = fn['name']
                    normalized.append(tool_call(fn['name'], fn.get('arguments', {}), identifier))
                message['tool_calls'] = normalized
            out.append(message)
            continue
        # Native content blocks encode tool dispatches/results in position, not
        # an arbitrary JSON object parsed from prose.
        if original.get('tool_calls') or original.get('function_call'):
            raise Reject('mixed block and outer tool call formats require explicit source mapping')
        first_block = len(out)
        outer_reasoning = original.get('reasoning_content', original.get('reasoning'))
        if outer_reasoning is not None:
            out.append({'role': role, 'content': '', 'reasoning': outer_reasoning if isinstance(outer_reasoning, str) else _json(outer_reasoning)})
        if not content:
            out.append({'role': role, 'content': ''})
        for block in content:
            if not isinstance(block, dict):
                out.append({'role': role, 'content': _json(block)})
                continue
            kind = block.get('type')
            if kind in ('text', 'input_text', 'output_text') and isinstance(block.get('text'), str):
                out.append({'role': role, 'content': block['text']})
            elif kind == 'thinking' and role == 'assistant' and isinstance(block.get('thinking'), str):
                out.append({'role': role, 'content': '', 'reasoning': block['thinking']})
            elif kind == 'tool_use':
                identifier, name = block.get('id'), block.get('name')
                if role != 'assistant' or not isinstance(identifier, str) or not identifier or not isinstance(name, str) or not name:
                    raise Reject('invalid original tool-use block')
                if identifier in calls:
                    raise Reject('duplicate original tool call identity')
                calls[identifier] = name
                out.append({'role': role, 'content': None,
                            'tool_calls': [tool_call(name, block.get('input', {}), identifier)]})
            elif kind == 'tool_result':
                identifier = block.get('tool_use_id')
                if role not in ('user', 'tool') or not isinstance(identifier, str) or not identifier:
                    raise Reject('invalid original tool-result block')
                # Missing earlier dispatch can be an authentic partial/compacted
                # context. Preserve its exact ID, never synthesize the call.
                body = block.get('content')
                out.append({'role': 'tool', 'content': (_json({'content': body, 'is_error': True}) if block.get('is_error') is True
                                        else body if isinstance(body, str) else _json(body)),
                            'tool_call_id': identifier, **({'name': calls[identifier]} if identifier in calls else {})})
            else:
                out.append({'role': role, 'content': _json(block)})
        if role == 'tool':
            for message in out[first_block:]:
                for key in ('tool_call_id', 'name'):
                    if original.get(key) is not None:
                        message.setdefault(key, original[key])
    return out


def _digest(value):
    from natlang_neuralese.common.hashing import canonical_json_sha256_hex
    from .acquired_agents import source_json_value
    return canonical_json_sha256_hex(source_json_value(value))


def _metadata(value):
    from .acquired_agents import source_json_value
    return source_json_value(value)


def preference_binding(receipt, rid, row):
    """Small exact references; full alternatives occur once in offline records."""
    path,index=rid.rsplit(':',1)
    source=next((entry for entry in receipt.get('files',[]) if entry['path']==path),None)
    return {'schema':'natlang.offline-preference-source-binding/1',
        'source_revision':receipt['revision'],'source_file':source,'source_row':int(index),
        'pair_sha256':_digest(row),
        'alternatives':{name:{'field':name,'original_messages_sha256':_digest(row[name]),
                              'original_message_count':len(row[name])} for name in ('chosen','rejected')}}


def _session(ref, receipt, rid, row, raw, *, task=None, suffix='', tools=None):
    from .acquired_agents import _reported_session
    messages = complete_messages(raw)
    identifier = rid + suffix
    bound_row = dict(row)
    if ref == 'D39' and row.get('owner'):
        bound_row.pop('repo', None)  # full owner/repo is added below, never a bare common repository name
    trajectory = _reported_session(ref, receipt, identifier, bound_row, messages,
        task_id=task or rid, session_id=identifier,
        teacher=row.get('model') or row.get('teacher_model'), tools=tools)
    # Already normalized exact block order; generic source cleanup must not parse
    # prose dispatch tags or delete literal control spellings in source files.
    trajectory.messages = messages
    trajectory.lineage['notes']['granted_source'] = {
        'original_messages_sha256': _digest(raw), 'original_message_count': len(raw),
        'original_metadata': _metadata({k: v for k, v in row.items()
            if k not in ('messages', 'chosen', 'rejected', 'text', 'extra')}),
        'whole_context': True, 'executed_locally': False}
    return trajectory


def miro_pair_trajectories(receipt, rid, row):
    from .records import group_key
    from .trajectory import task_hash
    chosen, rejected = row['chosen'], row['rejected']
    if not isinstance(chosen, list) or not isinstance(rejected, list) or not chosen or not rejected:
        raise Reject('DPO source lacks both complete original alternatives')
    # Source question, not row address, joins SFT and DPO variants.
    prompt = next((m.get('content') for m in chosen if m.get('role') == 'user'), None)
    other = next((m.get('content') for m in rejected if m.get('role') == 'user'), None)
    if prompt is None or prompt != other:
        raise Reject('DPO alternatives do not share the exact original request')
    for name, raw in (('chosen', chosen), ('rejected', rejected)):
        child = rid + ':' + name
        t = _session('D24', receipt, rid, row, raw, suffix=':'+name,
                     task='request:'+task_hash(_json(prompt)))
        t.family = 'trajectory_continuation_research'; t.noun = 'research session'
        t.split_groups.append(group_key('miroverse-preference-row', rid))
        t.lineage['notes']['source_preference'] = {
            **preference_binding(receipt, rid, row), 'alternative': name,
            'preference': 'chosen-over-rejected', 'source_row': rid,
            'local_correctness_verified': False}
        if name == 'rejected':
            # Rejection is a preference fact, not proof of task failure.
            t.lineage['notes']['source_quality_flags'] = ['source-rejected-preference-alternative']
        yield child, t


def miroverse_acquired(limit=None):
    """All released SFT subsets and both immutable DPO alternatives."""
    from .acquired_agents import acquired_rows
    from .records import group_key
    from .trajectory import task_hash
    for receipt, rid, row in acquired_rows('D24', limit):
        try:
            path = rid.rsplit(':', 1)[0]
            if path.startswith('jsonl_sft/'):
                # Upstream `split` names the source corpus (e.g. MiroVerse-2Wiki),
                # not a train/test assignment. Preserve it verbatim in metadata.
                t = _session('D24', receipt, rid, row, row['messages'])
                t.family = 'trajectory_continuation_research'; t.noun = 'research session'
                yield rid, t
            elif path.startswith('dpo/'):
                yield from miro_pair_trajectories(receipt, rid, row)
            else:
                raise Reject('alternate representation or source metadata; complete SFT/DPO files are consumers')
        except (Reject, ValueError, TypeError, KeyError) as error:
            yield rid, Reject(str(error))



def midtool_native_extra(raw):
    """Decode exact original messages; authenticate the observed missing-tools defect."""
    if not isinstance(raw,str):return raw,{}
    try:return json.loads(raw),{}
    except json.JSONDecodeError:
        # A released awm-rollout record has a complete valid messages array and
        # an absent tools VALUE. Decode only that authentic complete field;
        # neither repair arbitrary JSON nor invent a schema/tool observation.
        import re
        opening=re.match(r'\s*\{\s*"messages"\s*:\s*',raw)
        if opening is None:raise
        messages,end=json.JSONDecoder().raw_decode(raw,opening.end())
        if not isinstance(messages,list) or re.fullmatch(r'\s*,\s*"tools"\s*:\s*\}\s*',raw[end:]) is None:
            raise ValueError('native-agent extra is not the exact released missing-tools-value shape')
        return {'messages':messages},{'original_tools_field_missing_value':True,
            'schema_recovery':'complete-original-messages-only; no tool schemas fabricated'}

def midtool_acquired(limit=None):
    """Complete native-agent messages plus unabridged web/PDF/code text SFT."""
    from .acquired_agents import acquired_rows
    from .records import group_key
    from .trajectory import task_hash
    for receipt, rid, row in acquired_rows('D39', limit):
        try:
            subset = rid.split('/', 1)[0]
            text = row.get('text')
            if not isinstance(text, str) or not text:
                raise Reject('MidTool source lacks complete original text')
            if subset == 'native-agent-traj':
                extra,recovery = midtool_native_extra(row.get('extra'))
                if not isinstance(extra, dict) or not isinstance(extra.get('messages'), list):
                    raise Reject('native-agent source lacks original structured messages; serialized text is not reparsed')
                t = _session('D39', receipt, rid, row, extra['messages'], tools=extra.get('tools'))
                t.lineage['notes']['granted_source'].update(original_extra_sha256=_digest(row.get('extra')),
                    **recovery,
                    original_serialized_text_sha256=_digest(text),
                    original_source_fields=['extra','text'], duplicate_serialized_view_not_trained=True)
            elif subset in ('web', 'pdf', 'code'):
                # Document completion uses exactly the source text as the target;
                # no teacher answer or synthetic tool trajectory is invented.
                t = _session('D39', receipt, rid, row, [{'role':'assistant','content':text}],
                             task='document:'+task_hash(text))
                t.family = 'source_document_completion'; t.noun = 'original source document'
                t.lineage['notes']['granted_source'].update(original_document_sha256=_digest(text),
                    original_source_fields=['text'], supervision='source-text-completion', correctness_asserted=False)
                if subset == 'code':
                    owner, repo = row.get('owner'), row.get('repo')
                    if not owner or not repo or not row.get('commit_sha') or not row.get('relpath'):
                        raise Reject('code source lacks original repository/commit/file identity')
                    t.split_groups.append(group_key('repo', str(owner)+'/'+str(repo)))
            else:
                raise Reject('unknown MidTool source subset; catalogue conversion remains owed')
            if row.get('url'):
                t.split_groups.append(group_key('source-url', row['url']))
            t.lineage['notes']['granted_source']['subset'] = subset
            yield rid, t
        except (Reject, ValueError, TypeError, KeyError) as error:
            yield rid, Reject(str(error))


def swe_raw_contexts(records, *, source_id=None):
    """Exact native event ancestry, including reset boundaries and forked contexts.

    Returns (representative original row, normalized messages, event evidence).
    Nonconversational events remain byte-addressed evidence, not model targets.
    """
    if any(value.get('type') == 'response_item' for _, value in records if isinstance(value,dict)):
        from .acquired_swe_formats import codex_context
        yield codex_context(records,source_id)
        return
    if records and all(isinstance(value,dict) and value.get('v')==1 for _,value in records):
        from .acquired_swe_formats import compact_context
        if not source_id:raise Reject('compact SWE source lacks exact file/session identity')
        yield compact_context(records,source_id)
        return
    by_id = {}
    conversational = {}
    aliases = {}
    for index, value in records:
        if not isinstance(value, dict):
            raise Reject('SWE raw event is not an object')
        identifier = value.get('uuid')
        if identifier:
            if identifier in by_id:
                if by_id[identifier][1] != value:
                    raise Reject('conflicting duplicate SWE raw event identity')
                aliases.setdefault(identifier, []).append(index)
                continue
            by_id[identifier] = (index, value)
        message = value.get('message')
        if isinstance(message, dict) and message.get('role'):
            if not identifier:
                raise Reject('SWE conversational event lacks its native identity')
            conversational[identifier] = (index, value)
        elif value.get('type') in ('user', 'assistant'):
            raise Reject('unsupported SWE conversational event format; complete context remains owed')
    if not conversational:
        raise Reject('SWE source file contains no supported complete original conversation')
    predecessor = {}
    missing = {}
    for identifier, (_, value) in conversational.items():
        parent = value.get('parentUuid'); seen = {identifier}
        while parent:
            if parent in seen:
                raise Reject('cycle in original SWE event ancestry')
            seen.add(parent)
            if parent not in by_id:
                missing[identifier] = parent
                parent = None
                break
            if parent in conversational:
                break
            event = by_id[parent][1]
            if event.get('subtype') == 'compact_boundary':
                parent = None
                break
            parent = event.get('parentUuid')
        predecessor[identifier] = parent
    checked = set()
    for start in predecessor:
        current = start; chain = set()
        while current and current not in checked:
            if current in chain:
                raise Reject('cycle in original SWE conversational ancestry')
            chain.add(current); current = predecessor[current]
        checked.update(chain)
    ancestors = set(predecessor.values())
    leaves = [key for key in conversational if key not in ancestors]
    for leaf in leaves:
        ancestry = []; identifier = leaf; seen = set()
        while identifier:
            if identifier in seen:
                raise Reject('cycle in original SWE conversational ancestry')
            seen.add(identifier); ancestry.append(identifier); identifier = predecessor[identifier]
        ancestry.reverse()
        original = [conversational[key][1]['message'] for key in ancestry]
        sessions = {conversational[key][1].get('sessionId') for key in ancestry} - {None}
        if len(sessions) != 1:
            raise Reject('SWE causal context mixes or lacks original session identity')
        messages = complete_messages(original)
        if not any(m['role'] == 'assistant' for m in messages):
            continue
        yield conversational[leaf][0], messages, {
            'session_id': next(iter(sessions)), 'leaf_uuid': leaf,
            'original_message_rows': [conversational[key][0] for key in ancestry],
            'original_message_uuids': ancestry,
            'duplicate_event_aliases': {key: aliases[key] for key in ancestry if key in aliases},
            'missing_ancestry': {key: missing[key] for key in ancestry if key in missing},
            'original_models': sorted({str(conversational[key][1]['message']['model']) for key in ancestry
                                       if conversational[key][1]['message'].get('model')}),
            'raw_context_not_normalized_table': True}


def swe_chat_acquired(limit=None):
    """Whole raw contexts; every source event has an explicit export accounting row."""
    from .acquired_agents import acquired_rows, _reported_session
    from .records import group_key
    previous = None; buffered = []; receipt = None

    def finish(receipt, path, records):
        if not path.startswith(('transcripts/', 'transcripts_subagent/', 'transcripts_compact/')):
            for index, _ in records:
                yield path+':'+str(index), Reject('SWE source index/alternate view; raw complete transcripts are consumers')
            return
        produced = {}
        try:
            source_index = swe_source_index(receipt)
            known_files = {entry['path'] for entry in receipt['files']} | set(receipt.get('pending_files', []))
            if path.startswith('transcripts_compact/') and path.replace('transcripts_compact/','transcripts/',1) in known_files:
                raise Reject('duplicate compacted representation; complete original raw transcript is the consumer')
            original_sessions = source_index['sessions'].get(path, [])
            original_logs = source_index['logs'].get(path, [])
            original_tasks = source_index['tasks'].get(path, [])
            original_sessions = list(original_sessions)
            for item in original_logs + original_tasks:
                for session in source_index['session_ids'].get(item['value'].get('session_id'), []):
                    if session not in original_sessions:original_sessions.append(session)
            if not original_sessions and not original_logs and not original_tasks:
                raise Reject('SWE raw file lacks its exact original relational transcript-path binding')
            for index, messages, evidence in swe_raw_contexts(records,source_id=path):
                rid = path+':'+str(index)
                t = _reported_session('D20', receipt, rid, {}, messages,
                    task_id=evidence['session_id'], session_id=evidence['leaf_uuid'],
                    teacher=evidence['original_models'][0] if len(evidence['original_models'])==1 else None)
                t.messages = messages
                evidence.update(original_session_records=original_sessions, original_session_logs=original_logs,
                    original_subagent_task_aliases=original_tasks, relational_index_bindings=source_index['bindings'], source_metadata_encoding=source_index['metadata_encoding'],
                    communication_join_status='pending-exact-parent-spawn-and-observed-return-proof' if original_tasks else 'not-asserted',
                    fabricated_communication_edges=0)
                for item in original_sessions + original_tasks:
                    original = item['value']
                    if original.get('repo_id'):
                        repository = source_index['repositories'].get(original['repo_id'])
                        if repository is None or repository.get('error'):raise Reject('SWE context lacks unique exact canonical repository join')
                        t.split_groups.append(group_key('repo', repository['canonical']))
                        t.split_groups.append(group_key('swe-chat-repository-id', original['repo_id']))
                        evidence.setdefault('canonical_repository_bindings', {})[original['repo_id']] = repository
                    if original.get('split') in ('test','validation','eval','held_out') or original.get('held_out') is True or original.get('is_heldout') is True:
                        t.split = 'test'
                    if original.get('session_id'):
                        t.split_groups.append(group_key('swe-chat-source-session', original['session_id']))
                t.split_groups.append(group_key('swe-chat-session', evidence['session_id']))
                t.lineage['notes']['original_swe_context'] = evidence
                missing_repo = not evidence.get('canonical_repository_bindings')
                source_zero_score = any(str(item['value'].get('session_success')).strip() in ('0','0.0')
                                        for item in original_sessions)
                api_errors = [index for index, event in records if event.get('isApiErrorMessage') is True
                              and index in evidence['original_message_rows']]
                if evidence['missing_ancestry'] or evidence.get('redacted_context_rows') or source_zero_score or api_errors or missing_repo:
                    t.lineage['notes']['source_quality_flags'] = [
                        *(['incomplete-original-causal-ancestry'] if evidence['missing_ancestry'] else []),
                        *(['upstream-redacted-original-context'] if evidence.get('redacted_context_rows') else []),
                        *(['upstream-zero-session-success-annotation'] if source_zero_score else []),
                        *(['upstream-api-error-response-context'] if api_errors else []),
                        *(['missing-source-canonical-repository-binding'] if missing_repo else [])]
                evidence['original_api_error_rows'] = api_errors
                produced[index] = t
        except (Reject, ValueError, TypeError, KeyError) as error:
            for index, _ in records:
                yield path+':'+str(index), Reject(str(error))
            return
        for index, _ in records:
            yield path+':'+str(index), produced.get(index, Reject(
                'original SWE event retained by complete raw context or auxiliary event inventory'))

    # A source-row limit must never truncate a complete raw causal context.
    # Callers select exact receipt files with ACQUIRED_FILES for durable shards.
    if limit is not None:
        raise Reject('SWE complete contexts require exact file selection, not a raw-event limit')
    for current, rid, row in acquired_rows('D20'):
        path, index = rid.rsplit(':', 1)
        if previous is not None and path != previous:
            yield from finish(receipt, previous, buffered); buffered = []
        receipt = current; previous = path; buffered.append((int(index), row))
    if previous is not None:
        yield from finish(receipt, previous, buffered)


_SWE_INDEX = {}


def swe_source_index(receipt):
    """Prepare the exact pinned small relational indexes once across file shards."""
    from . import agents
    from .acquired_agents import _bound_acquired_path, iter_acquired_file_rows
    from natlang_neuralese.common.hashing import canonical_json_sha256_hex
    root = agents.EXPANSION_ROOT/'D20'
    names = ('sessions.parquet', 'session_logs.parquet', 'subagent_tasks.parquet', 'repositories.parquet')
    entries = {entry['path']: entry for entry in receipt['files'] if entry['path'] in names}
    if set(entries) != set(names):
        raise Reject('SWE raw source requires bound sessions, session_logs, subagent_tasks and repositories indexes')
    # Verification remains on each invocation, including reuse; digest cache
    # identity includes stat information in the shared hashing implementation.
    paths = {name: _bound_acquired_path(root, entry) for name, entry in entries.items()}
    key = (str(root.resolve()), receipt['revision'], canonical_json_sha256_hex(entries))
    if key not in _SWE_INDEX:
        sessions = {}; logs = {}; tasks = {}; session_ids = {}; repositories = {}
        for row_index, row in iter_acquired_file_rows(paths['repositories.parquet']):
            identity = row.get('repo_id'); owner = row.get('owner_id'); name = row.get('name')
            if not isinstance(identity,str) or not identity or not isinstance(owner,str) or not owner or not isinstance(name,str) or not name or '/' in owner or '/' in name:
                repositories[str(identity)] = {'error':'SWE repository index lacks exact canonical owner/name identity','row':row_index}
                continue
            canonical = owner+'/'+name
            if identity in repositories and repositories[identity].get('canonical') != canonical:
                repositories[identity] = {'error':'SWE repository index ambiguously maps one source ID','row':row_index}
                continue
            repositories[identity] = {'row':row_index,'canonical':canonical,'source_row_sha256':_digest(row)}
        for row_index, row in iter_acquired_file_rows(paths['sessions.parquet']):
            row = _metadata(row)
            if row.get('session_id'):
                session_ids.setdefault(row['session_id'], []).append({'row':row_index,'value':row})
            path = row.get('transcript_path')
            if path:
                sessions.setdefault(path, []).append({'row':row_index,'value':row})
        for row_index, row in iter_acquired_file_rows(paths['session_logs.parquet']):
            row = _metadata(row)
            for field in ('transcript_path', 'compact_transcript_path'):
                if row.get(field):logs.setdefault(row[field], []).append({'row':row_index,'value':row})
        for row_index, row in iter_acquired_file_rows(paths['subagent_tasks.parquet']):
            row = _metadata(row)
            if row.get('transcript_path'):
                tasks.setdefault(row['transcript_path'], []).append({'row':row_index,'value':row})
        _SWE_INDEX[key] = {'bindings':entries,'sessions':sessions,'logs':logs,'tasks':tasks,
                           'session_ids':session_ids,'repositories':repositories,'metadata_encoding':'typed-arrow-json-v1'}
    return _SWE_INDEX[key]
