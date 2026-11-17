"""Admission of observed live Natlang work; effects are verified, never called portable replays."""
import json
from pathlib import Path

from natlang_neuralese.common.hashing import sha256_file_hex


def bound_file(binding, root):
    path = Path(binding['path'])
    if not path.is_absolute():
        path = root / path
    if sha256_file_hex(path) != binding.get('sha256'):
        raise ValueError(f'live source artifact changed: {path}')
    return path


def load_episode(path):
    path = Path(path).resolve()
    spec = json.loads(path.read_text())
    if spec.get('schema') != 'natlang.live-teacher-task/1':
        raise ValueError('unsupported live teacher task manifest')
    for field in ('id', 'family', 'source_namespace'):
        if not isinstance(spec.get(field), str) or not spec[field]:
            raise ValueError(f'live task needs {field}')
    for field in ('source_ids', 'source_groups', 'source_revisions'):
        if not isinstance(spec.get(field), list) or not spec[field] or any(not isinstance(x, str) or not x for x in spec[field]):
            raise ValueError(f'live task needs {field}')
    if spec.get('split') != 'train':
        raise ValueError('live source is not a training task')
    if spec.get('learner_kinds') != ['natlang-executor']:
        raise ValueError('select actual Natlang executor generations explicitly')
    if (spec.get('code') or {}).get('dirty') is not False or not (spec.get('code') or {}).get('git_commit'):
        raise ValueError('live source needs clean capture-code provenance')
    world = spec.get('world') or {}
    if not world.get('run_id') or not world.get('instance'):
        raise ValueError('live task must identify the actual test world and outer run')
    verification = json.loads(bound_file(spec['verification'], path.parent).read_text())
    if (verification.get('schema') != 'natlang.live-task-verification/1' or
            verification.get('task_id') != spec['id'] or verification.get('run_id') != world['run_id'] or
            verification.get('world') != world or
            verification.get('status') != 'passed'):
        raise ValueError('live task verification does not attest this actual task/run')
    checks = verification.get('checks')
    if not isinstance(checks, dict) or not checks or any(v is not True for v in checks.values()):
        raise ValueError('independent live task checks did not all pass')
    bound_file(verification['validator'], path.parent)
    kinds = set()
    for evidence in verification.get('evidence') or []:
        bound_file(evidence, path.parent)
        kinds.add(evidence.get('kind'))
    required = set(spec.get('required_effect_kinds') or [])
    if not required or not required <= {'workspace', 'cdp'} or not required <= kinds:
        raise ValueError('missing actual workspace/CDP effect readback evidence')
    scopes = {}
    for scope in verification.get('calls') or []:
        if scope.get('call_id') in scopes:
            raise ValueError('duplicate call in live validation scope')
        if any(not scope.get(k) for k in ('call_id', 'owner_kind', 'owner_key', 'run_id')):
            raise ValueError('live validation needs exact authoritative eval ownership')
        scopes[scope['call_id']] = scope
    turns, seen, models = [], set(), set()
    for binding in spec.get('calls') or []:
        detail = json.loads(bound_file(binding, path.parent).read_text())
        record, summary = detail['record'], detail['summary']
        call_id = record['call_id']
        if call_id in seen or call_id not in scopes or detail.get('callId') != call_id:
            raise ValueError('live call is duplicate or outside the verified workflow')
        seen.add(call_id)
        if summary.get('call_id') != call_id or summary.get('task_id') != record.get('task_id'):
            raise ValueError('exported call identity differs from original record')
        if any(summary.get(k) != scopes[call_id].get(k) for k in ('owner_kind', 'owner_key', 'run_id')):
            raise ValueError('live call owner/eval run differs from verified scope')
        if record.get('version') != 'natlang.calls/1' or record.get('executor', {}).get('kind') != 'agent':
            raise ValueError('live source must contain original model-executed calls')
        for index, exchange in enumerate(record.get('model_turns') or []):
            if exchange.get('schema') != 'natlang.model-exchange/1':
                raise ValueError('unsupported live model exchange')
            if exchange.get('kind') != 'generation' or exchange.get('error'):
                continue
            request, reply = exchange['request'], exchange.get('reply')
            if not isinstance(reply, dict) or not isinstance(request.get('messages'), list):
                raise ValueError('live generation lacks actual request/reply')
            gateway = (reply.get('raw_response') or {}).get('vibestudio') or {}
            attribution = gateway.get('attribution') or {}
            original = gateway.get('response') or {}
            if (not gateway.get('model') or not gateway.get('api') or not gateway.get('turnId') or
                    not attribution.get('callerId') or not attribution.get('callerKind') or
                    not gateway.get('context') or original.get('role') != 'assistant' or
                    not isinstance(original.get('content'), list)):
                raise ValueError('live generation lacks original gateway identity/context/response')
            blocks = original['content']
            original_calls = [block for block in blocks if block.get('type') == 'toolCall']
            calls = reply.get('raw_calls')
            if not isinstance(calls, list) or len(calls) != len(original_calls):
                raise ValueError('live tool target lacks original assistant tool-call IDs/arguments')
            for raw, actual in zip(calls, original_calls):
                function = raw.get('function') or {}
                if (not actual.get('id') or raw.get('id') != actual['id'] or
                        function.get('name') != actual.get('name') or
                        json.loads(function.get('arguments', 'null')) != actual.get('arguments')):
                    raise ValueError('normalized live calls differ from original assistant response')
            if [[c['name'], c['arguments']] for c in original_calls] != (reply.get('calls') or []):
                raise ValueError('normalized live calls differ from original assistant response')
            text = ''.join(block['text'] for block in blocks if block.get('type') == 'text')
            reasoning = '\n'.join(block['thinking'] for block in blocks if block.get('type') == 'thinking')
            if (reply.get('text') or '') != text or (reply.get('reasoning') or '') != reasoning:
                raise ValueError('normalized live text/reasoning differs from original assistant response')
            models.add(gateway['model'])
            target = {'role': 'assistant', 'content': text}
            if reasoning:
                target['reasoning_content'] = reasoning
            if calls:
                target['tool_calls'] = calls
            if not target['content'] and not target.get('tool_calls'):
                continue
            definition = record.get('definition') or {}
            turns.append({'messages': request['messages'], 'tools': request.get('tools') or [], 'target': target,
                'invocation_id': call_id, 'function': definition.get('name') or 'unknown',
                'definition_id': definition.get('id'), 'parent_invocation': record.get('parent_call_id'),
                'invocation_role': 'child' if record.get('parent_call_id') else 'root',
                'live_exchange': {'call_artifact': binding, 'ordinal': index, 'gateway': gateway,
                                  'target_source': 'actual-model-reply', 'student_sampling_claimed': False}})
    if seen != set(scopes) or not turns:
        raise ValueError('workflow call coverage incomplete or has no generated executor targets')
    task = {key: spec[key] for key in ('id', 'family', 'split', 'source_namespace', 'source_ids', 'source_groups', 'source_revisions')}
    task['source'] = 'observed-live-workspace'
    return {'key': f'live:{spec["id"]}:{sha256_file_hex(path)}', 'source': 'teacher', 'run': str(path),
            'task_id': spec['id'], 'family': spec['family'], 'task': task, 'turns': turns,
            'teacher': {'model_id': ','.join(sorted(models))},
            'verification_mode': 'observed-live', 'live_source': {'path': str(path), 'sha256': sha256_file_hex(path)},
            'world': world}
