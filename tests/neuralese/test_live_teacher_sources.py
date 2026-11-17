"""Observed workspace episodes share SFT admission without claiming portable replay."""
import collections
import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/neuralese_data'))
import live_teacher_sources as live
import sft_sources as s
from natlang_neuralese.common.hashing import sha256_file_hex


def artifact(root, name, value):
    path = root / name
    path.write_text(json.dumps(value))
    return {'path': name, 'sha256': sha256_file_hex(path)}


def fixture(root):
    request = {'invocation_id': 'call-one', 'messages': [{'role': 'system', 'content': 'Actual executor prompt'},
                {'role': 'user', 'content': 'Create the requested draft'}],
               'tools': [{'name': 'eval', 'description': 'Execute code'}]}
    gateway = {'model': 'luna', 'api': 'actual-provider', 'turnId': 'actual-gateway-turn',
               'attribution': {'callerId': 'actual-runtime-caller', 'callerKind': 'eval'},
               'context': {'messages': [{'role': 'user', 'content': 'Actual Pi context'}]},
               'response': {'role': 'assistant', 'content': [{'type': 'toolCall', 'id': 'actual-tool-id',
                               'name': 'eval', 'arguments': {'code': 'workspace.write("draft", "hello")'}}]}}
    exchange = {'schema': 'natlang.model-exchange/1', 'kind': 'generation', 'request': request,
                'reply': {'text': '', 'raw_calls': [{'id': 'actual-tool-id', 'type': 'function', 'function':
                    {'name': 'eval', 'arguments': json.dumps({'code': 'workspace.write("draft", "hello")'})}}], 'calls': [['eval', {'code': 'workspace.write("draft", "hello")'}]],
                          'raw_response': {'vibestudio': gateway}}}
    summary = {'call_id': 'call-one', 'task_id': 'internal-task', 'owner_kind': 'eval',
               'owner_key': 'eval-owner', 'run_id': 'actual-eval-run'}
    detail = {'callId': 'call-one', 'summary': summary,
              'record': {'version': 'natlang.calls/1', 'call_id': 'call-one', 'task_id': 'internal-task',
                         'executor': {'kind': 'agent'}, 'definition': {'id': 'definition-one', 'name': 'createDraft'},
                         'model_turns': [exchange, {**exchange, 'kind': 'score'}]}}
    verification = {'schema': 'natlang.live-task-verification/1', 'task_id': 'live-draft',
                    'run_id': 'outer-world-run', 'world': {'instance': 'test-owned-instance', 'run_id': 'outer-world-run'},
                    'status': 'passed', 'checks': {'draft_contents': True},
                    'calls': [copy.deepcopy(summary)],
                    'validator': artifact(root, 'validator.json', {'code': 'independent validator source'}),
                    'evidence': [{**artifact(root, 'readback.json', {'before': None, 'after': 'hello'}),
                                  'kind': 'workspace'}]}
    spec = {'schema': 'natlang.live-teacher-task/1', 'id': 'live-draft', 'family': 'workspace-draft',
            'split': 'train', 'source_namespace': 'vibestudio:live:workspace', 'source_ids': ['draft-one'],
            'source_groups': ['world-family-one'], 'source_revisions': ['revision-one'],
            'code': {'git_commit': 'a' * 40, 'dirty': False}, 'learner_kinds': ['natlang-executor'],
            'world': {'instance': 'test-owned-instance', 'run_id': 'outer-world-run'},
            'required_effect_kinds': ['workspace'],
            'calls': [artifact(root, 'call.json', detail)],
            'verification': artifact(root, 'verification.json', verification)}
    path = root / 'manifest.json'
    path.write_text(json.dumps(spec))
    return path, spec, detail, verification


def rewrite(root, spec, detail, verification):
    spec['calls'] = [artifact(root, 'call.json', detail)]
    spec['verification'] = artifact(root, 'verification.json', verification)
    (root / 'manifest.json').write_text(json.dumps(spec))


def test_actual_context_and_tool_ids_are_preserved_without_score_targets(tmp_path):
    path, _, detail, _ = fixture(tmp_path)
    episode = live.load_episode(path)
    assert 'version' not in episode['task']  # no invented ProgramRecord
    assert episode['verification_mode'] == 'observed-live'
    assert len(episode['turns']) == 1
    turn = episode['turns'][0]
    assert turn['messages'] == detail['record']['model_turns'][0]['request']['messages']
    assert turn['target']['tool_calls'][0]['id'] == 'actual-tool-id'
    assert json.loads(turn['target']['tool_calls'][0]['function']['arguments']) == {'code': 'workspace.write("draft", "hello")'}
    assert turn['live_exchange']['gateway']['context'] == detail['record']['model_turns'][0]['reply']['raw_response']['vibestudio']['context']
    assert 'native_sampling' not in turn
    assert s.replay_episode(None, None, episode, None)['exact'] is False


@pytest.mark.parametrize('mutation', ['owner', 'eval-run', 'outer-run', 'effect', 'checks', 'kind', 'split', 'tool-id', 'tool-args', 'raw-id', 'text', 'coverage', 'instance'])
def test_rehashed_wrong_scope_effect_or_target_is_rejected(tmp_path, mutation):
    path, spec, detail, verification = fixture(tmp_path)
    if mutation == 'owner':
        detail['summary']['owner_key'] = 'foreign-owner'
    elif mutation == 'eval-run':
        detail['summary']['run_id'] = 'outer-world-run'
    elif mutation == 'outer-run':
        verification['run_id'] = 'foreign-world-run'
    elif mutation == 'instance':
        spec['world']['instance'] = 'foreign-instance'
    elif mutation == 'effect':
        verification['evidence'][0]['kind'] = 'model-claim'
    elif mutation == 'checks':
        verification['checks']['draft_contents'] = False
    elif mutation == 'kind':
        detail['record']['executor']['kind'] = 'crisp'
    elif mutation == 'split':
        spec['split'] = 'test'
    elif mutation == 'tool-id':
        del detail['record']['model_turns'][0]['reply']['raw_response']['vibestudio']['response']['content'][0]['id']
    elif mutation == 'raw-id':
        detail['record']['model_turns'][0]['reply']['raw_calls'][0]['id'] = 'fabricated-id'
    elif mutation == 'text':
        detail['record']['model_turns'][0]['reply']['text'] = 'invented-answer'
    elif mutation == 'coverage':
        verification['calls'].append({**verification['calls'][0], 'call_id': 'uncaptured-child'})
    elif mutation == 'tool-args':
        detail['record']['model_turns'][0]['reply']['calls'][0][1] = {'code': 'different'}
    rewrite(tmp_path, spec, detail, verification)
    with pytest.raises(ValueError):
        live.load_episode(path)


def test_shared_admission_rechecks_bound_originals_without_replay(tmp_path):
    path, _, _, _ = fixture(tmp_path)
    episode = live.load_episode(path)
    writer = s.ZstdWriter(tmp_path / 'episodes-live.jsonl.zst')
    writer.write(episode)
    writer.close()
    funnel = collections.defaultdict(collections.Counter)
    admitted = list(s.verified_episodes(tmp_path, funnel))
    assert admitted[0][0] == episode
    assert admitted[0][1]['name'] is None
    assert funnel['teacher']['observed_live_verified'] == 1
    assert not (tmp_path / 'replay.jsonl').exists()
    (tmp_path / 'readback.json').write_text('{}')
    with pytest.raises(ValueError, match='artifact changed'):
        list(s.verified_episodes(tmp_path, funnel))
