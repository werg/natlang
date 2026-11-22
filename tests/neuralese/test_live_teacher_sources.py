"""Observed workspace episodes share SFT admission without claiming portable replay."""
import collections
import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

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
    summary = {'call_id': 'call-one', 'parent_call_id': None, 'definition_id': 'definition-one', 'owner_kind': 'eval',
               'owner_key': 'eval-owner', 'run_id': 'actual-eval-run'}
    detail = {'callId': 'call-one', 'summary': summary,
              'record': {'version': 'natlang.calls/1', 'call_id': 'call-one', 'parent_call_id': None, 'outcome': 'done', 'task_id': 'internal-task',
                         'executor': {'kind': 'agent'}, 'definition': {'id': 'definition-one', 'name': 'createDraft'},
                         'model_turns': [exchange, {**exchange, 'kind': 'score'}]}}
    verification = {'schema': 'natlang.live-task-verification/1', 'task_id': 'live-draft',
                    'run_id': 'outer-world-run', 'world': {'instance': 'test-owned-instance', 'run_id': 'outer-world-run'},
                    'status': 'passed', 'checks': {'draft_contents': True},
                    'root_call_ids': ['call-one'], 'calls': [copy.deepcopy(summary)],
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


def open_goal(detail, spec, verification):
    goal = 'Bring the workshop update in line with the approved notes.'
    encoded = json.dumps(goal).encode()
    interaction = {'kind': 'open-user-goal', 'goal': goal, 'program': 'skills/open-user-goal/runUserGoal.nl'}
    spec['interaction'] = interaction
    verification['interaction'] = copy.deepcopy(interaction)
    detail['record']['definition']['name'] = 'runUserGoal'
    detail['record']['inputs'] = {'goal': {'complete': True, 'bytes': len(encoded),
                                          'hash': hashlib.sha256(encoded).hexdigest()}}
    return interaction


@pytest.mark.parametrize('mutation', [None, 'goal', 'extra-input', 'unattested'])
def test_open_goal_has_to_match_the_original_goal_only_call(tmp_path, mutation):
    path, spec, detail, verification = fixture(tmp_path)
    interaction = open_goal(detail, spec, verification)
    detail['events'] = [{'kind': 'action', 'outcome': 'completed', 'tool_call_id': 'actual-tool-id'}]
    if mutation == 'goal':
        interaction['goal'] = verification['interaction']['goal'] = 'Substituted goal'
    elif mutation == 'extra-input':
        detail['record']['inputs']['work'] = {}
    elif mutation == 'unattested':
        del verification['interaction']
    rewrite(tmp_path, spec, detail, verification)
    if mutation:
        with pytest.raises(ValueError, match='open goal|interaction'):
            live.load_episode(path)
    else:
        assert live.load_episode(path)['task']['interaction'] == interaction


def test_open_goal_sft_uses_native_trajectory_and_retains_rejected_actions(tmp_path):
    path, spec, detail, verification = recovery_fixture(tmp_path)
    interaction = open_goal(detail, spec, verification)
    detail['events'][0]['outcome'] = 'rejected'
    detail['events'].append({'kind': 'action', 'outcome': 'completed', 'tool_call_id': 'actual-recovery-id'})
    detail['record']['model_turns'].append({**detail['record']['model_turns'][0], 'kind': 'score'})
    rewrite(tmp_path, spec, detail, verification)
    episode = live.load_episode(path)
    ir = episode['trajectory_ir']
    assert ir['version'] == 'natlang.teacher_trajectory.native/1'
    assert ir['task']['interaction'] == interaction
    assert ir['outcome']['invocation_ledger'][0]['events'] == detail['events']
    assert [turn['training_admission']['approved'] for turn in ir['trajectory']] == [False, True, False]
    assert ir['trajectory'][0]['assistant']['calls'][0]['id'] == 'actual-tool-id'
    assert episode['turns'][0]['target']['tool_calls'][0]['id'] == 'actual-recovery-id'
    assert episode['turns'][0]['messages'][-1]['content'] == 'write failed'
    native_out = tmp_path / 'teacher.native.jsonl.zst'
    s.extract(SimpleNamespace(work=tmp_path, out_name='episodes-live', rl=[], teacher=[], live_run=[path.as_posix()],
                              live_trajectory_out=native_out))
    assert json.loads(next(s.zstd_lines(native_out))) == ir
    assert 'turns' not in json.loads(next(s.zstd_lines(tmp_path / 'episodes-live.jsonl.zst')))
    admitted = list(s.verified_episodes(tmp_path, collections.defaultdict(collections.Counter)))
    assert admitted[0][0] == episode


def test_open_goal_excludes_a_proposed_action_without_execution_evidence(tmp_path):
    path, spec, detail, verification = fixture(tmp_path)
    open_goal(detail, spec, verification)
    rewrite(tmp_path, spec, detail, verification)
    with pytest.raises(ValueError, match='no generated executor targets'):
        live.load_episode(path)


def test_open_goal_binds_the_full_recorded_child_inventory(tmp_path):
    path, spec, detail, verification = fixture(tmp_path)
    open_goal(detail, spec, verification)
    detail['events'] = [{'kind': 'action', 'outcome': 'completed', 'tool_call_id': 'actual-tool-id'}]
    custody = {'schema': 'natlang.call-graph-chronology/1', 'task_id': 'internal-task',
               'sha256': '1' * 64, 'child_call_ids': ['missing-crisp-attempt']}
    detail['record']['graph_custody'] = custody
    rewrite(tmp_path, spec, detail, verification)
    with pytest.raises(ValueError, match='recorded child inventory'):
        live.load_episode(path)
    custody['child_call_ids'] = []
    rewrite(tmp_path, spec, detail, verification)
    assert live.load_episode(path)['trajectory_ir']['outcome']['invocation_ledger'][0]['graph_custody'] == custody


@pytest.mark.parametrize('mutation', ['owner', 'definition', 'eval-run', 'outer-run', 'effect', 'checks', 'kind', 'split', 'tool-id', 'tool-args', 'raw-id', 'text', 'coverage', 'instance'])
def test_rehashed_wrong_scope_effect_or_target_is_rejected(tmp_path, mutation):
    path, spec, detail, verification = fixture(tmp_path)
    if mutation == 'owner':
        detail['summary']['owner_key'] = 'foreign-owner'
    elif mutation == 'definition':
        detail['summary']['definition_id'] = 'different-definition'
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
        detail['record']['executor']['kind'] = 'outer-agent'
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


def tree_fixture(root):
    path, spec, detail, verification = fixture(root)
    child = copy.deepcopy(detail)
    child['callId'] = child['record']['call_id'] = child['summary']['call_id'] = 'agent-child'
    child['record']['parent_call_id'] = child['summary']['parent_call_id'] = 'call-one'
    child['record']['executor']['kind'] = 'crisp-agent'
    child['record']['definition']['name'] = 'recoverAndWrite'
    crisp = copy.deepcopy(child)
    crisp['callId'] = crisp['record']['call_id'] = crisp['summary']['call_id'] = 'mechanism-child'
    crisp['record']['executor']['kind'] = 'crisp'
    del crisp['record']['model_turns']
    failed = copy.deepcopy(child)
    failed['callId'] = failed['record']['call_id'] = failed['summary']['call_id'] = 'failed-child'
    failed['record']['outcome'] = 'failed'
    spec['calls'] += [artifact(root, 'child.json', child), artifact(root, 'crisp.json', crisp),
                      artifact(root, 'failed.json', failed)]
    verification['calls'] += [child['summary'], crisp['summary'], failed['summary']]
    spec['verification'] = artifact(root, 'verification.json', verification)
    path.write_text(json.dumps(spec))
    return path, spec, child, verification


def test_whole_tree_keeps_crisp_and_failed_evidence_but_only_successful_agent_targets(tmp_path):
    path, _, _, _ = tree_fixture(tmp_path)
    episode = live.load_episode(path)
    assert len(episode['call_tree']) == 4
    assert [turn['invocation_id'] for turn in episode['turns']] == ['call-one', 'agent-child']
    assert episode['turns'][1]['live_exchange']['call_ancestry'] == ['call-one', 'agent-child']
    assert episode['turns'][1]['live_exchange']['root_call_id'] == 'call-one'
    assert episode['call_tree'][2]['executor'] == 'crisp'
    assert episode['call_tree'][3]['outcome'] == 'failed'


@pytest.mark.parametrize('mutation', ['missing-parent', 'cycle', 'wrong-root', 'summary-parent'])
def test_parent_closure_and_verified_root_attribution_are_required(tmp_path, mutation):
    path, spec, child, verification = tree_fixture(tmp_path)
    if mutation == 'missing-parent':
        child['record']['parent_call_id'] = child['summary']['parent_call_id'] = 'omitted-parent'
    elif mutation == 'cycle':
        child['record']['parent_call_id'] = child['summary']['parent_call_id'] = 'agent-child'
    elif mutation == 'wrong-root':
        verification['root_call_ids'] = ['different-root']
    elif mutation == 'summary-parent':
        child['summary']['parent_call_id'] = 'foreign-parent'
    spec['calls'][1] = artifact(tmp_path, 'child.json', child)
    spec['verification'] = artifact(tmp_path, 'verification.json', verification)
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match='parent|roots'):
        live.load_episode(path)


def recovery_fixture(root):
    path, spec, detail, verification = fixture(root)
    failed = detail['record']['model_turns'][0]
    recovery = copy.deepcopy(failed)
    recovery['reply']['raw_calls'][0]['id'] = 'actual-recovery-id'
    recovery['reply']['raw_response']['vibestudio']['response']['content'][0]['id'] = 'actual-recovery-id'
    recovery['request']['messages'] += [{'role': 'assistant', 'content': failed['reply']['text'],
                                          'tool_calls': copy.deepcopy(failed['reply']['raw_calls'])},
                                        {'role': 'tool', 'tool_call_id': 'actual-tool-id', 'content': 'write failed'}]
    detail['record']['model_turns'] = [failed, recovery]
    detail['events'] = [{'kind': 'action', 'outcome': 'error', 'tool_call_id': 'actual-tool-id'}]
    rewrite(root, spec, detail, verification)
    return path, spec, detail, verification


@pytest.mark.parametrize('multiple_actions', [False, True])
def test_recovered_error_is_context_evidence_and_never_supervised_target(tmp_path, multiple_actions):
    path, spec, detail, verification = recovery_fixture(tmp_path)
    if multiple_actions:
        failed = detail['record']['model_turns'][0]['reply']
        extra = copy.deepcopy(failed['raw_calls'][0])
        extra['id'] = 'co-emitted-successful-action'
        failed['raw_calls'].append(extra)
        failed['calls'].append(copy.deepcopy(failed['calls'][0]))
        block = copy.deepcopy(failed['raw_response']['vibestudio']['response']['content'][0])
        block['id'] = extra['id']
        failed['raw_response']['vibestudio']['response']['content'].append(block)
        detail['record']['model_turns'][1]['request']['messages'][-2]['tool_calls'] = copy.deepcopy(failed['raw_calls'])
        rewrite(tmp_path, spec, detail, verification)
    before = copy.deepcopy(detail)
    episode = live.load_episode(path)
    assert len(episode['turns']) == 1
    turn = episode['turns'][0]
    assert turn['target']['tool_calls'][0]['id'] == 'actual-recovery-id'
    assert turn['messages'] == detail['record']['model_turns'][1]['request']['messages']
    assert turn['messages'][-1]['content'] == 'write failed'
    assert turn['live_exchange']['ordinal'] == 1
    assert episode['call_tree'][0]['excluded_failed_action_turns'] == {'0': ['actual-tool-id']}
    assert detail == before
    writer = s.ZstdWriter(tmp_path / 'episodes-live.jsonl.zst')
    writer.write(episode)
    writer.close()
    assert len(list(s.verified_episodes(tmp_path, collections.defaultdict(collections.Counter)))) == 1


@pytest.mark.parametrize('mutation', ['missing-id', 'unmatched-id', 'ambiguous-id'])
def test_failed_action_needs_exact_original_generation_attribution(tmp_path, mutation):
    path, spec, detail, verification = recovery_fixture(tmp_path)
    if mutation == 'missing-id':
        del detail['events'][0]['tool_call_id']
    elif mutation == 'unmatched-id':
        detail['events'][0]['tool_call_id'] = 'foreign-call'
    else:
        detail['record']['model_turns'].append(copy.deepcopy(detail['record']['model_turns'][0]))
    rewrite(tmp_path, spec, detail, verification)
    with pytest.raises(ValueError, match='failed live action'):
        live.load_episode(path)


@pytest.mark.parametrize('mutation', [None, 'argument', 'callable', 'program', 'missing-input'])
def test_application_entry_binds_native_function_and_all_typed_inputs(tmp_path, mutation):
    path, spec, detail, verification = fixture(tmp_path)
    interaction = open_goal(detail, spec, verification)
    arguments = {'approval': {'revision': 'approved-2'}, 'draft': 'current draft'}
    entry = {'program': 'skills/documents/reconcileDraft.nl', 'callable': 'reconcileDraft', 'arguments': arguments}
    interaction.update(program=entry['program'], entry=entry)
    verification['interaction'] = copy.deepcopy(interaction)
    detail['record']['definition']['name'] = 'reconcileDraft'
    detail['record']['definition']['source'] = 'reconcileDraft.nl'
    detail['record']['program_root'] = 'skills/documents'
    detail['record']['inputs'] = {
        name: dict(zip(('bytes', 'hash'), live.open_goal_fingerprint(value)), complete=True)
        for name, value in arguments.items()}
    detail['events'] = [{'kind': 'action', 'outcome': 'completed', 'tool_call_id': 'actual-tool-id'}]
    if mutation == 'argument':
        detail['record']['inputs']['approval']['hash'] = '0' * 64
    elif mutation == 'callable':
        detail['record']['definition']['name'] = 'runUserGoal'
    elif mutation == 'program':
        entry['program'] = '../outside.nl'
        verification['interaction'] = copy.deepcopy(interaction)
    elif mutation == 'missing-input':
        del detail['record']['inputs']['draft']
    rewrite(tmp_path, spec, detail, verification)
    if mutation:
        with pytest.raises(ValueError, match='entry|invocation|input'):
            live.load_episode(path)
    else:
        episode = live.load_episode(path)
        assert episode['task']['interaction']['entry'] == entry
        assert episode['turns'][0]['function'] == 'reconcileDraft'


def partial_fixture(root):
    path, spec, _, verification = tree_fixture(root)
    original = artifact(root, 'original-outcome.json', {'passed': False, 'checks': {'final_answer': False}})
    detail = json.loads((root / 'call.json').read_text())
    open_goal(detail, spec, verification)
    detail['events'] = [{'kind': 'action', 'outcome': 'completed', 'tool_call_id': 'actual-tool-id'}]
    spec['calls'][0] = artifact(root, 'call.json', detail)
    child = json.loads((root / 'child.json').read_text())
    child['events'] = [{'kind': 'action', 'outcome': 'completed', 'tool_call_id': 'actual-tool-id'}]
    spec['calls'][1] = artifact(root, 'child.json', child)
    verification.update(status='partial', training_scope={
        'schema': 'natlang.training-scope/1', 'original_outcome': original,
        'invocations': [{'call_id': 'agent-child', 'checks': {'written_draft': True},
                         'evidence': copy.deepcopy(verification['evidence'])}]})
    spec['verification'] = artifact(root, 'verification.json', verification)
    path.write_text(json.dumps(spec))
    return path, spec, verification


def test_partial_success_keeps_original_graph_and_excludes_unsupported_parent(tmp_path):
    path, _, _ = partial_fixture(tmp_path)
    episode = live.load_episode(path)
    assert len(episode['call_tree']) == 4
    assert [turn['invocation_id'] for turn in episode['turns']] == ['agent-child']
    ir = episode['trajectory_ir']
    assert ir['outcome']['status'] == 'partial'
    assert ir['trajectory'][0]['training_admission'] == {
        'approved': False, 'reason': 'invocation lacks independent outcome evidence'}
    assert json.loads((tmp_path / 'original-outcome.json').read_text())['passed'] is False


@pytest.mark.parametrize('mutation', ['empty', 'unknown', 'failed', 'check', 'no-evidence', 'foreign-evidence', 'original'])
def test_partial_success_requires_independent_scoped_evidence(tmp_path, mutation):
    path, spec, verification = partial_fixture(tmp_path)
    scope = verification['training_scope']
    selected = scope['invocations'][0]
    if mutation == 'empty':
        scope['invocations'] = []
    elif mutation == 'unknown':
        selected['call_id'] = 'uncaptured'
    elif mutation == 'failed':
        selected['call_id'] = 'failed-child'
    elif mutation == 'check':
        selected['checks']['written_draft'] = False
    elif mutation == 'no-evidence':
        selected['evidence'] = []
    elif mutation == 'foreign-evidence':
        selected['evidence'][0]['sha256'] = '0' * 64
    else:
        scope['original_outcome'] = artifact(tmp_path, 'different-outcome.json', {'passed': True})
    spec['verification'] = artifact(tmp_path, 'verification.json', verification)
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match='scope|outcome|artifact|recovery'):
        live.load_episode(path)


def test_partial_scope_can_recover_actions_without_supervising_wrong_final_status(tmp_path):
    path, spec, verification = partial_fixture(tmp_path)
    detail = json.loads((tmp_path / 'child.json').read_text())
    later = copy.deepcopy(detail['record']['model_turns'][0])
    later['reply']['raw_calls'][0]['id'] = 'later-report'
    later['reply']['raw_response']['vibestudio']['response']['content'][0]['id'] = 'later-report'
    detail['record']['model_turns'].append(later)
    detail['events'].append({'kind': 'action', 'outcome': 'completed', 'tool_call_id': 'later-report'})
    spec['calls'][1] = artifact(tmp_path, 'child.json', detail)
    verification['training_scope']['invocations'][0]['ordinals'] = [0]
    spec['verification'] = artifact(tmp_path, 'verification.json', verification)
    path.write_text(json.dumps(spec))
    episode = live.load_episode(path)
    assert len(episode['turns']) == 1
    excluded = [row for row in episode['trajectory_ir']['trajectory']
                if row['invocation_id'] == 'agent-child' and not row['training_admission']['approved']]
    assert any(row['training_admission']['reason'] == 'decision lacks independent outcome evidence' for row in excluded)
    assert len(episode['trajectory_ir']['outcome']['invocation_ledger']) == 4


def test_structural_coverage_counts_actual_learned_functions(tmp_path):
    from recheck_live_trajectories import structural_coverage
    path, _, _ = partial_fixture(tmp_path)
    metrics = structural_coverage([path, tmp_path / 'missing.json'])
    assert metrics['episodes'] == 1
    assert metrics['learned_functions'] == {'recoverAndWrite': 1}
    assert metrics['learned_invocation_depths'] == {1: 1}
    assert metrics['learned_invocation_children'] == {0: 1}
    assert len(metrics['unreadable_manifests']) == 1


def test_offline_recovery_binds_independent_checks_and_preserves_original(tmp_path, monkeypatch):
    import recheck_live_trajectories as recovery
    root = tmp_path / 'source'
    root.mkdir()
    path, spec, detail, verification = fixture(root)
    interaction = open_goal(detail, spec, verification)
    detail['events'] = [{'kind': 'action', 'outcome': 'completed', 'tool_call_id': 'actual-tool-id'}]
    proof = {'goal': interaction['goal'], 'checks': {'draft_contents': True, 'reported_status': False}}
    packet = {'test': {'name': 'draft'}, 'result': {'passed': False}, 'execution': {
        'diagnostics': {'openGoalVerification': proof, 'natlangTeacher': {
            'programPath': interaction['program'], 'calls': [detail], 'rootCallIds': ['call-one']}}}}
    source = artifact(root, 'original.json', packet)
    pins = {'repositories': {'natlang': {'git_commit': 'a' * 40, 'dirty': False}},
            'instance': 'owned-instance', 'run_id': 'run'}
    artifact(root, 'pins.json', pins)
    catalog = {'scenario': 'draft', 'taskId': 'live-draft', 'family': 'draft', 'split': 'train',
               'sourceGroups': ['drafts']}
    artifact(root, 'catalog.jsonl', catalog)
    assessment = {'schema': 'natlang.live-reassessment/1', 'task_id': 'live-draft',
                  'original_sha256': source['sha256'], 'rationale': 'The independent state proves the draft contents.',
                  'invocations': [{'call_id': 'call-one', 'checks': ['draft_contents'], 'ordinals': [0]}]}
    artifact(root, 'assessment.json', assessment)
    monkeypatch.setattr(recovery, 'held_closure', lambda _: {})
    monkeypatch.setattr(recovery, 'held_hit', lambda *_: False)
    out = tmp_path / 'recovered'
    result = recovery.recover(root / 'original.json', root / 'pins.json', root / 'assessment.json',
                              root / 'catalog.jsonl', root / 'held.json', out)
    assert result['decisions'] == 1
    assert json.loads((out / 'original.json').read_text()) == packet
    assert live.load_episode(out / 'manifest.json')['trajectory_ir']['outcome']['status'] == 'partial'
    assessment['invocations'][0]['checks'] = ['reported_status']
    artifact(root, 'assessment.json', assessment)
    with pytest.raises(ValueError, match='failed independent'):
        recovery.recover(root / 'original.json', root / 'pins.json', root / 'assessment.json',
                         root / 'catalog.jsonl', root / 'held.json', tmp_path / 'bad-recovery')


def test_rejection_inventory_keeps_all_failed_calls_visible(tmp_path):
    import recheck_live_trajectories as recovery
    path, spec, detail, verification = fixture(tmp_path)
    packet = {'test': {'name': 'failed-task'}, 'result': {'passed': False}, 'execution': {'diagnostics': {
        'natlangTeacher': {'calls': [detail]},
        'openGoalVerification': {'goal': 'Create a draft', 'checks': {'draft_contents': True, 'final_report': False}}}}}
    original = tmp_path / 'packet.json'
    original.write_text(json.dumps(packet))
    catalog = tmp_path / 'catalog.jsonl'
    catalog.write_text(json.dumps({'scenario': 'failed-task', 'taskId': 'draft', 'split': 'train'}) + '\n')
    audit = recovery.audit_captures([original], catalog)
    assert audit['count'] == 1
    assert audit['captures'][0]['passing_checks'] == ['draft_contents']
    assert audit['captures'][0]['failed_checks'] == ['final_report']
    assert audit['observed_functions'] == {'createDraft': 1}
    assert audit['captures'][0]['status'] == 'requires-independent-reassessment'
    assert json.loads(original.read_text()) == packet


def test_declared_entry_coverage_is_separate_from_learned_behavior(tmp_path):
    from recheck_live_trajectories import catalog_coverage
    catalog = tmp_path / 'catalog.jsonl'
    catalog.write_text('\n'.join(json.dumps(row) for row in [
        {'scenario': 'draft-a', 'family': 'workspace-open-user-goal-draft', 'goalSha256': 'a', 'split': 'train'},
        {'scenario': 'draft-b', 'family': 'workspace-open-user-goal-draft', 'goalSha256': 'b', 'split': 'test'},
        {'scenario': 'missing', 'family': 'other', 'split': 'train'}]) + '\n')
    entries = tmp_path / 'entries.json'
    entries.write_text(json.dumps({'draft': {'program': 'skills/draft/reconcile.nl', 'resultType': 'ProposedRevision'}}))
    result = catalog_coverage(catalog, entries)
    assert result['declared_task_entries'] == {'skills/draft/reconcile.nl': 2}
    assert result['declared_return_types'] == {'ProposedRevision': 2}
    assert result['task_splits'] == {'train': 2, 'test': 1}
    assert result['unmapped_tasks'] == ['missing']
