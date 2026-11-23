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
    assert ir['outcome']['graph_metrics']['invocations'] == 1
    assert ir['outcome']['graph_metrics']['max_depth'] == 0
    assert ir['outcome']['graph_metrics']['unique_function_paths']
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
        list(s.verified_episodes(tmp_path, funnel, skip_mismatched_live=False))
    skipped = collections.defaultdict(collections.Counter)
    assert list(s.verified_episodes(tmp_path, skipped)) == []
    assert skipped['teacher']['observed_live_skipped:live_source_rejected'] == 1


def write_episodes(root, name, episodes):
    writer = s.ZstdWriter(root / f'{name}.jsonl.zst')
    for episode in episodes:
        writer.write(episode)
    writer.close()


def test_stale_extraction_is_rederived_and_a_mismatch_skips_only_itself(tmp_path, capsys):
    """An extraction made before the loader derived a field (the Luna campaign's c6bbe8f82 episodes lack
    ``outcome.task_completed``) is admitted as today's derivation of the same verified bytes; an extraction whose own
    field differs is skipped with its reason and the other episodes still fill (one used to fail the whole fill)."""
    first = tmp_path / 'first'
    second = tmp_path / 'second'
    first.mkdir()
    second.mkdir()
    path_a, _, _, _ = fixture(first)
    path_b, _, _, _ = fixture(second)
    path_b.write_text(path_b.read_text() + '\n')   # another manifest file: its own episode key
    a, b = live.load_episode(path_a), live.load_episode(path_b)
    stale = copy.deepcopy(a)
    stale['call_tree'][0].pop('scope')          # a field a later loader added
    stale.pop('world')
    tampered = copy.deepcopy(b)
    tampered['task_id'] = 'someone-else'
    write_episodes(tmp_path, 'episodes-live', [stale, tampered])
    funnel = collections.defaultdict(collections.Counter)
    admitted = list(s.verified_episodes(tmp_path, funnel))
    assert [e for e, _ in admitted] == [a]
    assert admitted[0][1]['live_source'] == a['live_source']
    assert funnel['teacher']['observed_live_verified'] == 1
    assert funnel['teacher']['observed_live_rederived'] == 1
    assert funnel['teacher']['observed_live_skipped:live_episode_differs'] == 1
    assert 'task_id changed' in capsys.readouterr().err
    with pytest.raises(ValueError, match='live_episode_differs'):
        list(s.verified_episodes(tmp_path, funnel, skip_mismatched_live=False))


def test_derived_superset_paths():
    assert s.derived_superset({'a': [{'x': 1, 'y': 2}]}, {'a': [{'x': 1}]}) is None
    assert s.derived_superset({'a': [{'x': 2}]}, {'a': [{'x': 1}]}) == '/a[0]/x changed'
    assert s.derived_superset({'a': []}, {'a': [{'x': 1}]}) == '/a length 1 -> 0'
    assert s.derived_superset({}, {'a': 1}) == '/a removed'
    assert s.derived_superset({'a': True}, {'a': 1}) == '/a changed'


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
    entry = {'program': 'skills/documents/reconcileDraft.nl', 'callable': 'reconcileDraft', 'arguments': arguments,
             'argumentJson': {name: json.dumps(value, ensure_ascii=False, separators=(',', ':'))
                              for name, value in arguments.items()}}
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


@pytest.mark.parametrize('outcome', ['done', 'error'])
def test_partial_scope_can_recover_actions_without_supervising_wrong_final_status(tmp_path, outcome):
    path, spec, verification = partial_fixture(tmp_path)
    detail = json.loads((tmp_path / 'child.json').read_text())
    detail['record']['outcome'] = outcome
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


@pytest.mark.parametrize('value,serialized', [
    (1.0, '1'),
    (1e-7, '1e-7'),
    ({'name': 'demo', '10': 'ten', '2': 'two'}, '{"2":"two","10":"ten","name":"demo"}'),
    ('\ud800', '"\\ud800"'),
])
def test_entry_hashes_exact_native_json_instead_of_python_reserialization(value, serialized):
    encoded = serialized.encode('utf-8')
    entry = {'program': 'skills/documents/acceptValue.nl', 'callable': 'acceptValue',
             'arguments': {'value': value}, 'argumentJson': {'value': serialized}}
    root = {'definition': {'name': 'acceptValue', 'source': 'acceptValue.nl'},
            'program_root': 'skills/documents', 'outcome': 'done',
            'inputs': {'value': {'complete': True, 'bytes': len(encoded),
                                 'hash': hashlib.sha256(encoded).hexdigest()}}}
    interaction = {'program': entry['program'], 'entry': entry, 'goal': 'Use the exact typed argument'}
    live.verify_entry(interaction, root)
    entry['arguments']['value'] = False
    with pytest.raises(ValueError, match='argument JSON differs'):
        live.verify_entry(interaction, root)


def test_entry_rejects_boolean_number_alias_and_missing_exact_json():
    entry = {'program': 'skills/documents/acceptValue.nl', 'callable': 'acceptValue',
             'arguments': {'value': False}, 'argumentJson': {'value': '0'}}
    root = {'definition': {'name': 'acceptValue', 'source': 'acceptValue.nl'},
            'program_root': 'skills/documents', 'outcome': 'done',
            'inputs': {'value': {'complete': True, 'bytes': 1,
                                 'hash': hashlib.sha256(b'0').hexdigest()}}}
    interaction = {'program': entry['program'], 'entry': entry}
    with pytest.raises(ValueError, match='argument JSON differs'):
        live.verify_entry(interaction, root)
    del entry['argumentJson']
    with pytest.raises(ValueError, match='exact native argument JSON'):
        live.verify_entry(interaction, root)


@pytest.mark.parametrize('return_kind', ['optional', 'incidental'])
@pytest.mark.parametrize('retain_prior_action', [True, False])
@pytest.mark.parametrize('terminal_tool,terminal_args', [
    ('return_result', {'status': 'success', 'value': {'incorrectReceipt': True}}),
    ('eval', {'code': 'return {incorrectReceipt: true}', 'finish': True}),
])
def test_successful_effects_do_not_supervise_known_incorrect_optional_return(tmp_path, terminal_tool, terminal_args, retain_prior_action, monkeypatch, return_kind):
    path, spec, detail, verification = fixture(tmp_path)
    open_goal(detail, spec, verification)
    detail['events'] = [{'kind': 'action', 'outcome': 'completed', 'tool_call_id': 'actual-tool-id'}]
    terminal = copy.deepcopy(detail['record']['model_turns'][0])
    terminal['reply']['raw_calls'] = [{'id': 'terminal-id', 'type': 'function',
        'function': {'name': terminal_tool, 'arguments': json.dumps(terminal_args)}}]
    terminal['reply']['calls'] = [[terminal_tool, terminal_args]]
    terminal['reply']['raw_response']['vibestudio']['response']['content'] = [
        {'type': 'toolCall', 'id': 'terminal-id', 'name': terminal_tool, 'arguments': terminal_args}]
    detail['record']['model_turns'] = ([detail['record']['model_turns'][0]] if retain_prior_action else []) + [terminal]
    detail['events'].append({'kind': 'action', 'outcome': 'completed', 'tool_call_id': 'terminal-id'})
    verification['evidence'] = [{**artifact(tmp_path, 'outcome-readback.json', {'proof': {
        'passed': True, 'outcomeDimensions': {'effects': {'passed': True},
            'returnedValue': {'required': False, 'passed': False, 'checks': {'receiptMatches': False}}}}}),
        'kind': 'workspace'}]
    if return_kind == 'incidental':
        text = '{"incorrectReceipt":true}'
        detail['record']['definition']['source'] = 'runUserGoal.nl'
        detail['record']['output'] = {'complete': True, 'hash': hashlib.sha256(text.encode()).hexdigest(), 'bytes': len(text)}
        selector = {'definition': detail['record']['definition'], 'valueJson': text, 'sha256': detail['record']['output']['hash']}
        verification['evidence'] = [{**artifact(tmp_path, 'incidental-readback.json', {'proof': {
            'actualResult': {'incorrectReceipt': True}, 'outcomeDimensions': {'incidentalValues': {
                'passed': False, 'incorrectReturns': [selector]}}}}), 'kind': 'workspace'}]
    rewrite(tmp_path, spec, detail, verification)
    episode = live.load_episode(path)
    assert len(episode['turns']) == int(retain_prior_action)
    decisions = episode['trajectory_ir']['trajectory']
    if retain_prior_action:
        assert decisions[0]['training_admission']['approved'] is True
    assert decisions[-1]['training_admission'] == {'approved': False,
        'reason': 'application return failed independent outcome checks'}
    assert episode['trajectory_ir']['outcome']['task_completed'] is True
    # Extraction stores native IR; SFT and student reasoning fill must project the same masked targets.
    import sft_sources
    extracted = copy.deepcopy(episode)
    del extracted['turns']
    assert sft_sources._episode_view(extracted, False)['turns'] == episode['turns']
    candidate = {'verification_mode': 'observed-live', 'live_source': episode['live_source'],
                 'key': episode['key'], 'task_id': episode['task_id'], 'work': str(tmp_path)}
    assert len(sft_sources.candidate_turns(tmp_path, candidate)) == int(retain_prior_action)
    import filled_reasoning
    import exp0_sft
    import threading
    monkeypatch.setattr(sft_sources, 'call_row', lambda renderer, source, key, index, turn: (turn['target'], 0))
    monkeypatch.setattr(exp0_sft, 'chains_of', lambda episodes: episodes[0]['rows'])
    monkeypatch.setattr(sft_sources, 'mark_open_goal', lambda *args: None)
    renderer = object.__new__(filled_reasoning.Renderer)
    renderer.renderer, renderer.lock = object(), threading.Lock()
    chains, _ = renderer.chains(candidate)
    assert len(chains) == int(retain_prior_action)
    assert all('terminal-id' not in json.dumps(chain) for chain in chains)



@pytest.mark.parametrize("optional_roles", [False, True])
def test_completed_reassessment_preserves_original_failure_and_binds_all_checker_evidence(tmp_path, monkeypatch, optional_roles):
    import recheck_live_trajectories as recovery
    root = tmp_path / 'original'
    root.mkdir()
    path, spec, detail, verification = fixture(root)
    interaction = open_goal(detail, spec, verification)
    detail['events'] = [{'kind': 'action', 'outcome': 'completed', 'tool_call_id': 'actual-tool-id'}]
    original_proof = {'goal': interaction['goal'], 'checks': {'native_state': True, 'old_oracle': False}}
    packet = {'test': {'name': 'draft'}, 'result': {'passed': False}, 'execution': {'diagnostics': {
        'openGoalVerification': original_proof, 'natlangTeacher': {'programPath': interaction['program'],
            'calls': [detail], 'rootCallIds': ['call-one'], 'report': {'value': 'actual'}}}}}
    original = artifact(root, 'original.json', packet)
    artifact(root, 'pins.json', {'repositories': {'natlang': {'git_commit': 'a' * 40, 'dirty': False}},
                                'instance': 'owned-instance', 'run_id': 'run'})
    artifact(root, 'catalog.jsonl', {'scenario': 'draft', 'taskId': 'live-draft', 'family': 'draft',
                                   'split': 'train', 'sourceGroups': ['drafts']})
    checker = artifact(root, 'checker.json', {'implementation': 'independent crisp verifier'})
    evidence = artifact(root, 'native-witness.json', {'state': 'actual native state'})
    proof = {'schema': 'natlang.live-outcome-reassessment/1', 'task_id': 'live-draft',
             'original_sha256': original['sha256'], 'goal': interaction['goal'], 'entry': None,
             'actualResult': {'value': 'actual'}, 'passed': True,
             'checks': {'native_state': True, 'old_oracle': True}, 'rationale': 'Recomputed corrected outcome.',
             'checker_sources': [checker], 'evidence': [evidence]}
    if optional_roles:
        contract = {'effects': {'checks': ['native_state'], 'rationale': 'Actual native state.'},
                    'returnedValue': {'checks': ['old_oracle'], 'required': False, 'rationale': 'No final report was requested.'}}
        proof.update(checks={'native_state': True}, outcomeContract=contract, outcomeDimensions={
            'effects': {'passed': True, 'checks': {'native_state': True}},
            'returnedValue': {'required': False, 'passed': False, 'checks': {'old_oracle': False}}})
        proof['checker_sources'].append({**artifact(root, 'contract.json', contract), 'role': 'outcome-contract'})
    completed = artifact(root, 'completed.json', proof)
    assessment = {'schema': 'natlang.live-reassessment/1', 'task_id': 'live-draft',
                  'original_sha256': original['sha256'], 'rationale': proof['rationale'],
                  'completed_outcome': completed}
    artifact(root, 'assessment.json', assessment)
    monkeypatch.setattr(recovery, 'held_closure', lambda _: {})
    monkeypatch.setattr(recovery, 'held_hit', lambda *_: False)
    out = tmp_path / 'completed'
    recovery.recover(root / 'original.json', root / 'pins.json', root / 'assessment.json',
                     root / 'catalog.jsonl', root / 'held.json', out)
    episode = live.load_episode(out / 'manifest.json')
    assert episode['trajectory_ir']['outcome']['task_completed'] is True
    assert episode['trajectory_ir']['outcome']['status'] == 'passed'
    assert json.loads((out / 'original-outcome.json').read_text()) == {'passed': False}
    output_spec = json.loads((out / 'manifest.json').read_text())
    original_binding = output_spec['calls'][0]
    original_call_bytes = (out / original_binding['path']).read_bytes()
    forged_call = json.loads(original_call_bytes)
    forged_call['events'].append({'kind': 'unrelated-forged-native-observation'})
    output_spec['calls'][0] = artifact(out, original_binding['path'], forged_call)
    (out / 'manifest.json').write_text(json.dumps(output_spec))
    with pytest.raises(ValueError, match='exact original application call graph'):
        live.load_episode(out / 'manifest.json')
    (out / original_binding['path']).write_bytes(original_call_bytes)
    output_spec['calls'][0] = original_binding
    (out / 'manifest.json').write_text(json.dumps(output_spec))
    copied = json.loads((out / 'workspace.json').read_text())['proof']
    (out / copied['evidence'][0]['path']).write_text('tampered native witness')
    with pytest.raises(ValueError, match='artifact changed'):
        live.load_episode(out / 'manifest.json')
    proof['actualResult'] = {'value': 'invented'}
    assessment['completed_outcome'] = artifact(root, 'completed.json', proof)
    artifact(root, 'assessment.json', assessment)
    with pytest.raises(ValueError, match='exact original whole outcome'):
        recovery.recover(root / 'original.json', root / 'pins.json', root / 'assessment.json',
                         root / 'catalog.jsonl', root / 'held.json', tmp_path / 'forged')


def test_partial_reassessment_binds_retained_assessor_source(tmp_path):
    path, spec, verification = partial_fixture(tmp_path)
    verification['training_scope']['assessor'] = artifact(tmp_path, 'assessor-source.json', {'code': 'native byte comparison'})
    spec['verification'] = artifact(tmp_path, 'verification.json', verification)
    path.write_text(json.dumps(spec))
    assert len(live.load_episode(path)['turns']) == 1
    (tmp_path / 'assessor-source.json').write_text('changed assessor source')
    with pytest.raises(ValueError, match='artifact changed'):
        live.load_episode(path)


def test_exact_optional_return_selector_matches_only_actual_function_value():
    text='"check-declaration"'
    selector={'definition':{'name':'chooseNext','source':'owned/chooseNext.nl'},
              'valueJson':text,'sha256':hashlib.sha256(text.encode()).hexdigest()}
    proof={'actualResult':{'admission':{'decision':'check-declaration'}},
           'outcomeDimensions':{'incidentalValues':{'passed':False,'incorrectReturns':[selector]}}}
    record={'definition':selector['definition'],'output':{'complete':True,'hash':selector['sha256'],'bytes':len(text)},
            'model_turns':[{'kind':'generation'},{'kind':'score'},{'kind':'generation'}]}
    assert live.declared_return_exclusions([proof],{'actual':record})=={'actual':2}
    optional=copy.deepcopy(proof)
    optional['outcomeDimensions']['incidentalValues']['incorrectReturns'][0]['required']=False
    assert live.declared_return_exclusions([optional],{})=={}
    assert live.declared_return_exclusions([optional],{'unrelated':{**record,'definition':{'name':'chooseNext','source':'foreign.nl'}}})=={}
    for changed in ({'definition':{'name':'different','source':'owned/chooseNext.nl'}},
                    {'output':{**record['output'],'hash':'0'*64}}):
        with pytest.raises(ValueError,match='actual original invocation'):
            live.declared_return_exclusions([proof],{'forged':{**record,**changed}})
    forged=copy.deepcopy(proof);forged['outcomeDimensions']['incidentalValues']['incorrectReturns'][0]['sha256']='0'*64
    with pytest.raises(ValueError,match='identity/hash'):
        live.declared_return_exclusions([forged],{'actual':record})


def test_reassessment_roles_cannot_drop_checks_or_rewrite_optional_observations(tmp_path):
    import copy
    original = {'native_state': True, 'summary': False}
    contract = {'effects': {'checks': ['native_state'], 'rationale': 'Native state.'},
                'returnedValue': {'checks': ['summary'], 'required': False, 'rationale': 'No summary requested.'}}
    proof = {'checks': {'native_state': True}, 'outcomeContract': contract,
             'checker_sources': [{**artifact(tmp_path, 'contract.json', contract), 'role': 'outcome-contract'}],
             'outcomeDimensions': {'effects': {'passed': True, 'checks': {'native_state': True}},
                 'returnedValue': {'required': False, 'passed': False, 'checks': {'summary': False}}}}
    live.validate_reassessment_check_roles(proof, original, tmp_path)
    mutations = [lambda p: p.pop('outcomeContract'),
        lambda p: p['checker_sources'].clear(),
        lambda p: p['outcomeDimensions']['returnedValue']['checks'].update(summary=True),
        lambda p: p['outcomeDimensions']['returnedValue'].update(passed=True),
        lambda p: p['outcomeContract']['returnedValue'].update(required=True),
        lambda p: p['checks'].update(unknown=True)]
    for mutate in mutations:
        candidate = copy.deepcopy(proof)
        mutate(candidate)
        with pytest.raises(ValueError):
            live.validate_reassessment_check_roles(candidate, original, tmp_path)

@pytest.mark.parametrize('mutation', [None, 'output', 'selected', 'root', 'json'])
def test_native_bridge_selected_return_binds_actual_authored_output(tmp_path, mutation):
    path, spec, detail, verification = fixture(tmp_path)
    selected = {'question': 'A source-backed question', 'statements': []}
    value = {'results': [{'kind': 'domain', 'report': selected}], 'remaining': []}
    text = json.dumps(value, separators=(',', ':'))
    detail['record']['output'] = {'complete': True, 'bytes': len(text.encode()), 'hash': hashlib.sha256(text.encode()).hexdigest()}
    binding = {'root_call_id': 'call-one', 'valueJson': text, 'selectedResult': selected}
    if mutation == 'output': detail['record']['output']['hash'] = '0' * 64
    if mutation == 'selected': binding['selectedResult'] = {'question': 'forged'}
    if mutation == 'root': binding['root_call_id'] = 'foreign'
    if mutation == 'json': binding['valueJson'] = '{'
    verification['evidence'] = [{**artifact(tmp_path, 'bound-return.json', {'applicationReturn': binding}), 'kind': 'workspace'}]
    rewrite(tmp_path, spec, detail, verification)
    if mutation:
        with pytest.raises(ValueError, match='application return|domain return'):
            live.load_episode(path)
    else:
        assert len(live.load_episode(path)['turns']) == 1

@pytest.mark.parametrize('mutation',[None,'accepted','input','missing','empty-input','empty-exchange','missing-reply'])
def test_original_review_rejection_is_bound_separately_from_native_success(tmp_path,mutation):
    observed={'actual':{'documents':[{'path':'note.md','text':'The checked draft.'}]},'expected':{'goal':'Prepare a draft'},'rubric':'Check content.'}
    packet={'result':{'passed':True},'execution':{'diagnostics':{'openGoalVerification':{'preparedContentProject':{'artifact':{'semanticReviewInput':observed}}}}}}
    review={'schema':'natlang.live-workflow-review/1','input':copy.deepcopy(observed),'verdict':{'accepted':False,'verdict':'Content is correct; old handle was mistaken for a new commit.'},'exchange':{'request':{},'reply':{}}}
    if mutation=='empty-exchange':review['exchange']={}
    if mutation=='missing-reply':review['exchange']={'request':{}}
    if mutation=='accepted':review['verdict']['accepted']=True
    if mutation=='input':review['input']['expected']['goal']='Different goal'
    if mutation=='empty-input':
        packet['execution']['diagnostics']['openGoalVerification']['preparedContentProject']['artifact']['semanticReviewInput']=None
        review['input']=None
    proof={'evidence':[{**artifact(tmp_path,'original-review.json',review),'role':'original-rejected-review'}]}
    if mutation=='missing':proof['evidence']=[]
    assert live.verify_original_rejection(packet,proof,tmp_path) is (mutation is None)

@pytest.mark.parametrize('mutation',[None,'hash','call','ordinal','score','definition','duplicate'])
def test_exact_generation_quality_review_is_conservative_and_source_bound(tmp_path,mutation):
    _,spec,detail,_=fixture(tmp_path)
    item={'call_id':detail['record']['call_id'],'ordinal':0,'definition':copy.deepcopy(detail['record']['definition']),
          'call_sha256':spec['calls'][0]['sha256'],'reason':'The exact retained decision uses an unsupported premise.'}
    if mutation=='hash':item['call_sha256']='0'*64
    if mutation=='call':item['call_id']='absent'
    if mutation=='ordinal':item['ordinal']=99
    if mutation=='score':item['ordinal']=1
    if mutation=='definition':item['definition']['name']='foreign'
    proof={'decisionReview':{'schema':'natlang.live-decision-review/1','excluded':[item,item] if mutation=='duplicate' else [item]}}
    if mutation is None:
        assert live.declared_decision_exclusions([proof],[(spec['calls'][0],detail)])=={('call-one',0):item}
    else:
        with pytest.raises(ValueError,match='decision review'):
            live.declared_decision_exclusions([proof],[(spec['calls'][0],detail)])


def test_void_application_return_requires_exact_completed_typed_contract():
    import copy
    import hashlib
    from live_teacher_sources import verify_application_return
    binding = {'kind': 'void', 'root_call_id': 'root', 'valueJson': 'null', 'selectedResult': None}
    entry = {'program': 'skills/reference/review.ts', 'resultType': 'void'}
    record = {'outcome': 'done', 'executor': {'kind': 'crisp'}, 'definition': {'returns': 'void'},
              'output': {'complete': True, 'bytes': 4, 'hash': hashlib.sha256(b'null').hexdigest()}}
    verify_application_return(binding, {'root': record}, ['root'], entry)
    for section, key, value in [('entry', 'resultType', 'string'), ('entry', 'program', 'review.nl'),
                                ('record', 'outcome', 'error'), ('binding', 'valueJson', ' null'),
                                ('binding', 'selectedResult', 'invented'), ('binding', 'kind', 'unknown')]:
        changed_binding, changed_entry, changed_record = map(copy.deepcopy, [binding, entry, record])
        {'binding': changed_binding, 'entry': changed_entry, 'record': changed_record}[section][key] = value
        with pytest.raises(ValueError):
            verify_application_return(changed_binding, {'root': changed_record}, ['root'], changed_entry)
    for field, value in [('definition', {'returns': 'string'}), ('executor', {'kind': 'agent'}),
                         ('output', {**record['output'], 'hash': '0'*64})]:
        with pytest.raises(ValueError):
            verify_application_return(binding, {'root': {**record, field: value}}, ['root'], entry)
    with pytest.raises(ValueError):
        verify_application_return({key: value for key, value in binding.items() if key != 'selectedResult'}, {'root': record}, ['root'], entry)


def test_ts_application_entry_binds_actual_default_export():
    text = '"reference"'
    entry = {'program': 'skills/reference/review.ts', 'callable': 'review', 'arguments': {'goal': 'reference'}, 'argumentJson': {'goal': text}}
    root = {'program_root': 'skills/reference', 'outcome': 'done', 'executor': {'kind': 'crisp'},
            'definition': {'name': 'review', 'source': 'review.ts', 'id': 'ts:review.ts#default'},
            'inputs': {'goal': {'complete': True, 'bytes': len(text), 'hash': hashlib.sha256(text.encode()).hexdigest()}}}
    interaction = {'program': entry['program'], 'entry': entry}
    live.verify_entry(interaction, root)
    for definition in [{**root['definition'], 'id': 'ts:review.ts#other'}, {**root['definition'], 'id': None}]:
        with pytest.raises(ValueError, match='default export'):
            live.verify_entry(interaction, {**root, 'definition': definition})
    with pytest.raises(ValueError, match='default export'):
        live.verify_entry(interaction, {**root, 'executor': {'kind': 'agent'}})


def contribution_fixture(kind='projection'):
    producer_value = {'answer': 'wrong', 'internal': 'actual producer state'}
    actual = producer_value if kind == 'forwarded' else {'answer': 'wrong'}
    def encoded(value):
        text = json.dumps(value, ensure_ascii=False, separators=(',', ':'))
        return text, hashlib.sha256(text.encode()).hexdigest()
    def record(call_id, parent, source, value, executor):
        text, digest = encoded(value)
        return {'call_id': call_id, 'parent_call_id': parent, 'definition': {'name': call_id, 'source': source},
                'executor': {'kind': executor, 'model_id': 'actual-luna' if executor == 'agent' else None},
                'cost': {'model_requests': 2 if executor == 'agent' else 0}, 'outcome': 'done',
                'output': {'complete': True, 'hash': digest, 'bytes': len(text.encode())},
                'model_turns': [{'kind': 'generation', 'reply': {}}, {'kind': 'score'},
                                {'kind': 'generation', 'reply': {}}] if executor == 'agent' else []}
    records = {'root': record('root', None, 'compose.ts', actual, 'crisp'),
               'producer': record('producer', 'root', 'domain/judge.nl', producer_value, 'agent')}
    text, digest = encoded(producer_value)
    root_text, root_digest = encoded(actual)
    selector = {'definition': records['producer']['definition'].copy(), 'valueJson': text, 'sha256': digest,
                'required': False, 'contribution': {'schema': 'natlang.application-return-contribution/1',
                'kind': kind, 'root_call_id': 'root', 'producer_call_id': 'producer',
                'root_value_json': root_text, 'root_sha256': root_digest,
                'root_path': [] if kind == 'forwarded' else ['answer'],
                'producer_path': [] if kind == 'forwarded' else ['answer'],
                'checks': ['answerCorrect'], 'lineage': ['root', 'producer']}}
    if kind == 'projection':
        selector['contribution'].update(producer_source_suffix='domain/judge.nl', candidates=[
            {'call_id': 'producer', 'value_json': text, 'sha256': digest}])
    proof = {'actualResult': actual, 'outcomeDimensions': {'returnedValue': {'checks': {'answerCorrect': False}},
             'incidentalValues': {'passed': False, 'checks': {}, 'incorrectReturns': [selector]}}}
    return proof, records, selector


def test_projected_optional_return_binds_only_the_actual_sampled_producer():
    proof, records, selector = contribution_fixture()
    # Full producer state is deliberately absent from the public value; the authored projection is required.
    assert live.declared_return_exclusions([proof], records) == {'producer': 2}
    unrelated = copy.deepcopy(records['producer'])
    unrelated['call_id'] = 'unrelated'
    unrelated['definition']['source'] = 'other/judge.nl'
    records['unrelated'] = unrelated
    assert live.declared_return_exclusions([proof], records) == {'producer': 2}
    legacy = copy.deepcopy(proof)
    del legacy['outcomeDimensions']['incidentalValues']['incorrectReturns'][0]['contribution']
    with pytest.raises(ValueError, match='outside the actual application result'):
        live.declared_return_exclusions([legacy], records)


@pytest.mark.parametrize(('target', 'path', 'value'), [
    ('selector', ['contribution', 'root_call_id'], 'foreign'),
    ('selector', ['contribution', 'producer_call_id'], 'foreign'),
    ('selector', ['contribution', 'root_sha256'], '0' * 64),
    ('selector', ['contribution', 'root_path'], ['absent']),
    ('selector', ['contribution', 'producer_path'], ['internal']),
    ('selector', ['contribution', 'checks'], ['nativeWriteCorrect']),
    ('selector', ['contribution', 'lineage'], ['root', 'foreign', 'producer']),
    ('records', ['root', 'executor', 'kind'], 'agent'),
    ('records', ['producer', 'parent_call_id'], 'foreign'),
    ('records', ['producer', 'executor', 'kind'], 'crisp'),
    ('records', ['producer', 'model_turns'], []),
    ('records', ['producer', 'definition', 'source'], 'foreign.nl'),
    ('records', ['producer', 'output', 'hash'], '0' * 64),
])
def test_projected_return_custody_rejects_wrong_paths_sources_lineage_and_unsampled_calls(target, path, value):
    proof, records, selector = contribution_fixture()
    current = selector if target == 'selector' else records
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = value
    with pytest.raises(ValueError):
        live.declared_return_exclusions([proof], records)


def test_forwarded_return_requires_the_sole_child_whole_output_identity():
    proof, records, selector = contribution_fixture('forwarded')
    assert live.declared_return_exclusions([proof], records) == {'producer': 2}
    records['other'] = {'call_id': 'other', 'parent_call_id': 'root'}
    with pytest.raises(ValueError, match='sole child'):
        live.declared_return_exclusions([proof], records)
    proof, records, selector = contribution_fixture()
    selector['contribution']['kind'] = 'forwarded'
    with pytest.raises(ValueError):
        live.declared_return_exclusions([proof], records)


def test_root_contribution_keeps_actual_root_semantics_without_inventing_sampled_targets():
    proof, records, selector = contribution_fixture()
    c = selector['contribution']
    c.update(kind='root', producer_call_id='root', root_path=[], producer_path=[], lineage=['root'])
    selector.update(definition=records['root']['definition'], valueJson=c['root_value_json'], sha256=c['root_sha256'])
    assert live.declared_return_exclusions([proof], records) == {}
    c['producer_path'] = ['answer']
    with pytest.raises(ValueError):
        live.declared_return_exclusions([proof], records)


def test_projected_return_requires_complete_actual_source_candidate_inventory():
    proof, records, selector = contribution_fixture()
    extra = copy.deepcopy(records['producer'])
    extra['call_id'] = 'second'
    records['second'] = extra
    with pytest.raises(ValueError, match='candidate inventory'):
        live.declared_return_exclusions([proof], records)
    selector['contribution']['candidates'].append({**selector['contribution']['candidates'][0], 'call_id': 'second'})
    with pytest.raises(ValueError, match='unique actual producer'):
        live.declared_return_exclusions([proof], records)


@pytest.mark.parametrize('missing_path', [False, True])
def test_uncertain_optional_producer_holds_every_possible_final_generation(missing_path):
    proof, records, selector = contribution_fixture()
    records['second'] = {**copy.deepcopy(records['producer']), 'call_id': 'second'}
    c = selector['contribution']
    c['kind'] = 'uncertain-projection'
    c['uncertainty'] = 'no-verified-actual-producer' if missing_path else 'ambiguous-actual-producer'
    if missing_path:
        c['root_path'] = ['missing']
    c['candidates'].append({**c['candidates'][0], 'call_id': 'second'})
    second = copy.deepcopy(selector)
    second['contribution'].update(producer_call_id='second', lineage=['root', 'second'])
    proof['outcomeDimensions']['incidentalValues']['incorrectReturns'].append(second)
    assert live.declared_return_exclusions([proof], records) == {'producer': 2, 'second': 2}
    proof['outcomeDimensions']['incidentalValues']['incorrectReturns'].pop()
    with pytest.raises(ValueError, match='every possible actual producer'):
        live.declared_return_exclusions([proof], records)


def test_optional_wrong_return_masks_do_not_depend_on_a_false_or_even_declared_incidental_role():
    proof, records, selector = contribution_fixture()
    proof['outcomeDimensions']['incidentalValues']['passed'] = True
    proof['outcomeDimensions']['returnedValue']['passed'] = False
    assert live.declared_return_exclusions([proof], records) == {'producer': 2}
    proof['outcomeDimensions']['returnedValue']['incorrectReturns'] = [selector]
    del proof['outcomeDimensions']['incidentalValues']
    assert live.declared_return_exclusions([proof], records) == {'producer': 2}


def test_independently_checked_unused_child_preserves_the_correct_public_answer():
    proof, records, selector = contribution_fixture()
    actual = {'answer': 'correct'}
    text = json.dumps(actual, separators=(',', ':'))
    root_hash = hashlib.sha256(text.encode()).hexdigest()
    records['root']['output'] = {'complete': True, 'hash': root_hash, 'bytes': len(text)}
    proof['actualResult'] = actual
    proof['outcomeDimensions']['returnedValue']['checks']['answerCorrect'] = True
    proof['outcomeDimensions']['incidentalValues']['checks'] = {'unusedClassification': False}
    c = selector['contribution']
    c.update(kind='incidental-child', root_value_json=text, root_sha256=root_hash,
             root_path=[], producer_path=[], checks=['unusedClassification'])
    del c['candidates'], c['producer_source_suffix']
    # Incidental child quality is independently checked; it claims no projection or crisp forwarding.
    records['parent'] = {**copy.deepcopy(records['producer']), 'call_id': 'parent', 'parent_call_id': 'root'}
    records['producer']['parent_call_id'] = 'parent'
    c['lineage'] = ['root', 'parent', 'producer']
    assert live.declared_return_exclusions([proof], records) == {'producer': 2}
    c['checks'] = ['answerCorrect']
    with pytest.raises(ValueError):
        live.declared_return_exclusions([proof], records)
