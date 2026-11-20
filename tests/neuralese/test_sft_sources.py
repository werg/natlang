"""SFT source builder (scripts/neuralese_data/sft_sources.py): replay matching, near-duplicates, held closure."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/neuralese_data'))
import sft_sources as s  # noqa: E402


def turn(messages, content='', calls=()):
    return {'messages': messages, 'tools': [], 'completion_tokens': 3,
            'target': {'role': 'assistant', 'content': content, 'reasoning_content': 'think',
                       'tool_calls': [{'id': f'c{i}', 'type': 'function',
                                       'function': {'name': n, 'arguments': a}} for i, (n, a) in enumerate(calls)]}}


SYSTEM = {'role': 'system', 'content': 'old system prompt'}
USER = {'role': 'user', 'content': 'task'}


def test_replay_serves_recorded_replies_by_visible_history():
    first = turn([SYSTEM, USER], calls=[('eval', '{"code":"1"}')])
    call = {'role': 'assistant', 'content': '', 'tool_calls': [
        {'id': 'other-id', 'type': 'function', 'function': {'name': 'eval', 'arguments': '{"code":"1"}'}}]}
    second = turn([SYSTEM, USER, call, {'role': 'tool', 'content': '1'}], content='done')
    episode = s.ReplayEpisode([first, second])
    new_system = {'role': 'system', 'content': 'current system prompt'}
    reply = episode.answer({'messages': [new_system, USER]})
    assert reply['choices'][0]['message']['tool_calls'][0]['function']['arguments'] == '{"code":"1"}'
    assert reply['choices'][0]['message']['reasoning_content'] == 'think'
    # ids and the system prompt are the runtime's own; the visible history still matches exactly
    reply = episode.answer({'messages': [new_system, USER, {**call, 'tool_calls': [
        {**call['tool_calls'][0], 'id': 'replay-0'}]}, {'role': 'tool', 'content': '1'}]})
    assert reply['choices'][0]['message']['content'] == 'done'
    assert [e['kind'] for e in episode.log] == ['exact', 'exact']
    assert [e['system_same'] for e in episode.log] == [False, False]
    assert episode.answer({'messages': [new_system, USER]}) is None   # each recorded call is served once


def test_replay_marks_a_different_tool_output_as_drift():
    first = turn([SYSTEM, USER], calls=[('eval', '{}')])
    second = turn([SYSTEM, USER, {'role': 'assistant', 'content': '', 'tool_calls': []},
                   {'role': 'tool', 'content': 'recorded output'}], content='x')
    episode = s.ReplayEpisode([first, second])
    episode.answer({'messages': [SYSTEM, USER]})
    episode.answer({'messages': [SYSTEM, USER, {'role': 'assistant', 'content': '', 'tool_calls': []},
                                 {'role': 'tool', 'content': 'another output'}]})
    assert episode.log[-1]['kind'] == 'drifted' and episode.log[-1]['first_difference'] == 2


def test_replay_serves_a_call_whose_opening_message_the_runtime_reworded():
    opening = {'role': 'user', 'content': 'You are inside this call: f(x: string): string\n\nold call notes'}
    reworded = {'role': 'user', 'content': 'You are inside this call: f(x: string): string\n\nnew call notes'}
    other = {'role': 'user', 'content': 'You are inside this call: g(x: string): string\n\nnew call notes'}
    call = {'role': 'assistant', 'content': '', 'tool_calls': [
        {'id': 'a', 'type': 'function', 'function': {'name': 'eval', 'arguments': '{"code":"1"}'}}]}
    first = turn([SYSTEM, opening], calls=[('eval', '{"code":"1"}')])
    second = turn([SYSTEM, opening, call, {'role': 'tool', 'content': '1'}], calls=[('return_result', '{}')])
    episode = s.ReplayEpisode([first, second])
    assert episode.answer({'messages': [SYSTEM, other]}) is None            # another signature is another call
    assert episode.answer({'messages': [SYSTEM, reworded]}) is not None
    assert episode.answer({'messages': [SYSTEM, reworded, call, {'role': 'tool', 'content': '2'}]}) is None or \
        episode.log[-1]['kind'] == 'drifted'                                 # a different tool output is never reworded
    assert [e['kind'] for e in episode.log][:2] == ['missing', 'reworded']


def test_replay_declines_a_decision_readout_as_the_recorded_teacher_did():
    """A teacher that cannot score replies ran the tool loop; the replay answers the readout's scoring requests with no
    prompt_logprobs (decision-unsupported) and serves the recorded tool-loop turns after it."""
    first = turn([SYSTEM, USER], content='"yes"')
    episode = s.ReplayEpisode([first])
    scoring = {'messages': [SYSTEM, USER, {'role': 'user', 'content': 'Answer this call now'},
                            {'role': 'assistant', 'content': '"yes"'}], 'prompt_logprobs': 0, 'max_tokens': 1}
    reply = episode.answer(scoring)
    assert 'prompt_logprobs' not in reply and episode.served == [False]
    assert episode.answer({'messages': [SYSTEM, USER]})['choices'][0]['message']['content'] == '"yes"'
    assert [e['kind'] for e in episode.log] == ['scoring-declined', 'exact']


def test_native_state_episodes_are_recognized():
    native = {'role': 'tool', 'content': [{'type': 'text', 'text': 'x = '}, {'type': 'native_state', 'id': 'nzs1_a'}]}
    assert s.has_native_state({'turns': [turn([SYSTEM, USER, native])]})
    assert not s.has_native_state({'turns': [turn([SYSTEM, USER])]})


def test_near_duplicates_drop_repeated_actions_and_keep_distinct_ones():
    dedup = s.NearDuplicates(0.8)
    text = ' '.join(f'word{i}' for i in range(200))
    assert not dedup.seen(s.minhash(s.shingles(text)))
    assert dedup.seen(s.minhash(s.shingles(text + ' tail')))
    assert not dedup.seen(s.minhash(s.shingles(' '.join(f'other{i}' for i in range(200)))))
    # numbers do not make a template new
    assert s.shingles('step 1 of the plan here') == s.shingles('step 7 of the plan here')
    assert s.minhash(set()) is None and not dedup.seen(None)


def test_held_closure_covers_ids_groups_sources_and_earlier_sets(tmp_path):
    def held(directory, row, also=()):
        directory.mkdir()
        cases = directory / 'cases.jsonl'
        cases.write_text(json.dumps(row) + '\n')
        manifest = {'tasks': [{'id': row['id']}], 'excluded_split_groups': ['split-' + row['id']],
                    'cases': {'path': 'cases.jsonl', 'sha256': s.sha256_file(cases)}, 'also_excludes': list(also)}
        (directory / 'manifest.json').write_text(json.dumps(manifest))
        return directory / 'manifest.json'
    v2 = held(tmp_path / 'v2', {'id': 'old', 'source_groups': ['g-old'], 'source_ids': ['s-old']})
    v3 = held(tmp_path / 'v3', {'id': 'new', 'source_groups': ['g-new'], 'source_ids': ['s-new']},
              also=[{'path': str(v2), 'sha256': s.sha256_file(v2)}])
    ids, groups, sources = s.held_closure(v3)
    assert ids == {'old', 'new'}
    assert {'g-old', 'g-new', 'split-old', 'split-new'} <= groups
    assert sources == {'s-old', 's-new'}


def test_zstd_round_trip(tmp_path):
    writer = s.ZstdWriter(tmp_path / 'x.jsonl.zst')
    writer.write({'b': 1, 'a': [np.int32(2).item()]})
    writer.close()
    assert [json.loads(line) for line in s.zstd_lines(tmp_path / 'x.jsonl.zst')] == [{'b': 1, 'a': [2]}]
    assert list(json.loads(next(s.zstd_lines(tmp_path / 'x.jsonl.zst')))) == ['b', 'a']   # key order kept (templates)


def test_water_fill_gives_a_small_group_all_and_the_rest_by_weight():
    budgets = s.water_fill({'a': 10, 'b': 1000, 'c': 1000}, {'a': 1, 'b': 1, 'c': 2}, 310)
    assert budgets['a'] == 10 and round(budgets['b']) == 100 and round(budgets['c']) == 200
    assert 'd' not in s.water_fill({'d': 5}, {}, 100)


def test_capped_keeps_preferred_members_per_task_then_caps_families():
    import collections
    import random
    funnel = collections.defaultdict(collections.Counter)
    candidates = [{'task_id': f't{i // 3}', 'family': 'f' if i < 30 else f'g{i}', 'source': 'x', 'pref': i % 3}
                  for i in range(60)]
    kept = s.capped(candidates, random.Random(0), funnel, lambda task: 1, 0.5, lambda c: c['pref'])
    assert all(c['pref'] == 0 for c in kept)          # the preferred member of each task
    assert sum(c['family'] == 'f' for c in kept) <= len([c for c in candidates if c['pref'] == 0]) // 2
    assert funnel['x']['over_task_cap'] == 40


def test_round_robin_alternates_producers_so_one_teacher_never_fills_a_task():
    import random
    members = ([{'source': 'teacher', 'teacher': 'luna', 'n': i} for i in range(5)]
               + [{'source': 'teacher', 'teacher': 'nemotron', 'n': 9}] + [{'source': 'rl-own', 'n': 7}])
    order = s.round_robin(members, random.Random(0), lambda c: 0 if c['source'] == 'teacher' else 1)
    first = {(c['source'], c.get('teacher')) for c in order[:3]}
    assert first == {('teacher', 'luna'), ('teacher', 'nemotron'), ('rl-own', None)}
    assert order[2]['source'] == 'rl-own' and len(order) == 7


def test_teacher_coverage_names_manifest_successes_not_extracted(tmp_path):
    pool = tmp_path / 'teacher-gen' / 'pool-v1'
    pool.mkdir(parents=True)
    rows = [{'rollouts': str(pool / 'episodes/t000001/rollouts.jsonl'),
             'episodes': [{'sample': 0, 'outcome': 'task_success'}]},
            {'rollouts': str(pool / 'episodes/t000002/rollouts.jsonl'),
             'episodes': [{'sample': 0, 'outcome': 'semantic_mismatch'}]}]
    (pool / 'manifest.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    assert s.teacher_coverage([pool], set()) == (1, ['teacher:teacher-gen/pool-v1:t000001:0'])
    assert s.teacher_coverage([pool], {'teacher:teacher-gen/pool-v1:t000001:0'}) == (1, [])


class _Tokens:
    """A tokenizer stand-in: ids index a fixed vocabulary."""
    vocab = ['<think>', '</think>', '\n\n', '<tool_call>', 'eval', '</tool_call>', '<|im_end|>', 'plan', 'note']

    def convert_tokens_to_ids(self, token):
        return self.vocab.index(token) if token in self.vocab else None

    def decode(self, ids):
        return ''.join(self.vocab[i] for i in ids)


def test_a_teacher_reply_without_reasoning_trains_only_the_action_after_the_empty_think_block():
    renderer = s.TemplateRenderer.__new__(s.TemplateRenderer)
    renderer.tokenizer = _Tokens()
    target = [1, 2, 3, 4, 5, 6]          # </think> \n\n <tool_call> eval </tool_call> <|im_end|>
    no_reasoning = {'role': 'assistant', 'content': '', 'tool_calls': [{'function': {'name': 'eval'}}]}
    context = s.think_context(renderer, 'teacher', no_reasoning, target)
    assert context == 2                  # </think> and the whitespace after it are context
    weight = (np.arange(len(target)) >= context).astype(np.float32)
    assert weight.tolist() == [0, 0, 1, 1, 1, 1]
    # the same weights reach the loss through exp0_sft's chains (row['weight'])
    import exp0_sft
    row = {'prompt': np.array([0], np.int32), 'sample': np.array(target, np.int32), 'weight': weight,
           'invocation': 'i', 'index': 0}
    chain = exp0_sft.chains_of([{'task_id': 't', 'round': 'r', 'rows': [row]}])[0]
    assert exp0_sft.chain_weights(chain, 'cpu').tolist() == [0, 0, 1, 1, 1, 1]
    # October rows too; the student's own replies keep every token as a target
    assert s.think_context(renderer, 'october:luna-curricula', no_reasoning, target) == 2
    assert s.think_context(renderer, 'rl-own', {**no_reasoning, 'reasoning_content': 'plan'}, [7, 1, 3, 4, 5, 6]) == 0
    # except an own reply without reasoning (a decision readout the actor scored, never sampled): its empty think
    # block is context and the option trains; own_empty_think='train' opts out
    readout = {'role': 'assistant', 'content': '"ENTY"'}
    assert s.think_context(renderer, 'rl-own', readout, [1, 2, 7, 6]) == 2
    assert s.think_context(renderer, 'rl-own', readout, [1, 2, 7, 6], own_empty_think='train') == 0


def test_a_teacher_reasoning_block_is_context_by_default_and_a_target_with_train():
    renderer = s.TemplateRenderer.__new__(s.TemplateRenderer)
    renderer.tokenizer = _Tokens()
    reasoning = {'role': 'assistant', 'content': '', 'reasoning_content': 'plan',
                 'tool_calls': [{'function': {'name': 'eval'}}]}
    target = [7, 7, 1, 2, 3, 4, 5, 6]    # plan plan </think> \n\n <tool_call> eval </tool_call> <|im_end|>
    for source in ('teacher', 'october:luna-curricula', 'live'):
        assert s.think_context(renderer, source, reasoning, target) == 4      # the summary and </think> are context
        assert s.think_context(renderer, source, reasoning, target, 'train') == 0
    # train still keeps an empty think block as context
    assert s.think_context(renderer, 'teacher', {**reasoning, 'reasoning_content': ''}, [1, 2, 3, 6], 'train') == 2
    with pytest.raises(ValueError):
        s.think_context(renderer, 'teacher', reasoning, target, 'drop')


def test_text_a_teacher_writes_before_its_tool_call_is_context_and_a_final_answer_trains_whole():
    renderer = s.TemplateRenderer.__new__(s.TemplateRenderer)
    renderer.tokenizer = _Tokens()
    narrated = {'role': 'assistant', 'content': 'note', 'tool_calls': [{'function': {'name': 'eval'}}]}
    target = [1, 8, 3, 4, 5, 6]          # </think> note <tool_call> eval </tool_call> <|im_end|>
    for source in ('teacher', 'october:qwen-s72', 'live'):
        assert s.think_context(renderer, source, narrated, target) == 2          # the action trains from <tool_call>
        assert s.think_context(renderer, source, narrated, target, lead_in='train') == 1
    reasoned = {**narrated, 'reasoning_content': 'plan'}
    assert s.think_context(renderer, 'teacher', reasoned, [7, 1, 8, 3, 4, 5, 6]) == 3
    assert s.think_context(renderer, 'teacher', reasoned, [7, 1, 8, 3, 4, 5, 6], 'train') == 0
    # a final answer (no tool call) keeps its text as the target; the student's own text stays a target
    answer = {'role': 'assistant', 'content': 'note'}
    assert s.think_context(renderer, 'teacher', answer, [1, 8, 6]) == 1
    assert s.think_context(renderer, 'rl-own', {**narrated, 'reasoning_content': 'plan'}, [7, 1, 8, 3, 4, 5, 6]) == 0
    with pytest.raises(ValueError):
        s.think_context(renderer, 'teacher', narrated, target, lead_in='drop')


def test_non_student_context_is_one_rule_for_sft_rows_and_program_tune_rows():
    from natlang_neuralese.serve.targets import non_student_context

    class Lfm25(_Tokens):   # a template whose tool calls open with <|tool_call_start|> and that has no </think>
        vocab = ['note', '<|tool_call_start|>', 'eval', '<|tool_call_end|>', '<|im_end|>']
    narrated = {'role': 'assistant', 'content': 'note', 'tool_calls': [{'function': {'name': 'eval'}}]}
    assert non_student_context(Lfm25(), narrated, [0, 1, 2, 3, 4]) == 1
    assert non_student_context(Lfm25(), {'role': 'assistant', 'content': 'note'}, [0, 4]) == 0
    assert non_student_context(_Tokens(), narrated, [1, 8, 3, 4, 5, 6]) == 2


def test_backend_workaround_marks_episodes_that_met_the_text_backends_neuralese_refusal():
    refused = {'role': 'tool', 'content': 'neuralese-unsupported-backend: plain-text Neuralese<string> results ...'}
    assert s.backend_workaround([turn([SYSTEM, USER]), turn([SYSTEM, USER, refused])])
    assert not s.backend_workaround([turn([SYSTEM, USER, {'role': 'tool', 'content': 'ok'}])])
    # a child's refusal reaches the parent wrapped in the child's error text: still the refusal
    wrapped = {'role': 'tool', 'content': 'rejected\nproduce failed with "neuralese-unsupported-backend: needs a port"'}
    assert s.backend_workaround([turn([SYSTEM, USER, wrapped])])


def test_call_row_weights_a_teacher_detour_after_the_refusal_as_context_for_training_and_validation():
    class Renderer:
        tokenizer = _Tokens()

        def render(self, messages, tools, target):
            return [0], [3, 4, 5, 6]
    refused = {'role': 'tool', 'content': 'x failed: neuralese-unsupported-backend: no write port'}
    detour = {**turn([SYSTEM, USER, refused]), 'target': {'role': 'assistant', 'content': ''}, 'tools': None}
    row, _ = s.call_row(Renderer(), 'teacher', 'k', 0, detour)
    assert row['weight'].tolist() == [0, 0, 0, 0]
    row, _ = s.call_row(Renderer(), 'teacher', 'k', 0, detour, backend_workarounds='train')
    assert 'weight' not in row
    before = {**turn([SYSTEM, USER]), 'target': {'role': 'assistant', 'content': ''}, 'tools': None}
    assert 'weight' not in s.call_row(Renderer(), 'teacher', 'k', 0, before)[0]


def test_parse_aux_reads_share_task_cap_and_think_policy():
    part = s.parse_aux('operators=/x/chains.pkl,share=0.03,task_tokens=4000')
    assert part == {'name': 'operators', 'path': '/x/chains.pkl', 'share': 0.03, 'task_tokens': 4000,
                    'think': 'context', 'replaces': None}
    assert s.parse_aux('filled=/f.pkl,replaces=teacher')['replaces'] == 'teacher'
    assert s.parse_aux('live=/y.pkl')['share'] == 'all'
    assert s.parse_aux('g=/z.pkl,think=train')['think'] == 'train'
    with pytest.raises(ValueError):
        s.parse_aux('g=/z.pkl,weight=2')


def test_load_aux_masks_empty_think_drops_held_tasks_and_groups_units(tmp_path):
    import pickle

    def chain(task, rnd, sample):
        row = {'sample': np.array(sample, np.int32), 'invocation': 'i', 'index': 0}
        return {'ids': np.array([0] + sample, np.int32), 'segs': [(1, len(sample), row)], 'task': task, 'round': rnd}
    chains = [chain('a', 'r1', [1, 2, 3, 6]), chain('a', 'r1', [7, 1, 3, 6]), chain('a', 'r2', [3, 6]),
              chain('held', 'r3', [3, 6])]
    path = tmp_path / 'chains.pkl'
    path.write_bytes(pickle.dumps({'chains': chains}))
    part = s.parse_aux(f'x={path}')
    units = s.load_aux(part, _Tokens(), ({'held'}, set(), set()))
    assert part['think_masked_rows'] == 1 and part['held_dropped'] == 1
    by_round = {u['chains'][0]['round']: u for u in units}
    assert set(by_round) == {'r1', 'r2'}
    assert by_round['r1']['weighted'] == 2 + 4          # </think> \n\n masked in the first chain
    assert by_round['r2']['weighted'] == 2


def test_mix_report_gives_part_shares_and_task_concentration():
    def chain(task, part, n):
        row = {'sample': np.zeros(n, np.int32)}
        return {'ids': np.zeros(n + 10, np.int32), 'segs': [(10, n, row)], 'task': task, 'part': part}
    report = s.mix_report([chain('a', 'teacher', 30), chain('b', 'rl-own', 10), chain('a', 'ops', 60)], [], {})
    assert report['parts']['teacher']['loss_share'] == 0.3
    assert report['parts']['ops']['forward_tokens'] == 70
    assert report['loss_concentration']['max_task_tokens'] == 90
    assert report['tasks'] == 2


def test_open_goal_mix_counts_distinct_goals_and_all_weighted_targets():
    def chain(task, round_id, part, n):
        row = {'sample': np.zeros(n, np.int32)}
        return {'ids': np.zeros(n + 10, np.int32), 'segs': [(10, n, row)],
                'task': task, 'round': round_id, 'part': part}

    first = chain('one', 'attempt1', 'teacher', 60)
    second = chain('one', 'attempt2', 'teacher', 20)
    auxiliary = chain('one', 'attempt1', 'aux', 20)
    episodes = [
        {'open_goal': 'Update the notice', 'open_goal_chains': [id(first)]},
        {'open_goal': 'ＵＰＤＡＴＥ   the notice', 'open_goal_chains': [id(second)]},
    ]
    mix = s.mix_report([first, second, auxiliary], episodes, {})['open_user_goals']
    assert mix == {'distinct_goals': 1, 'trajectories': 3, 'weighted_target_share': 0.8}


@pytest.mark.parametrize(('distinct', 'weighted'), [(3, 0.9), (4, 0.79), (4, 0.8)])
def test_only_an_admitted_open_goal_mix_replaces_published_rows(tmp_path, distinct, weighted):
    path = tmp_path / 'rows.jsonl.zst'
    prior = s.ZstdWriter(path)
    prior.write({'id': 'previous-admitted-build'})
    prior.close()
    next_rows = s.ZstdWriter(path)
    next_rows.write({'id': 'attempted-build'})
    report = {'open_user_goals': {'distinct_goals': distinct, 'trajectories': 5,
                                  'weighted_target_share': weighted}}
    if distinct == 4 and weighted == 0.8:
        s.publish_sft_rows(next_rows, report, 0.8)
        assert [json.loads(row)['id'] for row in s.zstd_lines(path)] == ['attempted-build']
    else:
        with pytest.raises(SystemExit, match='Open user goal minimum not met'):
            s.publish_sft_rows(next_rows, report, 0.8)
        assert [json.loads(row)['id'] for row in s.zstd_lines(path)] == ['previous-admitted-build']
    assert not next_rows.partial.exists()


def test_vibestudio_holdout_closes_over_source_group_siblings(tmp_path):
    from live_teacher_sources import open_goal_fingerprint

    catalog = tmp_path / 'goals.jsonl'
    goal = 'Send a café update.\n"Use the new time."'
    _, digest = open_goal_fingerprint(goal)
    training = {'interaction': 'open-user-goal', 'taskId': 'vibestudio:train', 'split': 'train',
                'sourceGroups': ['train-world'], 'goalSha256': digest}
    holdout = {'interaction': 'open-user-goal', 'taskId': 'vibestudio:held', 'split': 'test',
               'sourceGroups': ['test-world', 'shared-source'], 'goalSha256': '1' * 64}

    def write_catalog():
        catalog.write_text(json.dumps(training) + '\n' + json.dumps(holdout) + '\n')

    write_catalog()
    held = set(), set(), set()
    registration = s.live_holdout_registry(catalog, held)
    assert registration == {'vibestudio:train': (frozenset({'train-world'}), digest)}
    assert s.held_hit('renamed', [], ['shared-source'], held)
    assert s.held_hit('renamed', ['vibestudio:held'], [], held)
    assert not s.held_hit('vibestudio:train', [], ['train-world'], held)

    candidate = {'task_id': training['taskId'], 'source_groups': ['train-world'], 'open_goal': goal}
    s.require_registered_open_goal(candidate, registration)
    with pytest.raises(ValueError, match='differs from the pinned training registry'):
        s.require_registered_open_goal({**candidate, 'open_goal': 'A different goal'}, registration)

    training['sourceGroups'].append('shared-source')
    write_catalog()
    with pytest.raises(ValueError, match='source group crosses splits'):
        s.live_holdout_registry(catalog, (set(), set(), set()))

    training['sourceGroups'].remove('shared-source')
    holdout['goalSha256'] = digest
    write_catalog()
    with pytest.raises(ValueError, match='prompt crosses splits'):
        s.live_holdout_registry(catalog, (set(), set(), set()))


def test_wrapper_cap_keeps_dataset_wrappers_to_their_share_of_the_selection():
    import collections
    import random
    selected = [{'source': 'teacher', 'task_id': f'w{i}', 'w': True} for i in range(50)] + \
               [{'source': 'teacher', 'task_id': f'r{i}', 'w': False} for i in range(30)]
    funnel = collections.defaultdict(collections.Counter)
    kept = s.wrapper_capped(selected, random.Random(0), funnel, lambda c: c['w'], 0.25)
    assert sum(not c['w'] for c in kept) == 30 and sum(c['w'] for c in kept) == 10
    assert funnel['teacher']['over_wrapper_share'] == 40


def test_prefer_unseen_keeps_untrained_episodes_under_every_cap_and_changes_nothing_without_a_build(tmp_path):
    import collections
    import random
    candidates = [{'task_id': f't{i // 4}', 'family': 'f' if i < 40 else f'g{i}', 'source': 'teacher',
                   'key': f'k{i}', 'w': i % 8 == 0} for i in range(80)]
    trained = frozenset(f'k{i}' for i in range(80) if i % 4 != 3)       # one untrained episode per task
    funnel = collections.defaultdict(collections.Counter)
    kept = s.capped(candidates, random.Random(0), funnel, lambda task: 1, 0.5, lambda c: 0, trained)
    assert all(c['key'] not in trained for c in kept)
    plain = s.capped(candidates, random.Random(0), collections.defaultdict(collections.Counter), lambda task: 1, 0.5,
                     lambda c: 0)
    assert plain == s.capped(candidates, random.Random(0), collections.defaultdict(collections.Counter),
                             lambda task: 1, 0.5, lambda c: 0, frozenset())
    wrappers = [c for c in candidates if c['w']]
    kept = s.wrapper_capped(wrappers + [c for c in candidates if not c['w']][:20], random.Random(0), funnel,
                            lambda c: c['w'], 0.2, frozenset(c['key'] for c in wrappers[1:]))
    assert wrappers[0] in kept                                    # the one untrained wrapper survives the share
    rows = tmp_path / 'build'
    rows.mkdir()
    out = s.ZstdWriter(rows / 'rows.jsonl.zst')
    for key, task in (('k1', 't0'), ('k1', 't0'), ('k9', 't2')):
        out.write({'episode': key, 'task_id': task})
    out.close()
    assert s.trained_in([rows]) == (frozenset({'k1', 'k9'}), frozenset({'t0', 't2'}))
    assert s.trained_in([]) == (frozenset(), frozenset())


def test_task_constructs_names_held_constructs_and_wrappers():
    neuralese = {'semantics': {'root': 'main.nl', 'files': {
        'main.nl': '---\nargs: text: string\nreturns: string\n---\nUse helper.',
        'helper.nl': '---\nargs: text: string\nreturns: Neuralese<string>\n---\nSummarize.'}}}
    names, wrapper = s.task_constructs(neuralese)
    assert 'neuralese_result' in names and 'multi_function' in names and not wrapper
    assert s.task_constructs({'no': 'semantics'}) == ((), False)


def test_neuralese_text_keeps_calls_without_an_import_and_compares_surfaces():
    plain = {'messages': [SYSTEM, USER], 'tools': [{'function': {'name': 'eval'}}]}
    imported = {'messages': [SYSTEM, USER, {'role': 'tool', 'content': [
        {'type': 'text', 'text': 'Stored local plan = '}, {'type': 'native_state', 'id': 'nzs1_x'}]}],
        'tools': plain['tools']}
    assert s.neuralese_parts(plain['messages']) == 0
    assert s.neuralese_parts(imported['messages']) == 1
    assert s.surface_key(plain) == s.surface_key(imported)
    other = {'messages': [{'role': 'system', 'content': 'new system prompt'}, USER], 'tools': plain['tools']}
    assert s.surface_key(other) != s.surface_key(plain)


def test_extract_takes_teacher_successes_in_student_form(tmp_path):
    """A JOLT+ teacher attempt's success enters as an own episode without its guidance turn and without the guidance
    policy's own calls; a student success is unchanged."""
    import types
    run, archive = tmp_path / 'run', tmp_path / 'archive'
    group = run / 'groups' / 'g000001'
    group.mkdir(parents=True)
    archive.mkdir()
    task = {'id': 't', 'family': 'f', 'split': 'train'}
    (group / 'cases.jsonl').write_text(json.dumps(task) + '\n')
    (group / 'score.json').write_text('{}')
    capture = lambda prompt, **extra: {'prompt': prompt, 'samples': [{'token_id': 5, 'logp': -0.1}], **extra}
    plain = [{'kind': 'tokens', 'token_ids': [1, 2]}]
    guided = plain + [{'kind': 'tokens', 'token_ids': [9], 'privileged': True}]
    consumer = {'messages': [{'role': 'user', 'content': 'q'}], 'target': {'role': 'assistant', 'content': 'a'},
                'invocation_id': 'root'}
    rows = [{'task_id': 't', 'sample': 0, 'outcome': 'task_success', 'reward': {'quality': 1.0},
             'turns': [{**consumer, 'sampling': capture(plain)}]},
            {'task_id': 't', 'sample': 8, 'outcome': 'task_success', 'reward': {'quality': 1.0}, 'jolt_role': 'teacher',
             'turns': [{**consumer, 'invocation_id': 'writer', 'sampling': capture(plain, guidance_policy={
                 'function': 'writeGuidance', 'sha256': 'w'})},
                       {**consumer, 'sampling': capture(guided, privileged={'mode': 'import'})}]}]
    (group / 'rollouts.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    work = tmp_path / 'work'
    s.extract(types.SimpleNamespace(work=str(work), rl=[f'{run}:{archive}'], teacher=[], out_name='episodes'))
    episodes = [json.loads(line) for line in s.zstd_lines(work / 'episodes.jsonl.zst')]
    student, teacher = episodes
    assert 'role' not in student and teacher['role'] == 'teacher' and teacher['source'] == 'rl-own'
    assert [t['invocation_id'] for t in teacher['turns']] == ['root']           # the writer's call is gone
    assert teacher['turns'][0]['exact']['prompt_ids'] == [1, 2]                  # the guidance turn is gone
    census = json.loads((work / 'episodes-extract.json').read_text())['sources']
    assert list(census.values())[0]['teacher_attempt_successes'] == 1


def test_replay_skips_keys_with_a_verdict(tmp_path, monkeypatch):
    from types import SimpleNamespace
    writer = s.ZstdWriter(tmp_path / 'episodes-teacher.jsonl.zst')
    for key in ('a', 'b', 'c'):
        writer.write({'key': key})
    writer.close()
    (tmp_path / 'replay.jsonl').write_text(json.dumps({'key': 'a', 'outcome': 'task_success'}) + '\n'
                                           + json.dumps({'outcome': 'replay_error: x'}) + '\n')
    seen = []
    monkeypatch.setattr(s, 'replay_episodes', lambda args, episodes: seen.extend(e['key'] for e in episodes))
    s.replay(SimpleNamespace(work=str(tmp_path)))
    assert seen == ['b', 'c']
