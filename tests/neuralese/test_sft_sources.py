"""SFT source builder (scripts/neuralese_data/sft_sources.py): replay matching, near-duplicates, held closure."""
import collections
import random
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


def test_open_goal_depth_report_uses_effective_chain_weights():
    from exp0_sft import open_goal_depth_report

    child = {'open_goal_fingerprint': 'goal-a', 'open_goal_depth': 2,
             'segs': [(0, 3, {'weight': np.array([1., 0., .5])})]}
    root = {**child, 'open_goal_depth': 0, 'segs': [(0, 2, {})]}
    masked = {**child, 'open_goal_depth': 3, 'segs': [(0, 1, {'weight': np.array([0.])})]}
    legacy = {'open_goal_fingerprint': 'goal-b', 'segs': [(0, 4, {})]}
    assert open_goal_depth_report([child, child, root, masked, legacy, {'segs': [(0, 8, {})]}]) == {
        '0': {'distinct_goals': 1, 'weighted_target_tokens': 2},
        '2': {'distinct_goals': 1, 'weighted_target_tokens': 3},
        'unknown': {'distinct_goals': 1, 'weighted_target_tokens': 4},
    }


def test_live_capture_root_discovery_keeps_verified_manifests_in_order(tmp_path):
    single = tmp_path / 'single' / 'verified' / 'manifest.json'
    first = tmp_path / 'batch' / 'cases' / 'first' / 'verified' / 'manifest.json'
    second = tmp_path / 'batch' / 'cases' / 'second' / 'verified' / 'manifest.json'
    for path in (single, second, first):
        path.parent.mkdir(parents=True)
        path.touch()
    (tmp_path / 'batch' / 'cases' / 'failed').mkdir()

    assert s.live_manifest_paths([single], [tmp_path / 'single', tmp_path / 'batch']) == [
        str(single), str(first), str(second),
    ]


def test_live_capture_root_requires_contained_verified_manifests(tmp_path):
    capture = tmp_path / 'batch'
    capture.mkdir()
    with pytest.raises(ValueError, match='No verified live teacher manifests'):
        s.live_manifest_paths([], [capture])
    assert s.live_manifest_paths([], [capture], allow_empty=True) == []
    with pytest.raises(ValueError, match='No verified live teacher manifests'):
        s.live_manifest_paths([], [tmp_path / 'missing'], allow_empty=True)

    outside = tmp_path / 'unrelated'
    outside.mkdir()
    (outside / 'manifest.json').touch()
    (capture / 'verified').symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match='escapes the capture directory'):
        s.live_manifest_paths([], [capture])


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
    # a different tool output after a reworded opening is drift (the same invocation by its signature), never reworded
    assert episode.answer({'messages': [SYSTEM, reworded, call, {'role': 'tool', 'content': '2'}]}) is not None
    assert [e['kind'] for e in episode.log] == ['missing', 'reworded', 'drifted']


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


def test_held_closure_covers_ids_groups_sources_and_earlier_sets(tmp_path, monkeypatch):
    from natlang_neuralese.rl import held_eval
    monkeypatch.setattr(held_eval, 'held_manifests', lambda repo=None: [])   # no other registry held set

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
    # </think>, the whitespace after it and the token predicted there (serve.targets.empty_think_first_end) are context
    assert context == 3
    weight = (np.arange(len(target)) >= context).astype(np.float32)
    assert weight.tolist() == [0, 0, 0, 1, 1, 1]
    # the same weights reach the loss through exp0_sft's chains (row['weight'])
    import exp0_sft
    row = {'prompt': np.array([0], np.int32), 'sample': np.array(target, np.int32), 'weight': weight,
           'invocation': 'i', 'index': 0}
    chain = exp0_sft.chains_of([{'task_id': 't', 'round': 'r', 'rows': [row]}])[0]
    assert exp0_sft.chain_weights(chain, 'cpu').tolist() == [0, 0, 0, 1, 1, 1]
    # October rows too; the student's own replies keep every token as a target
    assert s.think_context(renderer, 'october:luna-curricula', no_reasoning, target) == 3
    assert s.think_context(renderer, 'rl-own', {**no_reasoning, 'reasoning_content': 'plan'}, [7, 1, 3, 4, 5, 6]) == 0
    # except an own reply without reasoning (a decision readout the actor scored, never sampled): its empty think
    # block is context and the option trains; own_empty_think='train' opts out
    readout = {'role': 'assistant', 'content': '"ENTY"'}
    assert s.think_context(renderer, 'rl-own', readout, [1, 2, 7, 6]) == 3
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
    assert s.think_context(renderer, 'teacher', {**reasoning, 'reasoning_content': ''}, [1, 2, 3, 6], 'train') == 3
    with pytest.raises(ValueError):
        s.think_context(renderer, 'teacher', reasoning, target, 'drop')


def test_a_weighted_teacher_think_block_trains_at_its_weight_and_its_lead_in_stays_context():
    renderer = s.TemplateRenderer.__new__(s.TemplateRenderer)
    renderer.tokenizer = _Tokens()
    reasoning = {'role': 'assistant', 'content': '', 'reasoning_content': 'plan',
                 'tool_calls': [{'function': {'name': 'eval'}}]}
    target = [7, 7, 1, 2, 3, 4, 5, 6]    # plan plan </think> \n\n <tool_call> eval </tool_call> <|im_end|>
    lead = s.think_context(renderer, 'teacher', reasoning, target, 'train@0.1')
    assert lead == 4 and isinstance(lead, s.WeightedLead) and lead.think_end == 4
    row, _ = s.rendered_row((np.array([0], np.int32), np.array(target, np.int32), lead), 'teacher', 'k', 0,
                            {'target': reasoning})
    assert row['weight'].tolist() == pytest.approx([0.1] * 4 + [1] * 4)
    # the text before the tool call stays zero weight; only the think block carries W
    narrated = {**reasoning, 'content': 'note'}
    narrated_target = [7, 1, 8, 3, 4, 5, 6]
    lead = s.think_context(renderer, 'teacher', narrated, narrated_target, 'train@0.25')
    row, _ = s.rendered_row((np.array([0], np.int32), np.array(narrated_target, np.int32), lead), 'teacher', 'k', 0,
                            {'target': narrated})
    assert row['weight'].tolist() == pytest.approx([0.25, 0.25, 0, 1, 1, 1, 1])
    # an empty block stays context; W 0 is context; W 1 is the full-weight train rule; own replies are untouched
    empty = s.think_context(renderer, 'teacher', {**reasoning, 'reasoning_content': ' '}, [1, 2, 3, 6], 'train@0.1')
    assert empty == 3 and not isinstance(empty, s.WeightedLead)
    assert s.think_context(renderer, 'teacher', reasoning, target, 'train@0') == 4
    assert s.think_context(renderer, 'teacher', reasoning, target, 'train@1') == 0
    assert s.think_context(renderer, 'rl-own', reasoning, target, 'train@0.1') == 0
    # the lead survives the worker-process boundary (RenderAhead pickles render results)
    import pickle
    again = pickle.loads(pickle.dumps(s.WeightedLead(4, 3, 0.1)))
    assert (int(again), again.think_end, again.think_weight) == (4, 3, 0.1)
    for bad in ('train@2', 'context@0.1', 'train@x', 'drop'):
        with pytest.raises(ValueError):
            s.think_context(renderer, 'teacher', reasoning, target, bad)


def test_text_a_teacher_writes_before_its_tool_call_is_context_and_a_final_answer_trains_whole():
    renderer = s.TemplateRenderer.__new__(s.TemplateRenderer)
    renderer.tokenizer = _Tokens()
    narrated = {'role': 'assistant', 'content': 'note', 'tool_calls': [{'function': {'name': 'eval'}}]}
    target = [1, 8, 3, 4, 5, 6]          # </think> note <tool_call> eval </tool_call> <|im_end|>
    for source in ('teacher', 'october:qwen-s72', 'live'):
        assert s.think_context(renderer, source, narrated, target) == 2          # the action trains from <tool_call>
        assert s.think_context(renderer, source, narrated, target, lead_in='train') == 2
    reasoned = {**narrated, 'reasoning_content': 'plan'}
    assert s.think_context(renderer, 'teacher', reasoned, [7, 1, 8, 3, 4, 5, 6]) == 3
    assert s.think_context(renderer, 'teacher', reasoned, [7, 1, 8, 3, 4, 5, 6], 'train') == 0
    # a final answer (no tool call) keeps its text as the target after the token predicted at an empty block's
    # </think> (serve.targets.empty_think_first_end); the student's own text stays a target
    answer = {'role': 'assistant', 'content': 'note'}
    assert s.think_context(renderer, 'teacher', answer, [1, 8, 8, 6]) == 2
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


def test_calls_the_detour_spawned_after_a_refusal_are_context_by_default():
    """A child invocation that first appears after the episode's first Neuralese refusal exists only because of the
    detour: its calls are context though their own prompts hold no refusal; an invocation already running keeps its
    calls; ``context-refused-only`` opts out."""
    class Renderer:
        tokenizer = _Tokens()

        def render(self, messages, tools, target):
            return [0], [3, 4, 5]
    refused = {'role': 'tool', 'content': 'x failed: neuralese-unsupported-backend: no write port'}

    def call(invocation, messages):
        return {**turn(messages), 'invocation_id': invocation, 'target': {'role': 'assistant', 'content': ''},
                'tools': None}
    turns = s.mark_detour_spawned([call('root', [SYSTEM, USER]), call('producer', [SYSTEM, USER]),
                                   call('root', [SYSTEM, USER, refused]), call('nl@eval:1', [SYSTEM, USER]),
                                   call('producer', [SYSTEM, USER])])
    assert [bool(t.get('detour_spawned')) for t in turns] == [False, False, False, True, False]
    weights = [s.call_row(Renderer(), 'teacher', 'k', i, t)[0].get('weight') for i, t in enumerate(turns)]
    assert [w is not None and not w.any() for w in weights] == [False, False, True, True, False]
    opt_out = s.call_row(Renderer(), 'teacher', 'k', 3, turns[3], backend_workarounds='context-refused-only')[0]
    assert 'weight' not in opt_out
    assert 'weight' not in s.call_row(Renderer(), 'rl-own', 'k', 3, turns[3])[0]   # own calls are never detours


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
    assert by_round['r1']['weighted'] == 1 + 4          # </think> \n\n and the token after them masked in the first chain
    assert by_round['r2']['weighted'] == 2


def test_mix_report_gives_part_shares_and_task_concentration():
    def chain(task, part, n):
        row = {'sample': np.zeros(n, np.int32)}
        return {'ids': np.zeros(n + 10, np.int32), 'segs': [(10, n, row)], 'task': task, 'part': part}
    report = s.mix_report([chain('a', 'teacher', 30), chain('b', 'rl-own', 10), chain('a', 'ops', 60)], {})
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
    first['open_goal_fingerprint'] = s.open_goal_key('Update the notice')
    second['open_goal_fingerprint'] = s.open_goal_key('ＵＰＤＡＴＥ   the notice')
    mix = s.mix_report([first, second, auxiliary], {})['open_user_goals']
    assert mix == {'distinct_goals': 1, 'trajectories': 3, 'weighted_target_share': 0.8, 'scope': 'all'}


def test_open_goal_share_of_a_mixed_build_is_measured_over_the_vibestudio_chains(tmp_path):
    """A mixed expert-iteration build (open goals beside every other part) holds its Vibestudio data to the open-goal
    contract: the share counts Vibestudio tasks' chains (open goals against historical assignments), not the rest; the
    bundle carries the scope and the learner's planned-schedule check measures the same chains."""
    import exp0_sft

    def chain(task, round_id, n, goal=None):
        row = {'sample': np.zeros(n, np.int32)}
        return {'ids': np.zeros(n + 10, np.int32), 'segs': [(10, n, row)], 'task': task, 'round': round_id,
                'part': 'teacher', **({'open_goal_fingerprint': goal} if goal else {})}
    goals = [chain(f'vibestudio:edit:g{i}', f'live:{i}', 50, f'goal{i}') for i in range(8)]
    legacy = chain('vibestudio:unit-repair:assigned', 'live:assigned', 40)
    other = [chain(f'task{i}', f'teacher:{i}', 500) for i in range(20)]
    scoped = s.mix_report([*goals, legacy, *other], {}, 'vibestudio')['open_user_goals']
    assert scoped['scope'] == 'vibestudio' and scoped['trajectories'] == 9 and scoped['distinct_goals'] == 8
    assert exp0_sft.check_open_goal_exposure(dict(scoped), 0.8)['weighted_target_share'] == 400 / 440
    with pytest.raises(SystemExit, match='Open user goal minimum not met'):
        exp0_sft.check_open_goal_exposure(s.mix_report([*goals, legacy, *other], {}, 'all')['open_user_goals'], 0.8)
    assert exp0_sft.open_goal_exposure([*goals, legacy, *other], 'vibestudio')['distinct_goals'] == 8
    assert not exp0_sft.in_open_goal_scope(other[0], 'vibestudio') and exp0_sft.in_open_goal_scope(other[0], None)


def test_open_goal_marks_follow_the_live_call_ancestry():
    rows = [{'invocation': 'root'}, {'invocation': 'child'}]
    chains = [{'segs': [(0, 1, rows[0])]}, {'segs': [(0, 1, rows[1])]}]
    turns = [{'invocation_id': 'root', 'live_exchange': {'call_ancestry': ['root']}},
             {'invocation_id': 'child', 'live_exchange': {'call_ancestry': ['root', 'child']}}]
    s.mark_open_goal(chains, 'Update  the Notice', turns)
    assert {c['open_goal_fingerprint'] for c in chains} == {
        __import__('hashlib').sha256(s.open_goal_key('update the notice').encode()).hexdigest()}
    assert [c['open_goal_depth'] for c in chains] == [0, 1]
    bare = [{'segs': [(0, 1, rows[0])]}]
    s.mark_open_goal(bare, None, turns)
    assert 'open_goal_fingerprint' not in bare[0]


def test_optimizer_exposure_rejects_a_post_mask_or_partial_schedule_below_the_floor(tmp_path):
    import pickle
    from exp0_sft import check_open_goal_exposure, load_chain_bundle, open_goal_exposure

    def chain(index):
        row = {'sample': np.zeros(12, np.int32)}
        return {'part': 'teacher', 'task': str(index), 'round': 'one', 'segs': [(8, 12, row)],
                **({'open_goal_fingerprint': str(index)} if index < 4 else {})}

    path = tmp_path / 'chains.pkl'
    path.write_bytes(pickle.dumps({'schema': 'natlang.sft-chains/1', 'minimum_open_goal_share': 0.8,
                                  'chains': [chain(index) for index in range(5)]}))
    bundle = load_chain_bundle(path)
    chains, minimum = bundle['chains'], bundle['minimum_open_goal_share']
    assert check_open_goal_exposure(open_goal_exposure(chains), minimum)['weighted_target_share'] == 0.8
    with pytest.raises(SystemExit, match='Open user goal minimum not met'):
        check_open_goal_exposure(open_goal_exposure(chains[index] for index in [0, 1, 2, 4]), minimum)
    chains[2]['segs'][0][-1]['weight'] = np.ones(12, np.float32) * 0.5
    with pytest.raises(SystemExit, match='Open user goal minimum not met'):
        check_open_goal_exposure(open_goal_exposure(chains), minimum)


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


def test_prefer_unseen_family_scope_keeps_family_turns_and_global_scope_hands_them_to_fresh_families():
    small = [{'task_id': f'a{i}', 'family': 'A', 'key': f'a{i}'} for i in range(5)]           # all trained
    mixed = [{'task_id': f'c{i}', 'family': 'C', 'key': f'c{i}'} for i in range(6)]           # c0-c2 trained
    large = [{'task_id': f'b{i}', 'family': 'B', 'key': f'b{i}'} for i in range(50)]          # fresh
    second = [{'task_id': 'b0', 'family': 'B', 'key': 'b0-second'}, {'task_id': 'a0', 'family': 'A', 'key': 'a0-new'}]
    pool = small + mixed + large + second
    trained = frozenset([c['key'] for c in small] + ['c0', 'c1', 'c2', 'b0'])
    key = lambda c: c['key']
    order = s.breadth_first(pool, key, trained, 'global')
    assert sum(c['family'] == 'A' for c in order[:49]) == 1       # every fresh episode first: A waits behind B's tail
    order = s.breadth_first(pool, key, trained, 'family')
    first = order[:15]
    assert sum(c['family'] == 'A' for c in first) == 5            # A keeps its turns with what the start learned
    assert first[0]['key'] == 'a0-new'                            # untrained episodes first within a task ...
    assert [c['key'] for c in order if c['family'] == 'C'][:3] == ['c3', 'c4', 'c5']   # ... and within a family's turn
    assert order.index(next(c for c in order if c['key'] == 'b0-second')) < order.index(
        next(c for c in order if c['key'] == 'b0'))
    plain = s.breadth_first(pool, key, frozenset(), 'family')
    assert plain == s.breadth_first(pool, key, frozenset(), 'global')   # nothing trained: the scopes agree
    with pytest.raises(ValueError):
        s.breadth_first(pool, key, trained, 'other')


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
    assert 'role' not in student and teacher['role'] == 'teacher' and teacher['source'] == 'jolt-guided'
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


def test_october_teacher_rows_become_replayable_teacher_episodes(tmp_path, monkeypatch):
    """October per-call rows of one model, grouped by trajectory in decision order, become `teacher` episodes with the
    task row read back from the source turns (its program_ir wins over a stale row task_id, both held-checked); held
    tasks, fixtures and trajectories without a task row are dropped."""
    import types
    monkeypatch.setattr(s.data_registry, 'require_registered', lambda paths: None)
    monkeypatch.setattr(s.data_registry, 'record_dataset', lambda *a, **k: None)
    monkeypatch.setattr(s, 'held_closure', lambda manifest: ({'held-row-id'}, set(), set()))
    monkeypatch.setattr(s, 'held_excluded_ids', lambda manifest, tasks, held: {'held-task'})
    source = tmp_path / 'turns.jsonl'
    programs = {'tr-a': {'id': 'task-a', 'family': 'f', 'split': 'train'},
                'tr-held': {'id': 'held-task', 'split': 'train'},
                'tr-fixture': {'id': 'fixture', 'split': 'train', 'kind': 'evaluation_fixture'},
                'tr-stale': {'id': 'task-s', 'split': 'train'}, 'tr-stale-held': {'id': 'task-sh', 'split': 'train'}}
    source.write_text(''.join(json.dumps({'id': f'{t}:decision:0000', 'source_ref': {'trajectory_id': t},
                                          'task': {'program_ir': p}}) + '\n' for t, p in programs.items()))

    def row(trajectory, task_id, decision, model='qwen'):
        return {'id': f'{trajectory}:decision:{decision:04d}', 'task_id': task_id, 'family': 'f',
                'messages': [{'role': 'user', 'content': f'{trajectory} {decision}'}], 'tools': [],
                'target': {'role': 'assistant', 'content': str(decision)},
                'provenance': {'model': model, 'trajectory': trajectory, 'invocation': 'i', 'source_file': str(source),
                               'target_tokens': 2}}
    rows = [row('tr-a', 'task-a', 1), row('tr-held', 'held-task', 0), row('tr-a', 'task-a', 0),
            row('tr-other', 'task-o', 0, model='luna'), row('tr-fixture', 'fixture', 0), row('tr-none', 'task-n', 0),
            row('tr-stale', 'stale-id', 0), row('tr-stale-held', 'held-row-id', 0)]
    group = tmp_path / 'qwen-s72.jsonl.zst'
    writer = s.ZstdWriter(group)
    for r in rows:
        writer.write(r)
    writer.close()
    work = tmp_path / 'work'
    s.extract(types.SimpleNamespace(work=str(work), rl=[], teacher=[], out_name='episodes-october',
                                    october_teacher=[str(group)], october_model=['qwen'], held_manifest='held'))
    episodes = [json.loads(line) for line in s.zstd_lines(work / 'episodes-october.jsonl.zst')]
    assert [e['key'] for e in episodes] == ['october-teacher:qwen-s72:tr-a', 'october-teacher:qwen-s72:tr-stale']
    assert episodes[1]['task_id'] == 'task-s'
    episode = episodes[0]
    assert episode['source'] == 'teacher' and episode['task'] == programs['tr-a']
    assert episode['teacher']['model_id'] == 'qwen' and episode['teacher']['provider'] == 'self-hosted-vllm'
    assert [t['target']['content'] for t in episode['turns']] == ['0', '1']   # decision order
    assert episode['turns'][0]['completion_tokens'] == 2
    census = json.loads((work / 'episodes-october-extract.json').read_text())['sources']['october-teacher:qwen-s72']
    assert census == {'rows': 8, 'rows_other_model': 1, 'trajectories': 6, 'episodes': 2, 'held_excluded': 2,
                      'fixture_not_admitted': 1, 'no_task_row': 1, 'task_id_from_program_ir': 2}


def test_replay_matches_tool_call_arguments_the_runtime_reserialized():
    """A teacher's spaced JSON arguments echoed back compact by the runtime are the same recorded call."""
    recorded = turn([SYSTEM, USER, {'role': 'assistant', 'content': '', 'tool_calls': [
        {'id': 'a', 'type': 'function', 'function': {'name': 'eval', 'arguments': '{"code": "1", "finish": true}'}}]},
        {'role': 'tool', 'content': 'ok'}], content='done')
    episode = s.ReplayEpisode([recorded])
    now = [SYSTEM, USER, {'role': 'assistant', 'content': '', 'tool_calls': [
        {'id': 'b', 'type': 'function', 'function': {'name': 'eval', 'arguments': '{"finish":true,"code":"1"}'}}]},
        {'role': 'tool', 'content': 'ok'}]
    assert episode.answer({'messages': now})['choices'][0]['message']['content'] == 'done'
    assert episode.log[0]['kind'] == 'exact'
    assert s.arguments_key('not json') == 'not json'


def test_extract_rl_specs_reads_every_named_extraction(tmp_path):
    """``neuralese-graph`` finds the rollout archives of every ``<out-name>-extract.json`` (and a legacy
    ``extract.json``), each spec once: it read only ``extract.json``, which no current WORK has."""
    (tmp_path / 'episodes-own-extract.json').write_text(json.dumps({'rl': ['/r/a:/x/a', '/r/b:/x/b']}))
    (tmp_path / 'episodes-sft-v7-extract.json').write_text(json.dumps({'rl': ['/r/c:/x/c', '/r/a:/x/a']}))
    (tmp_path / 'episodes-teacher-extract.json').write_text(json.dumps({'rl': [], 'teacher': ['/t']}))
    (tmp_path / 'extract.json').write_text(json.dumps({'rl': ['/r/d:/x/d']}))
    assert s.extract_rl_specs(tmp_path) == ['/r/a:/x/a', '/r/b:/x/b', '/r/c:/x/c', '/r/d:/x/d']


def _drift_work(tmp_path, episodes, records):
    writer = s.ZstdWriter(tmp_path / 'episodes.jsonl.zst')
    for e in episodes:
        writer.write(e)
    writer.close()
    (tmp_path / 'replay.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))


def test_drifted_teacher_successes_are_admitted_by_default_with_an_opt_out(tmp_path):
    """A teacher replay that served every recorded call (some with a drifted display) and succeeded is verified by
    default; an own episode keeps exact replay; a legacy record of an episode whose sibling calls it could have swapped
    is refused unless the replay resolved siblings; ``drifted=False`` restores exact-only."""
    import collections
    plain = [turn([SYSTEM, {'role': 'user', 'content': 'You are inside this call: f()\n\nbody'}])]
    siblings = [turn([SYSTEM, {'role': 'user', 'content': f'You are inside this call: nl@eval:1()\n\nitem {i}'}])
                for i in range(2)]
    episode = lambda key, source, turns: {'key': key, 'source': source, 'task_id': key, 'task': {'id': key},
                                          'turns': turns}
    drift = {'outcome': 'task_success', 'exact': False, 'unserved': 0, 'served': {'drifted': 1},
             'replayed_turns': 1, 'recorded_turns': 1}
    _drift_work(tmp_path, [episode('t', 'teacher', plain), episode('own', 'rl-own', plain),
                           episode('sib', 'teacher', siblings), episode('sib2', 'teacher', siblings),
                           episode('exact', 'teacher', plain)],
                [{**drift, 'key': 't', 'name': 'a'}, {**drift, 'key': 'own', 'name': 'b'},
                 {**drift, 'key': 'sib', 'name': 'c', 'replayed_turns': 2, 'recorded_turns': 2, 'served': {'drifted': 2}},
                 {**drift, 'key': 'sib2', 'name': 'd', 'replayed_turns': 2, 'recorded_turns': 2, 'served': {'drifted': 2},
                  'siblings_resolved': True},
                 {**drift, 'key': 'exact', 'name': 'e', 'exact': True, 'served': {'exact': 1}}])
    funnel = collections.defaultdict(collections.Counter)
    assert [e['key'] for e, _ in s.verified_episodes(tmp_path, funnel)] == ['t', 'sib2', 'exact']
    assert funnel['teacher']['replayed_drifted'] == 2 and funnel['teacher']['drifted_sibling_ambiguous'] == 1
    assert funnel['rl-own']['replay_inexact'] == 1
    off = collections.defaultdict(collections.Counter)
    assert [e['key'] for e, _ in s.verified_episodes(tmp_path, off, drifted=False)] == ['exact']


def test_replay_serves_the_sibling_call_whose_opening_is_closest():
    """Sibling child calls share a signature: a reworded request gets the recorded call of its own item, not the first
    unserved sibling."""
    recorded = [turn([SYSTEM, {'role': 'user', 'content': f'You are inside this call: nl@eval:1()\n\nitem {i}\n\nold notes'}],
                     content=f'answer {i}') for i in range(3)]
    episode = s.ReplayEpisode(recorded)
    now = [SYSTEM, {'role': 'user', 'content': 'You are inside this call: nl@eval:1()\n\nitem 2\n\nnew notes'}]
    assert episode.answer({'messages': now})['choices'][0]['message']['content'] == 'answer 2'


class AheadRenderer:
    """A renderer the spawned RenderAhead workers can build (module level): ids from the text, 'boom' fails."""
    identity = {'ahead': True}

    class tokenizer:
        @staticmethod
        def convert_tokens_to_ids(token):
            return {'</think>': 7}.get(token, -1)

        @staticmethod
        def decode(ids):
            return {7: '</think>'}.get(ids[0], 'x')

    def render(self, messages, tools, target):
        if target.get('content') == 'boom':
            raise ValueError('cannot render')
        return [len(m['content']) for m in messages], [7, 8, len(target.get('content') or '')]


def ahead_work(tmp_path, n):
    work = tmp_path / 'work'
    candidates = []
    for i in range(n):
        name = f'e{i}'
        content = 'boom' if i == 3 else 'x' * i
        if i != 5:            # e5: no replay record (its turns cannot be read)
            (work / 'replay' / name).mkdir(parents=True)
            turns = [{'messages': [{'role': 'user', 'content': 'q' * (i + k)}], 'tools': [], 'invocation_id': 'root',
                      'target': {'role': 'assistant', 'content': content}} for k in range(3)]
            (work / 'replay' / name / 'rollouts.jsonl').write_text(json.dumps({'turns': turns}) + '\n')
        candidates.append({'key': f'teacher:p:{name}', 'source': 'teacher' if i % 2 else 'rl-own', 'name': name,
                           'task_id': f't{i}'})
    return work, candidates


def test_render_ahead_gives_the_serial_results_and_byte_identical_rows(tmp_path):
    import functools
    import pickle
    work, candidates = ahead_work(tmp_path, 12)
    think = lambda c: 'context'   # noqa: E731

    def walk(workers, asked):
        ahead = s.RenderAhead(AheadRenderer(), work, think, 'context', 'context', workers=workers,
                              make_renderer=functools.partial(AheadRenderer), window=3)
        ahead.walk(candidates)
        results = {}
        for c in asked:
            results[c['key']] = ahead.get(c)
            assert ahead.get(c) is results[c['key']]       # measurement and admission share one render
        outside = {'key': 'teacher:p:e1-again', 'source': 'teacher', 'name': 'e1'}
        results['outside'] = ahead.get(outside)               # not in the walk: rendered here
        ahead.close()
        return results

    asked = [candidates[i] for i in (0, 1, 3, 5, 9, 10, 11)]  # skips inside and beyond the window
    serial, parallel = walk(1, asked), walk(2, asked)
    assert serial.keys() == parallel.keys()
    for key in serial:
        (t1, c1), (t2, c2) = serial[key], parallel[key]
        if isinstance(t1, Exception):
            assert type(t1) is type(t2) and c1 is None and c2 is None
            continue
        assert t1 == t2 and len(c1) == len(c2)
        for a, b in zip(c1, c2):
            assert (a == b) if isinstance(a, str) else (a[2] == b[2] and all(np.array_equal(x, y) for x, y in zip(a[:2], b[:2])))
    assert isinstance(serial['teacher:p:e5'][0], FileNotFoundError)
    assert all(isinstance(call, str) and 'cannot render' in call for call in serial['teacher:p:e3'][1])

    def rows(results):
        out = []
        for c in asked:
            turns, calls = results[c['key']]
            if isinstance(turns, Exception) or any(isinstance(call, str) for call in calls):
                continue
            out += [s.rendered_row(call, c['source'], c['key'], i, turn)[0] for i, (turn, call) in enumerate(zip(turns, calls))]
        return out
    # pickle memoizes dtypes by identity: rows from worker processes must pickle to the same bytes as serial ones
    assert pickle.dumps(rows(serial), protocol=4) == pickle.dumps(rows(parallel), protocol=4)
    assert s.call_row(AheadRenderer(), 'teacher', 'k', 0, serial['teacher:p:e1'][0][0])[0]['weight'].tolist() == [0, 0, 1]


def test_light_episodes_read_by_workers_equal_the_serial_reading(tmp_path):
    work = tmp_path / 'work'
    work.mkdir()
    for name, keys in (('episodes.jsonl.zst', range(0, 150)), ('episodes-more.jsonl.zst', range(100, 200))):
        writer = s.ZstdWriter(work / name)
        for i in keys:       # 100..149 twice: the first extraction's record wins
            calls = [('eval', json.dumps({'code': f'x{i % 7}'}))]
            writer.write({'key': f'k{i}', 'source': 'teacher', 'task_id': f't{i}', 'file': name,
                          'turns': [turn([SYSTEM, {'role': 'user', 'content': f'task {i % 5}\nmore'}], f'answer {i}', calls),
                                    turn([SYSTEM, {'role': 'user', 'content': f'task {i % 5}\nother'}], 'done')]})
        writer.close()
    full = list(s.all_episodes(work))
    serial, parallel = list(s.all_episodes(work, 1, light=True)), list(s.all_episodes(work, 3, light=True))
    assert [e['key'] for e in full] == [e['key'] for e in serial] == [e['key'] for e in parallel]
    assert all(e['file'] == 'episodes-more.jsonl.zst' for e in parallel if int(e['key'][1:]) >= 100)
    for f, a, b in zip(full, serial, parallel):
        assert 'turns' not in a and 'turns' not in b
        assert a.keys() == b.keys() and {k: v for k, v in a.items() if k != 'facts'} == \
            {k: v for k, v in b.items() if k != 'facts'}
        facts = s.episode_facts(f['turns'])
        for x in (a['facts'], b['facts']):
            assert np.array_equal(x['signature'], facts['signature'])
            assert x['backend_workaround'] == facts['backend_workaround']
            assert x['sibling_ambiguous'] == facts['sibling_ambiguous'] is True


def test_repeated_lines_are_skipped_unparsed_and_the_reading_is_unchanged(tmp_path, monkeypatch):
    assert s.line_key('{"key": "a\\"b", "x": 1}') == 'a"b' and s.line_key('{"key": "k"}') == 'k'
    assert s.line_key('{"source": "x", "key": "k"}') is None and s.line_key('{"key": 3, "x": 1}') is None
    work = tmp_path / 'work'
    work.mkdir()
    for name, keys in (('episodes-a.jsonl.zst', range(0, 40)), ('episodes-b.jsonl.zst', range(20, 60))):
        writer = s.ZstdWriter(work / name)
        for i in keys:       # 20..39 twice: the first extraction's record wins
            writer.write({'key': f'k{i}', 'source': 'rl-own', 'file': name, 'turns': []})
        writer.close()
    with (work / 'episodes-c.jsonl').open('w') as handle:   # key not first: parsed, deduped after parsing
        handle.write(json.dumps({'source': 'rl-own', 'key': 'k5', 'file': 'c', 'turns': []}) + '\n')
        handle.write(json.dumps({'source': 'rl-own', 'key': 'k99', 'file': 'c', 'turns': []}) + '\n')
    monkeypatch.setattr(s, 'episode_files', lambda w: sorted(Path(w).glob('episodes*.jsonl*')))
    parsed = []
    loads = json.loads
    monkeypatch.setattr(s.json, 'loads', lambda line, *a, **k: parsed.append(line) or loads(line, *a, **k))
    fast = [(e['key'], e['file']) for e in s.all_episodes(work)]
    fast_parsed, parsed[:] = len(parsed), []
    skipped = [(e['key'], e['file']) for e in s.all_episodes(work, skip={'k1', 'k25', 'k99'})]
    monkeypatch.setenv('NATLANG_EPISODE_LINE_DEDUPE', '0')
    parsed[:] = []
    slow = [(e['key'], e['file']) for e in s.all_episodes(work)]
    assert fast == slow and len(fast) == 61 and ('k25', 'episodes-a.jsonl.zst') in fast and ('k99', 'c') in fast
    assert fast_parsed == 62 and len(parsed) == 82        # repeats unparsed; the key-last lines always parsed
    assert skipped == [x for x in slow if x[0] not in {'k1', 'k25', 'k99'}]
    assert [(e['key'], e['file']) for e in s.all_episodes(work, 3)] == slow


def test_replay_surfaces_and_kept_episodes_read_by_workers_equal_the_serial_reading(tmp_path):
    work = tmp_path / 'work'
    (work / 'replay').mkdir(parents=True)
    with (work / 'replay.jsonl').open('w') as replays:
        for i in range(70):
            name = f'r{i}'
            replays.write(json.dumps({'key': f'k{i}', 'name': name, 'exact': i % 3 != 0}) + '\n')
            (work / 'replay' / name).mkdir()
            system = {'role': 'system', 'content': f'system {i % 4}'}
            (work / 'replay' / name / 'rollouts.jsonl').write_text(json.dumps(
                {'turns': [{'messages': [system, USER], 'tools': [{'function': {'name': f'f{i % 5}'}}]}]}) + '\n')
    assert s.replay_surfaces(work, 1) == s.replay_surfaces(work, 3) and len(s.replay_surfaces(work, 1)) == 20
    writer = s.ZstdWriter(work / 'episodes.jsonl.zst')
    native = [{'type': 'native_state', 'id': 'nzs1_x'}]
    for i in range(90):
        content = native if i % 4 == 0 else 'plain'
        writer.write({'key': f'k{i % 80}', 'source': 'rl-own', 'n': i,
                      'turns': [{'messages': [SYSTEM, {'role': 'tool', 'content': content}], 'tools': [],
                                 'target': {'role': 'assistant', 'content': 'a'}}]})
    writer.close()
    serial = list(s.all_episodes(work, 1, keep=s.has_native_state))
    assert serial == list(s.all_episodes(work, 3, keep=s.has_native_state))
    assert [e['n'] for e in serial] == [i for i in range(80) if i % 4 == 0]   # k0..k9 again at 80..89: first wins


def _student_candidates():
    own = [{'key': f'own{i}', 'source': 'rl-own', 'task_id': 'a', 'family': 'f'} for i in range(3)]
    guided = [{'key': f'g{t}{i}', 'source': 'jolt-guided', 'task_id': t, 'family': 'small', 'guided': True}
              for t in ('a', 'b') for i in range(3)]
    return own + guided


def test_own_and_guided_successes_share_one_task_cap_own_first():
    funnel = collections.defaultdict(collections.Counter)
    pool, _, _ = s.student_selection(_student_candidates(), random.Random(0), funnel, lambda task: 2, 1.0,
                                     lambda c: False, 1.0)
    by_task = collections.defaultdict(list)
    for c in pool:
        by_task[c['task_id']].append(c['source'])
    assert sorted(by_task['a']) == ['rl-own', 'rl-own']            # one cap of 2 for both tiers, own taken first
    assert len(by_task['b']) == 2 and funnel['jolt-guided']['over_task_cap'] == 4


def test_the_family_share_is_measured_on_the_joint_student_pool():
    # 3 own episodes of family f, 3 guided of family small: a 0.5 share of the joint pool (6) keeps 3 small ones,
    # where a separate guided pool (3) would have kept 1
    candidates = [c for c in _student_candidates() if c['task_id'] == 'b' or c['source'] == 'rl-own']
    pool, _, _ = s.student_selection(candidates, random.Random(0), collections.defaultdict(collections.Counter),
                                     lambda task: 9, 0.5, lambda c: False, 1.0)
    assert sum(c['source'] == 'jolt-guided' for c in pool) == 3


def test_frontier_guided_only_tasks_keep_their_teacher_data_until_guided_enters():
    funnel = collections.defaultdict(collections.Counter)
    tiers = {'a': 'sometimes', 'b': 'never'}
    pool, own_tasks, guided_tasks = s.student_selection(_student_candidates(), random.Random(0), funnel,
                                                        lambda task: 2, 1.0, lambda c: False, 1.0, tiers=tiers)
    assert own_tasks == {'a'} and guided_tasks == {'b'}
    assert funnel['jolt-guided']['frontier_own_success'] == 3         # task a: its unguided successes train instead
    teacher = [{'task_id': 'b', 'key': 't1'}, {'task_id': 'c', 'key': 't2'}]
    replaced, trains = s.settle_deferred(teacher, {'a', 'b'}, lambda c: c['task_id'])
    assert [c['key'] for c in replaced] == ['t1'] and [c['key'] for c in trains] == ['t2']
    replaced, trains = s.settle_deferred(teacher, {'a'}, lambda c: c['task_id'])   # b's guided cut by the own share
    assert not replaced and len(trains) == 2


def test_the_task_cap_takes_the_frontier_policys_own_successes_before_older_policies():
    # aei1 round 7: the per-task cap picked producers at random, so 35% of the own share was grpo-v3/v4 episodes
    own = [{'key': f'old{i}', 'source': 'rl-own', 'task_id': 'a', 'family': 'f', 'policy': 'grpo'} for i in range(6)]
    own += [{'key': 'new', 'source': 'rl-own', 'task_id': 'a', 'family': 'f', 'policy': 'P'},
            {'key': 'prev', 'source': 'rl-own', 'task_id': 'a', 'family': 'f', 'policy': 'Q'}]
    for seed in range(5):
        pool, _, _ = s.student_selection(own, random.Random(seed), collections.defaultdict(collections.Counter),
                                         lambda task: 2, 1.0, lambda c: False, 1.0, recent=('P', 'Q'))
        assert sorted(c['key'] for c in pool) == ['new', 'prev']
    # an untrained older success still goes before a trained recent one (--prefer-unseen first)
    pool, _, _ = s.student_selection(own, random.Random(0), collections.defaultdict(collections.Counter),
                                     lambda task: 1, 1.0, lambda c: False, 1.0, trained={'new', 'prev'},
                                     recent=('P', 'Q'))
    assert pool[0]['policy'] == 'grpo'
    picks = {s.student_selection(own, random.Random(seed), collections.defaultdict(collections.Counter),
                                 lambda task: 1, 1.0, lambda c: False, 1.0)[0][0]['key'] for seed in range(12)}
    assert len(picks) > 1                                              # without recent (--own-cap-order any): random


def test_guided_replays_get_the_actors_context():
    assert s.replay_context_tokens('jolt-guided') == s.replay_context_tokens('rl-own') == 131072
    assert s.replay_context_tokens('teacher') == 65536


def test_extract_pass_rates_count_student_samples_only_and_policies_come_from_each_call(tmp_path):
    import types
    run, archive = tmp_path / 'run', tmp_path / 'archive'
    group = run / 'groups' / 'g000001'
    group.mkdir(parents=True)
    archive.mkdir()
    (group / 'cases.jsonl').write_text(json.dumps({'id': 't', 'family': 'f', 'split': 'train'}) + '\n')
    (group / 'score.json').write_text('{}')
    turn = lambda policy: {'messages': [{'role': 'user', 'content': 'q'}], 'target': {'role': 'assistant', 'content': 'a'},
                           'invocation_id': 'root', 'sampling': {'prompt': [{'kind': 'tokens', 'token_ids': [1]}],
                                                                 'samples': [{'token_id': 5, 'logp': -0.1}],
                                                                 'policy': {'checkpoint_sha256': policy}}}
    rows = [{'task_id': 't', 'sample': 0, 'outcome': 'wrong_result', 'reward': {'quality': 0.0},
             'policy': {'checkpoint_sha256': 'p2'}, 'turns': [turn('p2')]},
            {'task_id': 't', 'sample': 1, 'outcome': 'task_success', 'reward': {'quality': 1.0},
             'policy': {'checkpoint_sha256': 'p2'}, 'turns': [turn('p1'), turn('p2')]}]
    rows += [{'task_id': 't', 'sample': 8 + i, 'outcome': 'task_success', 'reward': {'quality': 1.0},
              'jolt_role': 'teacher', 'turns': [turn('p2')]} for i in range(2)]
    (group / 'rollouts.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    work = tmp_path / 'work'
    s.extract(types.SimpleNamespace(work=str(work), rl=[f'{run}:{archive}'], teacher=[], out_name='episodes'))
    assert json.loads((work / 'rl-group-rates.json').read_text())['t'] == {'successes': 1, 'scored': 2}
    own = [json.loads(line) for line in s.zstd_lines(work / 'episodes.jsonl.zst') if '"rl-own"' in line]
    assert own[0]['policies'] == ['p1', 'p2']                          # an episode straddling a rebind


def test_extract_by_worker_processes_writes_the_serial_extraction(tmp_path):
    import types
    import zstandard
    run, archive, pool = tmp_path / 'run', tmp_path / 'archive', tmp_path / 'pool'
    archive.mkdir()
    turn = lambda i: {'messages': [{'role': 'user', 'content': f'q{i}'}], 'target': {'role': 'assistant', 'content': 'a'},
                      'invocation_id': 'root', 'sampling': {'prompt': [{'kind': 'tokens', 'token_ids': [1, i]}],
                                                            'samples': [{'token_id': 5, 'logp': -0.1}]}}
    for g in range(23):
        name = f'g{g:06d}'
        group = run / 'groups' / name
        group.mkdir(parents=True)
        (group / 'cases.jsonl').write_text(json.dumps({'id': f't{g % 5}', 'family': 'f', 'split': 'train'}) + '\n')
        (group / 'score.json').write_text('{}')
        rows = [{'task_id': f't{g % 5}', 'sample': k, 'outcome': 'task_success' if (g + k) % 3 else 'wrong_result',
                 'reward': {'quality': float((g + k) % 3 > 0)}, 'turns': [turn(g * 10 + k)],
                 **({'unscored': True} if (g, k) == (4, 1) else {})} for k in range(4)]
        text = ''.join(json.dumps(r) + '\n' for r in rows)
        if g % 4 == 0:      # archived groups
            (archive / f'{name}.rollouts.jsonl.zst').write_bytes(zstandard.ZstdCompressor().compress(text.encode()))
        else:
            (group / 'rollouts.jsonl').write_text(text)
    pool.mkdir()
    with (pool / 'manifest.jsonl').open('w') as manifest:
        for t in range(17):
            directory = pool / f't{t:06d}'
            directory.mkdir()
            (directory / 'cases.jsonl').write_text(json.dumps({'id': f'x{t}', 'family': 'h'}) + '\n')
            (directory / 'rollouts.jsonl').write_text(json.dumps(
                {'task_id': f'x{t}', 'sample': 0, 'outcome': 'task_success' if t % 2 else 'timeout',
                 'reward': {'quality': 1.0}, 'turns': [turn(t)]}) + '\n')
            manifest.write(json.dumps({'rollouts': str(directory / 'rollouts.jsonl'), 'teacher': 'm'}) + '\n')
    outputs = []
    for workers in (1, 3):
        work = tmp_path / f'work{workers}'
        s.extract(types.SimpleNamespace(work=str(work), rl=[f'{run}:{archive}'], teacher=[str(pool)],
                                        out_name='episodes', workers=workers))
        outputs.append([list(s.zstd_lines(work / 'episodes.jsonl.zst')),
                        (work / 'episodes-extract.json').read_text(), (work / 'rl-group-rates.json').read_text()])
    assert outputs[0] == outputs[1]
    census = json.loads(outputs[0][1])['sources']
    assert census[f'rl:{tmp_path.name}/run']['groups'] == 23 and census[f'teacher:{tmp_path.name}/pool']['successes'] == 8
    assert len(outputs[0][0]) == json.loads(outputs[0][1])['episodes'] == 8 + sum(
        1 for g in range(23) for k in range(4) if (g + k) % 3 and (g, k) != (4, 1))


def test_unfilled_live_teacher_episodes_train_with_teacher_reasoning_as_context_unless_dropped():
    """Owner 2026-11-24: an observed-live teacher episode no filled unit replaced stays in an expert-iteration build
    (its think block is zero-weight context, its actions train); ``drop`` is the opt-out (the old auto rule)."""
    chains = [{'part': 'teacher', 'round': 'live:vibestudio:a:1'}, {'part': 'teacher', 'round': 'live:vibestudio:a:1'},
              {'part': 'teacher', 'round': 'teacher:pool:t:0'}, {'part': 'filled', 'round': 'filled:live:vibestudio:b:2'}]
    filled = [s.parse_aux('filled=/f.pkl,share=all,replaces=teacher')]
    assert s.drop_unfilled_live(chains, 'auto', filled) == (chains, 0)
    assert s.drop_unfilled_live(chains, 'auto', [s.parse_aux('live=/l.pkl')]) == (chains, 0)
    assert s.drop_unfilled_live(chains, 'train', filled) == (chains, 0)
    kept, n = s.drop_unfilled_live(chains, 'drop', filled)
    assert n == 1 and [c['round'] for c in kept] == ['teacher:pool:t:0', 'filled:live:vibestudio:b:2']
    # a 'filled' treatment without a fill renders its think block as context, never as a target
    treatment = s.ThinkTreatment({'default': 'context', 'models': {'m': 'filled'}})
    assert treatment.render({'source': 'teacher', 'teacher': 'm'}) == 'context'


def test_partial_actions_do_not_displace_completed_demonstrations_for_same_goal():
    def candidate(key, text, completed):
        return {'key': key, 'source': 'teacher', 'open_goal': 'Same goal', 'task_completed': completed,
                'signature': s.minhash(s.shingles(text))}
    partial = candidate('partial', 'inspect independently bound original source and verify current revision', False)
    complete = candidate('complete', 'write requested correction then read persisted application output', True)
    repeated = candidate('repeated', 'a wholly different successful path for same user goal', True)
    funnel = collections.defaultdict(collections.Counter)
    selected = s.distinct_candidates([partial, complete, repeated], 0.8, funnel)
    assert [row['key'] for row in selected] == ['complete', 'partial']
    assert funnel['teacher']['repeated_open_goal'] == 1


def test_evidence_only_live_episodes_do_not_enter_supervised_candidates(monkeypatch, tmp_path):
    import types
    monkeypatch.setattr(s, 'verified_episodes', lambda *args, **kwargs: iter([
        ({'source': 'teacher', 'verification_mode': 'observed-live', 'turns': []}, {})]))
    monkeypatch.setattr(s, 'teacher_coverage', lambda *args: (0, []))
    monkeypatch.setattr(s, 'held_excluded_ids', lambda *args: set())
    args = types.SimpleNamespace(teacher_pool=[], allow_stale_teacher=False, held_manifest='held')
    funnel = collections.defaultdict(collections.Counter)
    candidates, _ = s.core_candidates(args, tmp_path, funnel, set())
    assert candidates == []
    assert funnel['teacher']['observed_evidence_without_supervised_targets'] == 1


def test_replay_answers_a_field_mode_readout_with_its_whole_record_and_declines_scoring():
    """A field-mode readout's steps were recorded as calls with the same messages (each its open step text): a
    declined readout asks once by generation, and the replay serves the call whose reply is the whole record, its step
    calls with it. A /v1/natlang/score request is declined item by item."""
    request = {'role': 'user', 'content': 'Answer this call now, without tools. Reply with exactly one JSON value of '
                                          'this form and nothing else: {"a":<1 | 2>,"b":<true | false>}.'}
    steps = [turn([SYSTEM, USER, request], content='{"a":2,"'), turn([SYSTEM, USER, request], content='{"a":2,"b":true}')]
    episode = s.ReplayEpisode(steps)
    assert episode.answer({'messages': [SYSTEM, USER, request]})['choices'][0]['message']['content'] == '{"a":2,"b":true}'
    assert episode.served == [True, True]
    declined = episode.answer({'items': [{'messages': [SYSTEM, USER, request], 'continuations': ['{"a":1,"']}]})
    assert declined['results'][0]['error'].startswith('decision-unsupported')


def test_neuralese_graph_admits_episodes_whose_reads_were_in_context():
    """A Neuralese episode whose reads the engine fed inline (no lowered import) is admitted as plain rows by default;
    ``require_import`` restores the old drop; an episode with an import is admitted either way."""
    plain, spliced = [{'imports': []}, {'imports': []}], [{'imports': []}, {'imports': [(3, 'nzs1_x')]}]
    assert s.import_gate(plain) == 'in_context_reads'
    assert s.import_gate(plain, require_import=True) == 'no_lowered_import'
    assert s.import_gate(spliced) == s.import_gate(spliced, require_import=True) == 'imports'
