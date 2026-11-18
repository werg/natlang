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
    vocab = ['<think>', '</think>', '\n\n', '<tool_call>', 'eval', '</tool_call>', '<|im_end|>', 'plan']

    def convert_tokens_to_ids(self, token):
        return self.vocab.index(token)

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
    assert s.think_context(renderer, 'rl-own', no_reasoning, target) == 0
    assert s.think_context(renderer, 'rl-own', {**no_reasoning, 'reasoning_content': 'plan'}, [7, 1, 3, 4, 5, 6]) == 0


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


def test_wrapper_cap_keeps_dataset_wrappers_to_their_share_of_the_selection():
    import collections
    import random
    selected = [{'source': 'teacher', 'task_id': f'w{i}', 'w': True} for i in range(50)] + \
               [{'source': 'teacher', 'task_id': f'r{i}', 'w': False} for i in range(30)]
    funnel = collections.defaultdict(collections.Counter)
    kept = s.wrapper_capped(selected, random.Random(0), funnel, lambda c: c['w'], 0.25)
    assert sum(not c['w'] for c in kept) == 30 and sum(c['w'] for c in kept) == 10
    assert funnel['teacher']['over_wrapper_share'] == 40


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
