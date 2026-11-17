"""SFT source builder (scripts/neuralese_data/sft_sources.py): replay matching, near-duplicates, held closure."""
import json
import sys
from pathlib import Path

import numpy as np

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
    context = s.reasoning_free_context(renderer, 'teacher', no_reasoning, target)
    assert context == 2                  # </think> and the whitespace after it are context
    weight = (np.arange(len(target)) >= context).astype(np.float32)
    assert weight.tolist() == [0, 0, 1, 1, 1, 1]
    # the same weights reach the loss through exp0_sft's chains (row['weight'])
    import exp0_sft
    row = {'prompt': np.array([0], np.int32), 'sample': np.array(target, np.int32), 'weight': weight,
           'invocation': 'i', 'index': 0}
    chain = exp0_sft.chains_of([{'task_id': 't', 'round': 'r', 'rows': [row]}])[0]
    assert exp0_sft.chain_weights(chain, 'cpu').tolist() == [0, 0, 1, 1, 1, 1]
    # October rows too; a reply with reasoning and the student's own replies keep every token as a target
    assert s.reasoning_free_context(renderer, 'october:luna-curricula', no_reasoning, target) == 2
    assert s.reasoning_free_context(renderer, 'teacher', {**no_reasoning, 'reasoning_content': 'plan'},
                                    [7, 1, 3, 4, 5, 6]) == 0
    assert s.reasoning_free_context(renderer, 'rl-own', no_reasoning, target) == 0
