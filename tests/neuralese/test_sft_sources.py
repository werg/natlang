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
