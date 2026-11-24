"""FALLOW-22 broad-graph trajectories -> October rows (scripts/neuralese_data/broad_graph_october.py)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/neuralese_data'))
import broad_graph_october as b  # noqa: E402
import sft_sources as s  # noqa: E402


def record(trajectory, index, groups, *, converted=True):
    program = {'id': f'task-{trajectory}', 'family': 'curriculum_broad_graph_x', 'split': 'train',
               'source_ids': [f'src-{trajectory}'], 'source_groups': groups, 'semantics': {'files': {}}}
    plain = [{'role': 'system', 'content': 'plain prompt'}, {'role': 'user', 'content': f'call {index}'}]
    target = {'role': 'assistant', 'content': f'answer {index}', 'reasoning_content': 'I run the next step in eval.'}
    r = {'id': f'teacher-program:{trajectory}:decision:{index:04d}', 'split': 'train',
         'source_ref': {'trajectory_id': f'teacher-program:{trajectory}', 'invocation_id': f'inv-{index}'},
         'provenance': {'model': 'port-record-reference'}, 'task': {'program_ir': program},
         'messages': [{'role': 'system', 'content': 'converted prompt with native spans'}],
         'target': {'role': 'assistant', 'content': '<native span>'}, 'tools': [{'type': 'function'}]}
    if converted:
        r['native_source_conversion'] = {'source_messages': plain, 'source_target': target}
    return r


def fallow(tmp_path):
    root, work = tmp_path / 'fallow', tmp_path / 'work'
    shard = root / 'x-s0'
    shard.mkdir(parents=True)
    work.mkdir()
    (shard / 'graphs-train.jsonl.zst').write_bytes(b'')
    (shard / 'receipt.json').write_text(json.dumps({'source_binding': {'inputs': {'work': str(work)}}}))
    (shard / 'ids-graphs-heldfiltered.txt').write_text('chain:x:teacher-program:a\nchain:x:teacher-program:b\n')
    rows = [record('a', 1, ['g-a']), record('a', 0, ['g-a']), record('b', 0, ['held-group']),
            record('c', 0, ['g-c'])]          # c: not a kept train graph
    (work / 'shard-0000.records.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return root


def test_plain_rows_in_decision_order_held_groups_out(tmp_path):
    held = tmp_path / 'held-groups.json'
    held.write_text(json.dumps(['held-group']))
    out = tmp_path / 'out'
    report = b.convert(fallow(tmp_path), out, exclude_groups=[held])
    assert report['totals'] == {'wanted': 2, 'found': 2, 'held_excluded': 1, 'calls': 2, 'trajectories': 1}
    rows = [json.loads(line) for line in s.zstd_lines(out / 'october' / f'{b.GROUP}.jsonl.zst')]
    assert [r['id'] for r in rows] == ['teacher-program:a:decision:0000', 'teacher-program:a:decision:0001']
    first = rows[0]
    # the call as it originally ran (no native spans), the synthetic reasoning note dropped
    assert first['messages'][0]['content'] == 'plain prompt'
    assert first['target'] == {'role': 'assistant', 'content': 'answer 0'}
    assert first['provenance']['trajectory'] == 'teacher-program:a'
    assert first['provenance']['invocation'] == 'inv-0'
    assert first['task_id'] == 'task-a' and first['source_groups'] == ['g-a']
    # the extract stage reads each trajectory's task row back from the rows' source file
    tasks = s.october_task_rows([first['provenance']['source_file']], {'teacher-program:a'})
    assert tasks['teacher-program:a']['id'] == 'task-a'
    assert json.loads((out / 'avoid-groups.json').read_text()) == ['g-a']
    assert set(json.loads((out / 'avoid-ids.json').read_text())) == {'task-a', 'src-a'}


def test_record_without_conversion_keeps_its_messages(tmp_path):
    r = record('z', 0, [], converted=False)
    messages, tools, target = b.plain_call(r)
    assert messages == r['messages'] and target == {'role': 'assistant', 'content': '<native span>'}
