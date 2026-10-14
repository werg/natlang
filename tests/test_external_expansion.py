"""External source fidelity and causal task grouping, without downloading fixtures."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from neuralese_data.agents import expansion_messages, expansion_trajectory
from neuralese_data.trajectory import record


def test_tagged_dispatch_keeps_arguments_and_reasoning():
    raw = [{'role': 'assistant', 'content': '<think>Choose workers.</think>\n<tool_call>\n'
            '{"name":"create_sub_agents","arguments":{"sub_agents":[{"prompt":"Find X"}]}}'
            '\n</tool_call>'}]
    message = expansion_messages(raw)[0]
    assert message['reasoning'] == 'Choose workers.'
    assert message['tool_calls'][0]['function']['name'] == 'create_sub_agents'
    assert 'Find X' in message['tool_calls'][0]['function']['arguments']
    assert '<tool_call>' not in (message['content'] or '')


def test_sibling_sessions_share_task_group_and_no_future_enters_inputs():
    receipt = {'repo': 'test/source', 'revision': 'abc', 'card': {'license': 'apache-2.0'}}
    messages = [{'role': 'user', 'content': 'Solve the shared task'}] + [
        {'role': role, 'content': content} for role, content in [
            ('assistant', 'Earlier step'), ('tool', 'Earlier observation'),
            ('assistant', 'Recent step'), ('tool', 'Recent observation'),
            ('assistant', 'Target'), ('tool', 'FUTURE SECRET')]]
    def trajectory(session):
        return expansion_trajectory('D01', receipt, session, {'messages': messages,
            'metadata': {'trajectory_id': 'shared', 'session_id': session}})
    first, second = trajectory('planner'), trajectory('worker')
    assert first.id != second.id and first.split_groups == second.split_groups
    row = record(first, 3, 5)
    from neuralese_data.records import validate
    assert validate(row) == []
    import json
    assert 'FUTURE SECRET' not in json.dumps(row)
    assert row['writer']['result_type'] == 'Neuralese<SessionHistory>'


def test_native_builder_preserves_structured_target_and_source_calls(tmp_path):
    import json
    import subprocess
    import shutil
    receipt = {'repo': 'test/source', 'revision': 'abc'}
    call = {'type': 'function', 'function': {'name': 'search', 'arguments': '{"query":"KEEP QUERY"}'}}
    messages = [{'role': 'user', 'content': 'Find the answer'},
                {'role': 'assistant', 'content': None, 'tool_calls': [call]},
                {'role': 'tool', 'content': 'Observed answer'},
                {'role': 'assistant', 'content': 'Recent step'},
                {'role': 'tool', 'content': 'Recent observation'},
                {'role': 'assistant', 'content': None, 'tool_calls': [call]}]
    trajectory = expansion_trajectory('D01', receipt, 'sample', {'messages': messages})
    row = record(trajectory, 3, 5)
    source = tmp_path / 'fixture.port-records.jsonl'
    source.write_text(json.dumps(row) + '\n')
    reserved = tmp_path / 'reservations.jsonl'
    reserved.write_text('')
    root = Path(__file__).resolve().parents[1]
    toolchain_node = Path.home() / '.local/bin/node'
    node = str(toolchain_node) if toolchain_node.exists() else shutil.which('node')
    subprocess.run([node, str(root / 'ts-host/scripts/inline-curriculum/build-port-record-calls.mjs'),
                    '--input', str(source), '--out', str(tmp_path / 'built'), '--reservations', str(reserved)],
                   check=True, capture_output=True)
    built = tmp_path / 'built/fixture'
    assert json.loads((built / 'build.json').read_text())['items'] == 1
    def strings(value):
        if isinstance(value, str): yield value
        elif isinstance(value, dict):
            for item in value.values(): yield from strings(item)
        elif isinstance(value, list):
            for item in value: yield from strings(item)
    values = list(strings(json.loads(next(built.glob('*.ir.jsonl')).read_text())))
    assert json.dumps(row['target']['value'], separators=(',', ':'), ensure_ascii=False) in values
    assert any('KEEP QUERY' in value and 'evidence' in value for value in values)


def test_codeagent_task_handoff_keeps_problem_and_prior_history_out_of_consumer(tmp_path, monkeypatch):
    import json
    from neuralese_data import agents
    from neuralese_data.records import validate
    data = tmp_path / 'D05'
    data.mkdir()
    (data / 'acquisition.json').write_text(json.dumps({'repo': 'test/codeagent', 'revision': 'pinned'}))
    (data / 'CodeAgentSFTDataset.json').write_text(json.dumps([{'instruction': 'Write a sum function.',
        'input': 'Use integers.', 'history': [['Previous question', 'Previous answer']],
        'output': 'def sum_two(a, b): return a + b'}]))
    monkeypatch.setattr(agents, 'EXPANSION_ROOT', tmp_path)
    _, trajectory = next(agents.codeagent_sft())
    row = record(trajectory, trajectory.head, trajectory.head)
    assert validate(row) == []
    assert 'Write a sum function.' in row['sources'][0]['text']
    assert 'Previous answer' in row['sources'][0]['text']
    assert 'Write a sum function.' not in json.dumps(row['consumer'])
    assert row['target']['value']['content'] == 'def sum_two(a, b): return a + b'
    assert row['writer']['result_type'] == 'Neuralese<Background>'


def test_evidence_subagent_keeps_published_eval_split_and_legacy_tool_observation(tmp_path, monkeypatch):
    import json
    import pytest
    pa = pytest.importorskip('pyarrow')
    import pyarrow.parquet as pq
    from neuralese_data import agents
    root = tmp_path / 'D10'
    root.mkdir()
    (root / 'acquisition.json').write_text(json.dumps({'repo': 'test/evidence', 'revision': 'pinned'}))
    example = {'conversations': [{'role': 'user', 'content': 'Find evidence'},
        {'role': 'function', 'content': 'Exact observed result', 'name': 'search'},
        {'role': 'assistant', 'content': 'Report the result'}]}
    for filename in ('train.parquet', 'eval.parquet'):
        pq.write_table(pa.Table.from_pylist([example]), root / filename)
    monkeypatch.setattr(agents, 'EXPANSION_ROOT', tmp_path)
    rows = dict(agents.evidence_subagent())
    assert rows['eval.parquet:0'].split == 'test'
    assert rows['train.parquet:0'].split == 'train'
    observation = rows['eval.parquet:0'].messages[1]
    assert observation['role'] == 'tool' and observation['content'] == 'Exact observed result'
