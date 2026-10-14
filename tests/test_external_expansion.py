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
    import json
    assert 'FUTURE SECRET' not in json.dumps(row)
    assert row['writer']['result_type'] == 'Neuralese<SessionHistory>'


def test_native_builder_preserves_structured_target_and_source_calls(tmp_path):
    import json
    import subprocess
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
    subprocess.run(['/home/werg/.local/bin/node', str(root / 'ts-host/scripts/inline-curriculum/build-port-record-calls.mjs'),
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
