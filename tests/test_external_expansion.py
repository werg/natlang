"""External source fidelity and causal task grouping, without downloading fixtures."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from neuralese_data.agents import expansion_messages, expansion_trajectory
from neuralese_data.trajectory import record


def test_gatc_preserves_tools_and_protects_family_through_flagged_parent(tmp_path, monkeypatch):
    import json
    from neuralese_data import agents
    from neuralese_data.records import group_key
    root = tmp_path / 'D03'; root.mkdir()
    (root / 'acquisition.json').write_text(json.dumps({'repo':'fixture/gatc','revision':'abc'}))
    def row(tid, **extra):
        return {'trajectory_id':tid, 'messages':[
            {'role':'user','content':'Shared harness wrapper','is_wrapper':True},
            {'role':'user','content':'Repair task '+tid,'is_wrapper':False},
            {'role':'assistant','content':'Edit','tool_calls':[{'id':'c1','function':{'name':'edit','arguments':'{}'}}]},
            {'role':'tool','content':'Observed','tool_call_id':'c1'},
            {'role':'assistant','content':'Done'}], 'tools':json.dumps([{'type':'function','function':{'name':'edit'}}]),
            **extra}
    rows=[row('parent',is_eval=True),row('bridge',parent_trajectory_id='parent',ft_issues=['incomplete_tool_args']),
          row('child',parent_trajectory_id='bridge',last_test_result='failed'),row('other')]
    monkeypatch.setattr(agents,'EXPANSION_ROOT',tmp_path)
    monkeypatch.setattr(agents,'_gatc_rows',lambda root:iter(enumerate(rows)))
    values=[t for _,t in agents.gatc_sessions()]
    assert [t.split for t in values]==['test','test','test','train']
    assert set(values[0].split_groups)&set(values[2].split_groups)
    assert group_key('task',agents.task_hash('Shared harness wrapper')) not in values[0].split_groups
    assert values[2].tools[0]['function']['name']=='edit'
    assert values[2].messages[2]['tool_calls'][0]['id']=='c1'
    assert values[2].messages[3]['tool_call_id']=='c1'
    assert values[2].outcome['label']=='teacher'
    assert values[2].lineage['notes']['metadata']['last_test_result']=='failed'


def test_source_quality_filter_keeps_held_protection_and_records_train_omission(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    import neuralese_port_records as cli
    def trajectory(split, flags):
        return SimpleNamespace(split=split,lineage={'notes':{'source_quality_flags':flags}},outcome={'label':'teacher'})
    values=[trajectory('train',['withheld_text']),trajectory('test',['withheld_text']),trajectory('train',[])]
    monkeypatch.setitem(cli.agents.CORPORA,'fixture',(lambda limit:iter(enumerate(values)),4))
    seen=[]
    monkeypatch.setattr(cli.trajectory,'records_for',lambda t,*args:seen.append(t))
    cli.cmd_convert_agents(SimpleNamespace(corpora=['fixture'],limit=None,out=tmp_path,windows=None,
                                          exclude_failed=False,exclude_source_quality_flags=True))
    assert seen==values[1:]
    omitted=json.loads((tmp_path/'upstream.fixture.rejects.jsonl').read_text())
    assert omitted['reason']=='source-quality-flags: withheld_text'


def test_coding_sessions_preserve_dispatch_outcome_and_shared_benchmark_group(tmp_path, monkeypatch):
    import json
    from neuralese_data import agents
    from neuralese_data.records import group_key
    root = tmp_path / 'D40'
    root.mkdir()
    (root / 'acquisition.json').write_text(json.dumps({'repo':'fixture/coding', 'revision':'abc'}))
    calls = [{'id':'call-1','function':{'name':'edit','arguments':'{"path":"a.py"}'}}]
    row = {'session_id':'session-1','source_id':'instance-1','recorded_model':'teacher',
           'messages_json':json.dumps([
               {'role':'user','content':'Repair the file.'},
               {'role':'assistant','content':'Apply edit','tool_calls_json':json.dumps(calls)},
               {'role':'tool','content':'Observed edit','tool_call_id':'call-1'},
               {'role':'assistant','content':'Done'}]),
           'ground_truth_meta_json':json.dumps({'instance_id':'instance-1','resolved':False})}
    monkeypatch.setattr(agents, 'EXPANSION_ROOT', tmp_path)
    monkeypatch.setattr(agents, '_parquet_groups', lambda path: iter([(0,row)]))
    monkeypatch.setattr(agents, '_bench_ids', lambda: {'instance-1'})
    _, t = next(agents.agentic_coding_sessions())
    assert t.messages[1]['tool_calls'] == [{'type':'function', **calls[0]}]
    assert t.messages[2]['tool_call_id'] == 'call-1'
    assert t.outcome['label'] == 'failed' and t.outcome['checked'] is None
    assert t.split == 'test'
    assert group_key('swe-instance','instance-1') in t.split_groups
    assert 'Repair the file.' in json.dumps(record(t, t.head, 1)['sources'])


def test_excluding_failed_conversion_records_omission_and_continues(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    import neuralese_port_records as cli
    failed = SimpleNamespace(outcome={'label':'failed'})
    good = SimpleNamespace(outcome={'label':'teacher'})
    monkeypatch.setitem(cli.agents.CORPORA, 'fixture', (lambda limit: iter([('bad',failed),('good',good)]), 4))
    seen = []
    monkeypatch.setattr(cli.trajectory, 'records_for', lambda t,*args: seen.append(t))
    cli.cmd_convert_agents(SimpleNamespace(corpora=['fixture'],limit=None,out=tmp_path,
                                          windows=None,exclude_failed=True))
    assert seen == [good]
    omitted = json.loads((tmp_path/'upstream.fixture.rejects.jsonl').read_text())
    assert omitted['row'] == 'bad' and omitted['reason'].startswith('upstream-reported-failure')


def test_literal_function_dispatch_never_executes_nonliteral_arguments():
    from neuralese_data.agents import _literal_function_calls
    import json
    calls = _literal_function_calls('weather.forecast(city="Berlin", days=2)\nmath.pairs(nums=[2, 5], target=7)')
    assert [c['function']['name'] for c in calls] == ['weather.forecast', 'math.pairs']
    assert json.loads(calls[1]['function']['arguments']) == {'nums': [2, 5], 'target': 7}
    assert _literal_function_calls('tool.run(value=__import__("os").getcwd())') is None
    assert _literal_function_calls('tool.run(**options)') is None


def test_dolci_keeps_first_dispatch_schemas_and_observations(tmp_path, monkeypatch):
    import json
    from neuralese_data import agents
    from neuralese_data.trajectory import windows
    data = tmp_path / 'D30'
    data.mkdir()
    (data / 'acquisition.json').write_text(json.dumps({'repo':'fixture/dolci', 'revision':'abc'}))
    (data / 'train.parquet').touch()
    schema = {'type':'function','function':{'name':'math.pairs','parameters':{'type':'object'}}}
    row = {'id':'session-1','dataset_source':'fixture/mix','messages':[
        {'role':'system','content':'Call the provided functions.', 'functions':json.dumps([schema])},
        {'role':'user','content':'Find pairs from [2, 5] summing to 7.'},
        {'role':'assistant','function_calls':'math.pairs(nums=[2, 5], target=7)'},
        {'role':'environment','content':'OBSERVATION'},
        {'role':'assistant','content':'Final response'}]}
    monkeypatch.setattr(agents, 'EXPANSION_ROOT', tmp_path)
    monkeypatch.setattr(agents, '_parquet_groups', lambda path: iter([(0,row)]))
    _, trajectory = next(agents.dolci_tool_use())
    assert trajectory.tools == [schema]
    assert trajectory.messages[3]['role'] == 'tool'
    assert trajectory.messages[3]['content'] == 'OBSERVATION'
    first = record(trajectory, trajectory.head, 2)
    assert 'Find pairs' in json.dumps(first['sources'])
    assert 'Find pairs' not in json.dumps(first['consumer'])
    assert 'OBSERVATION' not in json.dumps(first)
    assert first['target']['value']['tool_calls'][0]['function']['name'] == 'math.pairs'
    assert (trajectory.head, 2) in windows(trajectory, 4, seed=1)


def test_cognitive_single_action_gets_a_causal_task_background(tmp_path, monkeypatch):
    import json
    from neuralese_data import agents
    from neuralese_data.trajectory import windows
    data = tmp_path / 'D34'
    data.mkdir()
    (data / 'acquisition.json').write_text(json.dumps({'repo':'fixture/kernel', 'revision':'abc'}))
    (data / 'docbench.sft.jsonl').write_text(json.dumps({'messages':[
        {'role':'system','content':'Produce the next Python action.'},
        {'role':'user','content':'TASK EVIDENCE: inspect report.pdf and extract its total.'},
        {'role':'assistant','content':'print(read_document("report.pdf"))'}]})+'\n')
    monkeypatch.setattr(agents, 'EXPANSION_ROOT', tmp_path)
    _, trajectory = next(agents.cognitive_kernel())
    assert windows(trajectory, 4, seed=1) == [(2, 2)]
    row = record(trajectory, 2, 2)
    assert 'TASK EVIDENCE' in json.dumps(row['sources'])
    assert 'TASK EVIDENCE' not in json.dumps(row['consumer'])
    assert row['target']['value']['content'] == 'print(read_document("report.pdf"))'
    assert trajectory.lineage['notes']['benchmark_source_file'] == 'docbench.sft.jsonl'


def test_tagged_dispatch_keeps_arguments_and_reasoning():
    raw = [{'role': 'assistant', 'content': '<think>Choose workers.</think>\n<tool_call>\n'
            '{"name":"create_sub_agents","arguments":{"sub_agents":[{"prompt":"Find X"}]}}'
            '\n</tool_call>'}]
    message = expansion_messages(raw)[0]
    assert message['reasoning'] == 'Choose workers.'
    assert message['tool_calls'][0]['function']['name'] == 'create_sub_agents'
    assert 'Find X' in message['tool_calls'][0]['function']['arguments']
    assert '<tool_call>' not in (message['content'] or '')


def test_evidence_function_role_is_a_dispatch_when_it_contains_a_call():
    raw = [{'role': 'function', 'content': '<think>Search the source.</think>\n'
            '{"name":"google_search","arguments":{"arguments":[{"query":"evidence"}]}}'},
           {'role': 'tool', 'content': 'Observed evidence'},
           {'role': 'assistant', 'content': 'Final report'}]
    messages = expansion_messages(raw)
    assert [m['role'] for m in messages] == ['assistant', 'tool', 'assistant']
    assert messages[0]['reasoning'] == 'Search the source.'
    assert messages[0]['tool_calls'][0]['function']['name'] == 'google_search'
    assert messages[1]['content'] == 'Observed evidence'


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
                {'role': 'tool', 'content': 'Observed answer', 'name':'search', 'tool_call_id':'source-call-42'},
                {'role': 'assistant', 'content': 'Recent step'},
                {'role': 'tool', 'content': 'Recent observation'},
                {'role': 'assistant', 'content': None, 'tool_calls': [call]}]
    trajectory = expansion_trajectory('D01', receipt, 'sample', {'messages': messages})
    trajectory.tools = [{'type':'function','function':{'name':'search','description':'SCHEMA CONTRACT',
                         'parameters':{'type':'object','properties':{'query':{'type':'string'}}}}}]
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
    assert any('SCHEMA CONTRACT' in value for value in values)
    assert any('source-call-42' in value for value in values)


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


def test_native_replay_disambiguates_reused_upstream_ids(tmp_path):
    import json, subprocess, shutil
    root = Path(__file__).resolve().parents[1]
    node = str(Path.home() / '.local/bin/node') if (Path.home() / '.local/bin/node').exists() else shutil.which('node')
    receipt = {'repo':'fixture/reused','revision':'abc'}
    def make(answer):
        t = expansion_trajectory('D40',receipt,'session',{'messages':[
            {'role':'user','content':'Continue coding'},
            {'role':'assistant','content':'Earlier '+answer},
            {'role':'tool','content':'Observed '+answer},
            {'role':'assistant','content':'Recent step'},
            {'role':'tool','content':'Recent observation'},
            {'role':'assistant','content':answer}]})
        return record(t,3,5)
    rows=[make('FIRST ANSWER'),make('SECOND ANSWER')]
    assert rows[0]['id']==rows[1]['id']
    source=tmp_path/'fixture.port-records.jsonl';source.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    reservations=tmp_path/'reserved.jsonl';reservations.write_text('')
    subprocess.run([node,str(root/'ts-host/scripts/inline-curriculum/build-port-record-calls.mjs'),'--input',str(source),'--out',str(tmp_path/'built'),'--reservations',str(reservations)],check=True,capture_output=True)
    ir=next((tmp_path/'built/fixture').glob('*.ir.jsonl'))
    case=json.loads(ir.read_text());children=case['curriculum']['reference']['children']
    assert len({c['match'] for c in children})==2
    assert case['source_ids']==[r['id'] for r in rows]
    subprocess.run([node,str(root/'ts-host/scripts/inline-curriculum/replay-port-record-calls.mjs'),str(ir),str(tmp_path/'replayed')],check=True,capture_output=True)
    summary=json.loads((tmp_path/'replayed.summary.json').read_text())
    assert summary['admitted']==1 and not summary['omitted']
    row=json.loads((tmp_path/'replayed.rows.jsonl').read_text())
    assert json.loads(row['outcome']['files']['answers.json'])=={'item-01':'FIRST ANSWER','item-02':'SECOND ANSWER'}
    subprocess.run([node,str(root/'ts-host/scripts/inline-curriculum/build-port-record-calls.mjs'),'--input',str(source),'--out',str(tmp_path/'separate'),'--reservations',str(reservations),'--items','1'],check=True,capture_output=True)
    separate=[json.loads(line) for line in next((tmp_path/'separate/fixture').glob('*.ir.jsonl')).read_text().splitlines()]
    assert len({case['id'] for case in separate})==2
    assert separate[0]['source_ids']==separate[1]['source_ids']

