"""OpenHands trajectories as pi transcripts (plans/neuralese/HARNESS_BENCH.md §1.2)."""
import json

from natlang_neuralese.harness_bench.openhands import normalize

CWD = '/workspace/acme__lib__1.0'


def call(id_, name, **args):
    return {'id': id_, 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}}


ROW = {
    'trajectory_id': 't1', 'instance_id': 'acme__lib-7', 'repo': 'acme/lib', 'resolved': 1, 'model_patch': 'diff',
    'trajectory': [
        {'role': 'system', 'content': 'You are OpenHands agent.'},
        {'role': 'user', 'content': f'<uploaded_files>\n{CWD}\n</uploaded_files>\nConsider the following issue:\n'
                                    '<issue_description>\nParsing fails on empty input\n</issue_description>\nFollow these phases...'},
        {'role': 'assistant', 'content': 'Looking.', 'tool_calls': [call('a', 'think', thought='plan it'),
                                                                    call('b', 'str_replace_editor', command='view', path='/workspace')]},
        {'role': 'tool', 'tool_call_id': 'a', 'name': 'think', 'content': 'Your thought has been logged.'},
        {'role': 'tool', 'tool_call_id': 'b', 'name': 'str_replace_editor', 'content': 'files...'},
        {'role': 'assistant', 'content': '', 'tool_calls': [call('c', 'str_replace_editor', command='view', path=f'{CWD}/lib/parse.py', view_range=[3, 9])]},
        {'role': 'tool', 'tool_call_id': 'c', 'name': 'str_replace_editor', 'content': '3\tdef parse(s):'},
        {'role': 'assistant', 'content': '', 'tool_calls': [call('d', 'str_replace_editor', command='str_replace', path=f'{CWD}/lib/parse.py',
                                                                 old_str='s.split()', new_str='(s or "").split()')]},
        {'role': 'tool', 'tool_call_id': 'd', 'name': 'str_replace_editor', 'content': 'The file has been edited.'},
        {'role': 'assistant', 'content': '', 'tool_calls': [call('e', 'execute_bash', command='pytest -q', timeout=120),
                                                            call('f', 'task_tracker', command='view')]},
        {'role': 'tool', 'tool_call_id': 'e', 'name': 'execute_bash', 'content': '3 passed'},
        {'role': 'tool', 'tool_call_id': 'f', 'name': 'task_tracker', 'content': 'no tasks'},
        {'role': 'assistant', 'content': '', 'tool_calls': [call('g', 'finish', message='Fixed empty input.')]},
        {'role': 'tool', 'tool_call_id': 'g', 'name': 'finish', 'content': 'done'},
    ],
}


def test_pi_mapping_uses_pi_tools_strips_the_harness_and_lists_lossy_steps():
    item = normalize(ROW, 'pi')
    assert item.cwd == CWD and item.goal == 'Parsing fails on empty input' and item.resolved
    messages = item.messages
    assert messages[0] == {'role': 'user', 'content': 'Parsing fails on empty input'}
    first = messages[1]['content']
    assert first[0] == {'type': 'text', 'text': 'Looking.'}
    assert first[1] == {'type': 'thinking', 'thinking': 'plan it'}
    assert first[2]['name'] == 'bash' and first[2]['arguments'] == {'command': "find /workspace -maxdepth 2 -not -path '*/.*'"}
    # The think result is dropped; the directory view's result stays, under the pi tool's name.
    assert [m['toolName'] for m in messages if m['role'] == 'toolResult'] == ['bash', 'read', 'edit', 'bash', 'task_tracker']
    calls = [part for m in messages if m['role'] == 'assistant' for part in m['content'] if part['type'] == 'toolCall']
    assert calls[1]['arguments'] == {'path': 'lib/parse.py', 'offset': 3, 'limit': 7}
    assert calls[2]['arguments'] == {'path': 'lib/parse.py', 'edits': [{'oldText': 's.split()', 'newText': '(s or "").split()'}]}
    assert calls[3]['arguments'] == {'command': 'pytest -q', 'timeout': 120}
    assert item.lossy == [{'message': 7, 'step': 9, 'reason': 'task_tracker (no pi tool)'}]
    assert messages[-1] == {'role': 'assistant', 'content': [{'type': 'text', 'text': 'Fixed empty input.'}], 'stopReason': 'stop'}


def test_native_mapping_keeps_the_teacher_tools_with_relative_paths():
    item = normalize(ROW, 'native')
    calls = [part for m in item.messages if m['role'] == 'assistant' for part in m['content'] if part['type'] == 'toolCall']
    assert [c['name'] for c in calls] == ['think', 'str_replace_editor', 'str_replace_editor', 'str_replace_editor', 'execute_bash', 'task_tracker', 'finish']
    assert calls[2]['arguments']['path'] == 'lib/parse.py' and item.lossy == []
