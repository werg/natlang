"""subagent_neuralese: the source handoffs and the Neuralese read of a written report (no model, no tokenizer)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'scripts/neuralese_data'), str(ROOT / 'training/neuralese'), str(ROOT)]

import subagent_neuralese as s  # noqa: E402


def test_split_reply_reads_reasoning_text_and_calls():
    reasoning, text, calls = s.split_reply('<think>\nplan\n</think>\n\nok <tool_call>\n{"name": "search", '
                                           '"arguments": {"q": 1}}\n</tool_call>')
    assert (reasoning, text, calls) == ('plan', 'ok', [{'name': 'search', 'arguments': {'q': 1}}])
    assert s.split_reply('<think>x</think><tool_call>{not json}</tool_call>') is None


def test_wideseek_handoffs_are_the_report_spans_without_the_turn_note():
    text = ('# Subtask 1:\nfind a\n# Result:\nA is 1.\n\n# Subtask 2:\nfind b\n# Result:\nB is 2.\n\n'
            'Your next answer will be on turn 2. You MUST finish the entire answer by turn 10.')
    found = s.wideseek_handoffs(text)
    assert [(b, text[lo:hi]) for b, lo, hi in found] == [('find a', 'A is 1.'), ('find b', 'B is 2.')]


def test_swarm_handoffs_split_on_the_separator_and_keep_the_report_whole():
    text = ('A sub-agent dispatched for goal "g1" returned the following report:\n\n## R1\nx\n---\ny'
            '\n\n---\n\nA sub-agent dispatched for goal "g2" returned the following report:\n\n## R2\n')
    found = s.swarm_handoffs(text)
    assert [(g, text[lo:hi]) for g, lo, hi in found] == [('g1', '## R1\nx\n---\ny'), ('g2', '## R2')]


def test_swarm_tool_block_becomes_structured_tools():
    tool = {'type': 'function', 'function': {'name': 'search', 'parameters': {}}}
    system = ('Be careful.\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n'
              'You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n' + json.dumps(tool) +
              '\n</tools>\n\nFor each function call, return a json object with function name and arguments within '
              '<tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}'
              '\n</tool_call>')
    assert s.swarm_tools(system) == ('Be careful.', [tool])
    assert s.swarm_tools('no tools here') == ('no tools here', None)


def test_handoff_messages_put_each_report_in_a_native_state_part():
    text = '# Subtask 1:\nfind a\n# Result:\nA is 1.\n\nnote'
    lo = text.index('A is')
    lead = {'messages': [{'role': 'user', 'content': 'q'}, {'role': 'tool', 'content': text}],
            'handoffs': [(1, lo, lo + len('A is 1.'), 'find a')]}
    messages, bodies = s.handoff_messages(lead)
    parts = messages[1]['content']
    assert [p['type'] for p in parts] == ['text', 'native_state', 'text']
    assert bodies == {parts[1]['id']: 'A is 1.'} and parts[2]['text'] == '\n\nnote'
    assert parts[1]['id'].startswith('nzs1_') and len(parts[1]['id']) == 57


def test_read_splices_the_written_ids_with_a_pass_through_producer_document():
    prompt, imports, sources = [1, 2], [], {}
    s.read(prompt, imports, sources, 'nzs1_x', [7, 8, 9], 'splice', (100, 101, 102))
    assert prompt == [1, 2, 100, 7, 8, 9, 101] and imports == [(3, 'nzs1_x')]
    doc = sources['nzs1_x']
    assert list(doc['ids']) == [102, 7, 8, 9] and (doc['start'], doc['stop']) == (1, 4)
    # the placeholder rows are the producer span's own ids, as an actor's import records them
    assert prompt[3:6] == list(doc['ids'][doc['start']:doc['stop']])
    inline, none, nosrc = [1], [], {}
    s.read(inline, none, nosrc, 'nzs1_x', [7, 8], 'inline', (100, 101, 102))
    assert inline == [1, 7, 8] and none == [] and nosrc == {}


def test_held_content_check_and_answer_containment():
    held = ({'what is the capital of france'}, {' '.join(f'w{i}' for i in range(10))})
    assert s.held_content_hit('What is the capital of France?', held) == 'question'
    assert s.held_content_hit(' '.join(f'w{i}' for i in range(12)), held) == 'shingles'
    assert s.held_content_hit('Who wrote Hamlet and when was it first staged?', held) is None
    assert s.qa_norm('The Marie-Soleil Tougas') in s.qa_norm('Answer: marie soleil tougas.')
