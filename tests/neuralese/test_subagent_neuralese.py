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
    bundle = {'lead': {'messages': [{'role': 'user', 'content': 'q'}, {'role': 'tool', 'content': text}],
                       'handoffs': [(1, lo, lo + len('A is 1.'), 'find a')]}, 'subagents': []}
    messages, bodies = s.handoff_messages(None, bundle)
    parts = messages[1]['content']
    assert [p['type'] for p in parts] == ['text', 'native_state', 'text']
    assert bodies == {parts[1]['id']: 'A is 1.'} and parts[2]['text'] == '\n\nnote'
    assert parts[1]['id'].startswith('nzs1_') and len(parts[1]['id']) == 57


def test_read_splices_the_written_ids_with_their_producer_document():
    import numpy as np
    prompt, imports, sources = [1, 2], [], {}
    document = {'ids': np.array([5, 6, 102, 7, 8, 9], np.int32), 'start': 3, 'stop': 6}
    s.read(prompt, imports, sources, 'nzs1_x', [7, 8, 9], 'splice', (100, 101), document)
    assert prompt == [1, 2, 100, 7, 8, 9, 101] and imports == [(3, 'nzs1_x')] and sources['nzs1_x'] is document
    # the placeholder rows are the producer span's own ids, as an actor's import records them
    assert prompt[3:6] == list(document['ids'][document['start']:document['stop']])
    inline, none, nosrc = [1], [], {}
    s.read(inline, none, nosrc, 'nzs1_x', [7, 8], 'inline', (100, 101), document)
    assert inline == [1, 7, 8] and none == [] and nosrc == {}


THINK_OPEN, THINK_CLOSE, IM_END, PRODUCE_OPEN = 50, 51, 52, 102


def _lowering(monkeypatch):
    """A Lowering whose renderer is a stub: one token per message (11 + its index), the generation prompt ends with
    <think>, and a reply renders as reasoning 60, </think>, content 70, <|im_end|>."""
    import collections
    import types
    from natlang_neuralese.serve import targets
    names = {'<think>': THINK_OPEN, '</think>': THINK_CLOSE}
    lowering = object.__new__(s.Lowering)
    lowering.renderer = types.SimpleNamespace(tokenizer=types.SimpleNamespace(convert_tokens_to_ids=names.get))
    lowering.produce_open, lowering.max_producer_tokens = PRODUCE_OPEN, 8192
    seen = []

    def target_items(_renderer, messages, tools, target):
        seen.append([m.get('content') for m in messages])
        before = [('tok', 1)] + [('tok', 11 + k) for k, _ in enumerate(messages)] + [('tok', THINK_OPEN)]
        rest = [('tok', 60), ('tok', THINK_CLOSE), ('tok', 70), ('tok', IM_END)] if target.get('content') else [('tok', IM_END)]
        return before, rest
    monkeypatch.setattr(targets, 'target_items', target_items)
    return lowering, seen, collections.Counter()


def test_a_report_is_written_at_the_end_of_its_subagents_own_conversation(monkeypatch):
    """The producer document (DECISIONS 2026-11-25): the subagent's final call as it saw it, its final reply's
    reasoning, produce_open, the report as the span; never the report alone."""
    lowering, seen, stats = _lowering(monkeypatch)
    sub = {'messages': [{'role': 'system', 'content': 'sys'}, {'role': 'user', 'content': 'brief'},
                        {'role': 'assistant', 'content': 'search'}, {'role': 'tool', 'content': 'result'},
                        {'role': 'assistant', 'content': 'search more'}, {'role': 'tool', 'content': 'result 2'},
                        {'role': 'assistant', 'content': 'the report'}], 'tools': None}
    doc = lowering.producer_document([7, 8], sub=sub, stats=stats)
    prompt = [1, 11, 12, 13, 14, 15, 16, THINK_OPEN]
    assert list(doc['ids']) == prompt + [60, THINK_CLOSE, PRODUCE_OPEN, 7, 8] and (doc['start'], doc['stop']) == (11, 13)
    assert seen[-1] == ['sys', 'brief', 'search', 'result', 'search more', 'result 2'] and stats['producer_full'] == 1
    # over the budget: the opening and the latest exchange (the runtime's collapseHistory), then the opening alone
    doc = lowering.producer_document([7, 8], sub=sub, max_tokens=12, stats=stats)
    assert seen[-1] == ['sys', 'brief', 'search more', 'result 2'] and stats['producer_compacted'] == 1
    assert list(doc['ids'][-3:]) == [PRODUCE_OPEN, 7, 8]
    lowering.producer_document([7, 8], sub=sub, max_tokens=3, stats=stats)
    assert seen[-1] == ['sys', 'brief'] and stats['producer_opening'] == 1
    # a subagent the source did not keep: its opening frame, an empty answer opened and closed, then the report
    doc = lowering.producer_document([7, 8], frame=([{'role': 'user', 'content': 'brief'}], None), stats=stats)
    assert list(doc['ids']) == [1, 11, THINK_OPEN, THINK_CLOSE, PRODUCE_OPEN, 7, 8] and stats['producer_frame'] == 1


def test_handoffs_find_their_subagent_and_unrecorded_ones_a_siblings_opening(monkeypatch):
    lowering, seen, stats = _lowering(monkeypatch)
    lowering.written = lambda text: [7] * len(text.split())
    text = 'R1: a b\n\nR2: c'
    a, b = text.index('a b'), text.index('c')
    sub = {'messages': [{'role': 'system', 'content': 'sys'}, {'role': 'user', 'content': 'Task: find a'},
                        {'role': 'assistant', 'content': 'a b'}], 'tools': [{'name': 'search'}]}
    bundle = {'lead': {'messages': [{'role': 'user', 'content': 'q'}, {'role': 'tool', 'content': text}],
                       'handoffs': [(1, a, a + 3, 'find a'), (1, b, b + 1, 'find c')]}, 'subagents': [sub]}
    assert s.producers(bundle) == [0, None]
    assert s.opening_frame(bundle, 'find c') == ([{'role': 'system', 'content': 'sys'},
                                                 {'role': 'user', 'content': 'Task: find c'}], [{'name': 'search'}])
    messages, bodies = s.handoff_messages(lowering, bundle, stats)
    states = [p['id'] for p in messages[1]['content'] if p['type'] == 'native_state']
    assert [bodies[k]['text'] for k in states] == ['a b', 'c'] and stats['producer_full'] == stats['producer_frame'] == 1
    assert seen[-1] == ['sys', 'Task: find c']
    # one report text written by two producers is two states (the id names the producing context)
    assert len({s.state_id('c'), states[1]}) == 2


def test_held_content_check_and_answer_containment():
    held = ({'what is the capital of france'}, {' '.join(f'w{i}' for i in range(10))})
    assert s.held_content_hit('What is the capital of France?', held) == 'question'
    assert s.held_content_hit(' '.join(f'w{i}' for i in range(12)), held) == 'shingles'
    assert s.held_content_hit('Who wrote Hamlet and when was it first staged?', held) is None
    assert s.qa_norm('The Marie-Soleil Tougas') in s.qa_norm('Answer: marie soleil tougas.')
