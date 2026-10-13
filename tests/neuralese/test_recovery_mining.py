"""Recovery supervision source 1: the exact no-progress classifier on the dragnet loop cases
(plans/neuralese/REPETITION_DRAGNET_20261013.md) and the scanner's selection and held-split guard."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'training' / 'neuralese'))
sys.path.insert(0, str(ROOT / 'scripts' / 'neuralese_data'))
from natlang_neuralese.train.recovery_mining import (  # noqa: E402
    is_empty_read, no_progress_rules, normalise_observation, parse_turns, recovery_rules)
import mine_recovery  # noqa: E402


def call(code, think='x'):
    return f'<think>\n{think}\n</think>\n\n<tool_call>\n' + json.dumps({'name': 'eval', 'arguments': {'code': code}}) + '\n</tool_call>'


def doc(*pairs):
    """pairs of (action message, observation or None)."""
    out = '<|im_start|>user\nYou are inside this call: f()<|im_end|>\n'
    for action, obs in pairs:
        out += f'<|im_start|>assistant\n{action}<|im_end|>\n'
        if obs is not None:
            out += f'<|im_start|>user\n<tool_response>\n{obs}\n</tool_response><|im_end|>\n'
    return out


def rules(text, index):
    return recovery_rules(parse_turns(text), index)


def test_multihop_empty_read_then_other_direction():
    out = call('return graph.outgoing(parent, "owns", 1);')
    fix = call('return graph.incoming(parent, "owned_by", 1);')
    assert rules(doc((out, '[]'), (fix, None)), 1) == ['empty']
    # the same call again is not a recovery
    assert rules(doc((out, '[]'), (out, None)), 1) == []


def test_route_replan_identical_look_after_stored_line_drops():
    look = call('const w = world.look(); return w;')
    first = '{ room: "hall", exits: ["n"] }\nStored local w = { room: "hall", exits: ["n"] }.'
    rerun = '{ room: "hall", exits: ["n"] }'
    move = call('return world.move("n");')
    text = doc((look, first), (look, rerun), (move, None))
    assert rules(text, 1) == []              # observation 0 is the first one: progress
    assert rules(text, 2) == ['identical']   # observation 1 repeats observation 0 modulo the Stored line


def test_late_argmax_page_count_console_null():
    pages = call('console.log("pages:", page_count()); let p = page_count();')
    first = 'console:\npages: 3\nnull\nStored local p = 3.'
    again = 'console:\npages: 3\nnull'
    other = call('return page(1);')
    text = doc((pages, first), (pages, again), (other, None))
    assert rules(text, 2) == ['identical']
    assert not is_empty_read(first)          # console output is information, not an empty read


def test_entailment_page_identical_values():
    page = call('return facts.page(1);')
    obs = '[{ id: "F1" }, { id: "F2" }]'
    text = doc((page, obs), (page, obs), (call('return facts.page(2);'), None))
    assert rules(text, 2) == ['identical']
    # identical observation of a different action is not a repeat
    text = doc((page, obs), (call('return facts.all();'), obs), (call('return 1;'), None))
    assert rules(text, 2) == []


def test_frontier_graph_links_failed_form():
    links = call('return graph.links;')
    err = 'TypeError: graph.links is not a function.\nNothing else from this eval was kept.'
    text = doc((links, err), (links, err + '\nThis is the code of your previous eval, and it failed the same way: change it before running it again.'),
               (call('return graph.neighbors(node);'), None))
    assert rules(text, 1) == []              # same action again: not a recovery
    assert rules(text, 2) == ['failed']
    assert no_progress_rules(parse_turns(text), 0) == ['failed']
    rejected = doc((call('x'), 'rejected status: bad-action'), (call('y'), None))
    assert rules(rejected, 1) == ['failed']
    refused = doc((call('x'), 'value is bad, so it is not the result'), (call('y'), None))
    assert rules(refused, 1) == ['failed']


def test_empty_forms_and_normalisation():
    for body in ('[]', 'null', '""', '{}', '[]\nStored local xs = [].'):
        assert is_empty_read(body), body
    for body in ('ok', '[1]', '"a"', 'undefined', 'console:\nnull'):
        assert not is_empty_read(body), body
    assert normalise_observation('1\nStored local a = 1; local b = 2.\n') == '1'


def test_action_comparison_ignores_whitespace_and_think():
    a = call('return  graph.outgoing(p);', think='first')
    b = call('return graph.outgoing(p);', think='second').replace('"code": ', '"code":  ')
    assert parse_turns(doc((a, '[]'), (b, None)))[0].action == parse_turns(doc((b, None)))[0].action
    assert rules(doc((a, '[]'), (b, None)), 1) == []


def test_scanner_rules_and_held_group_guard(tmp_path):
    look = call('return a.b();')
    fix = call('return a.c();')
    def row(i, split, groups, rendering='program-header'):
        return {'id': i, 'split': split, 'source_groups': groups, 'corpus': 'c', 'text': doc((look, '[]'), (fix, None))}
    ph = tmp_path / 'ph.jsonl'
    ph.write_text('\n'.join(json.dumps(r) for r in [
        row('a', 'train', ['g1', 'inline-curriculum:fam:x']), row('b', 'train', ['g2']),
        row('c', 'test', ['g2']), row('d', 'train', ['g3'])]) + '\n')
    out = tmp_path / 'out'
    sys.argv = ['x', '--out-dir', str(out), '--input', f'program-header={ph}']
    mine_recovery.main()
    selected = [json.loads(l) for l in (out / 'recovery-selection.jsonl').read_text().splitlines()]
    assert sorted((s['id'], s['split']) for s in selected) == [('a', 'train'), ('c', 'test'), ('d', 'train')]
    assert {s['id']: s['family'] for s in selected}['a'] == 'fam'
    summary = json.loads((out / 'summary.json').read_text())
    assert summary['dropped_train_shared_held_group'] == {'program-header': 1}
    assert summary['by_rule'] == {'empty': 3}
