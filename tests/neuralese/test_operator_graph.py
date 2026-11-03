"""Learned-operator cases render to native graphs plus text-cut siblings (data/operator_graph.py)."""
import glob
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT)]
LFM3 = glob.glob(str(Path.home() / '.cache/huggingface/hub/models--LiquidAI--LFM3-1.7B/snapshots/*'))
pytestmark = pytest.mark.skipif(not LFM3, reason='LFM3 tokenizer not cached')


@pytest.fixture(scope='module')
def renderer():
    from scripts.neuralese_data.build_operator_graphs import make_renderer
    return make_renderer(32768)


def case(operator, sources, *, consumer='What is the answer?', target='forty-two', label='gold', split='train'):
    from neuralese_data.records import VERSION, license_, seal
    return seal({'version': VERSION, 'id': f'test:{operator}:1', 'family': 'unit', 'task': 'consume', 'sources': sources,
                 'writer': {'instructions': 'Read this for: the answer', 'result_type': 'Neuralese<string>',
                            'context': [{'role': 'user', 'content': 'Fix the bug.'}] if operator == 'compact-history' else []},
                 'consumer': {'context': [{'role': 'user', 'content': consumer}], 'withheld': ['sources']},
                 'target': {'kind': 'text', 'value': target, 'alternatives': []},
                 'outcome': {'label': label, 'checked': 'unit'},
                 'lineage': {'project': 'upstream', 'store': 'unit', 'store_version': None, 'row': 1, 'upstream': None,
                             'upstream_id': None, 'upstream_revision': None, 'teacher': None, 'converter': 'unit',
                             'sha256': '', 'notes': {}},
                 'license': license_('MIT'), 'split': split, 'split_groups': ['unit:1']})


@pytest.mark.parametrize('operator', ['compact-value', 'compact-params', 'compact-history', 'extract'])
def test_every_operator_renders_a_valid_graph_with_bound_spans_and_text_cut(renderer, operator):
    from natlang_neuralese.data.native_graph_schema import parse_graph
    from natlang_neuralese.model.learned_operators import OPERATORS, resolve_spans
    text = 'line of the value with path src/app/main.py:12\n' * 120
    if operator == 'compact-history':
        sources = [{'role': 'trajectory', 'exact_refs': [], 'messages': [
            {'role': 'assistant', 'content': 'Looking at the file. ' * 50},
            {'role': 'user', 'content': 'Output: ' + text}]}]
    elif operator == 'extract':
        sources = [{'role': 'document', 'text': text[:900], 'exact_refs': [],
                    'meta': {'operator_seed': {'text': 'path src/app/main.py:12', 'provenance': 'gold', 'origin': 'unit'}}},
                   {'role': 'document', 'text': text[:700], 'exact_refs': []}]
    else:
        sources = [{'role': 'tool_output', 'text': text, 'exact_refs': []}]
    graph, cut = renderer.render(case(operator, sources), operator, corpus_id='unit', source_row_sha256='0' * 64,
                                 source_ref={})
    example = parse_graph(graph, tokenizer_fingerprint=renderer.fingerprint)
    spans = resolve_spans(graph, example.plan, renderer.bindings)
    assert len(spans) == 2 * len(sources)
    consumer = graph['nodes'][-1]
    assert [edge['source'] for edge in consumer['imports']] == [f'op{i}' for i in range(len(sources))]
    seeds = [span for span in spans if span.kind == 'seed']
    for edge, node, seed in zip(consumer['imports'], graph['nodes'][:-1], seeds):
        assert consumer['token_ids'][edge['at'] - 1] == renderer.role_ids[2]
        assert consumer['token_ids'][edge['at']] == renderer.role_ids[3]
        assert 0 < edge['stop'] - edge['start'] <= OPERATORS[operator].seed_cap
        assert [edge['start'], edge['stop']] == node['producing'][0]
        assert node['token_ids'][edge['start'] - 1] == renderer.role_ids[0]
        assert node['token_ids'][edge['stop']] == renderer.role_ids[1] and edge['stop'] + 1 == len(node['token_ids'])
        # Only the seed and its closing marker are supervised in the producer.
        assert [i for i, w in enumerate(node['target_weights']) if w] == list(range(edge['start'], edge['stop'] + 1))
    conversion = graph['operator_conversion']
    assert len(conversion['seeds']) == len(sources) and conversion['span_tokens'] == [
        edge['stop'] - edge['start'] for edge in consumer['imports']]
    if operator == 'extract':
        assert [seed['provenance'] for seed in conversion['seeds']] == ['gold', 'mechanical:outline']
        assert 'main.py:12' in renderer.tokenizer.decode(
            graph['nodes'][0]['token_ids'][consumer['imports'][0]['start']:consumer['imports'][0]['stop']])
    else:
        assert all(seed['provenance'].startswith('mechanical:') for seed in conversion['seeds'])
    supervised = [token for token, weight in zip(consumer['token_ids'], consumer['target_weights']) if weight]
    assert 'forty-two' in renderer.tokenizer.decode(supervised)
    assert cut['text_cut_of'] == graph['id']
    cut_text = renderer.tokenizer.decode(cut['nodes'][0]['token_ids'])
    assert ('<<cut off:' in cut_text) == (operator != 'extract') and 'forty-two' in cut_text
    from natlang_neuralese.eval.neuralese_ablation import span_report
    report = span_report({graph['id']: conversion})
    assert report['spans'] == len(sources) and report['by_seed_provenance']


def test_teacher_targets_and_wrong_site_counts_are_omitted(renderer):
    from natlang_neuralese.data.operator_graph import OperatorCaseError
    text = 'x' * 3000
    with pytest.raises(OperatorCaseError, match='imitation'):
        renderer.render(case('compact-value', [{'role': 'tool_output', 'text': text, 'exact_refs': []}], label='teacher'),
                        'compact-value', corpus_id='unit', source_row_sha256='0' * 64, source_ref={})
    with pytest.raises(OperatorCaseError, match='site-count'):
        renderer.render(case('compact-value', [{'role': 'tool_output', 'text': text, 'exact_refs': []}] * 2),
                        'compact-value', corpus_id='unit', source_row_sha256='0' * 64, source_ref={})


def test_cut_text_matches_the_runtime_cut():
    from natlang_neuralese.data.operator_graph import cut_text
    text = '\n'.join(f'line {index:04d}' for index in range(1000))
    cut = cut_text(text)
    assert cut.startswith('line 0000') and cut.rstrip().endswith('line 0999')
    assert '<<cut off: ' in cut and f'of {len(text)} characters not shown>>' in cut
    assert cut_text('short') == 'short'


def test_multi_argument_compact_params_compacts_long_arguments_and_shows_short_ones(renderer):
    from natlang_neuralese.data.native_graph_schema import parse_graph
    from natlang_neuralese.model.learned_operators import resolve_spans
    long_a = 'first passage about the river Elbe and its bridges\n' * 120
    long_b = 'second passage about the city archives in Dresden\n' * 110
    short = 'A short note: the archive opened in 1902.'
    sources = [{'role': 'passage', 'text': long_a, 'exact_refs': []},
               {'role': 'passage', 'text': short, 'exact_refs': []},
               {'role': 'passage', 'text': long_b, 'exact_refs': []}]
    graph, cut = renderer.render(case('compact-params', sources), 'compact-params', corpus_id='unit',
                                 source_row_sha256='1' * 64, source_ref={})
    example = parse_graph(graph, tokenizer_fingerprint=renderer.fingerprint)
    assert len(resolve_spans(graph, example.plan, renderer.bindings)) == 4
    assert [node['name'] for node in graph['nodes']] == ['op0', 'op1', 'consumer']
    consumer = renderer.tokenizer.decode(graph['nodes'][-1]['token_ids'])
    assert consumer.count('Compacted argument: ') == 2 and short in consumer and 'river Elbe' not in consumer
    assert consumer.index('Compacted argument: ') < consumer.index(short) < consumer.rindex('Compacted argument: ')
    assert graph['operator_conversion']['inline_source_chars'] == [len(short)]
    cut_consumer = renderer.tokenizer.decode(cut['nodes'][0]['token_ids'])
    assert cut_consumer.count('<<cut off:') == 2 and short in cut_consumer
    from natlang_neuralese.data.operator_graph import OperatorCaseError
    with pytest.raises(OperatorCaseError, match='no-compacted-source'):
        renderer.render(case('compact-params', [sources[1], sources[1]]), 'compact-params', corpus_id='unit',
                        source_row_sha256='2' * 64, source_ref={})
