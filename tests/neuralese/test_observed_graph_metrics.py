import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/neuralese_data'))
from observed_graph_metrics import graph_metrics, aggregate_graph_metrics


def node(ident, parent=None, source='skills/editor/review.nl'):
    return {'call_id': ident, 'parent_call_id': parent, 'program_root': 'skills/editor',
            'definition': {'name': 'review', 'source': source}}


def test_actual_graph_composition_counts_calls_not_only_unique_functions():
    actual = graph_metrics([node('root'), node('one', 'root'), node('two', 'root', 'skills/browser/inspect.nl'), node('leaf', 'one')])
    assert actual['max_depth'] == 2
    assert actual['max_branching'] == 2
    assert len(actual['unique_function_paths']) == 2
    assert actual['domain_composition'] == {'browser': 1, 'editor': 3}
    assert actual['cross_domain_edges'] == 1
    assert sum(edge['count'] for edge in actual['parent_child_edges']) == 3
    summary = aggregate_graph_metrics([{'category': category, 'metrics': actual} for category in ['completed', 'partial', 'preserved_failure']])
    assert all(bucket['captures'] == 1 for bucket in summary['categories'].values())


def test_missing_ancestry_and_cycles_remain_visible_without_fabricated_depth():
    actual = graph_metrics([node('root'), node('orphan', 'absent'), node('cycle-a', 'cycle-b'), node('cycle-b', 'cycle-a'), node('root')])
    assert actual['max_depth'] == 0
    assert actual['anomalies'] == {'duplicate_call_ids': 1, 'orphan_call_ids': ['orphan'],
                                   'cycle_call_ids': ['cycle-a', 'cycle-b'], 'unresolved_depth_calls': 3}
    assert graph_metrics([])['max_depth'] is None
