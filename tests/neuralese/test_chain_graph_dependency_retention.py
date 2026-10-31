from types import SimpleNamespace

from natlang_neuralese.data import native_graph_schema
from natlang_neuralese.train import native_prefix_packing
from scripts.neuralese_data import build_chain_graphs
from scripts.neuralese_data import build_native_semantic_pairs_e2e
from scripts.neuralese_data.build_chain_graphs import dependency_edge_plan, edge_digest, source_chain_records


def test_loss_deduplication_does_not_remove_a_source_dependency_edge():
    edge = {
        'kind': 'result',
        'writer_record_id': 'trajectory:decision:0',
        'reader_record_id': 'trajectory:decision:1',
        'body_sha256': 'a' * 64,
    }

    assert dependency_edge_plan([edge]) == {edge_digest(edge)}  # Reader loss caps are applied separately.


def test_source_graph_keeps_context_only_and_deduped_decisions_in_order():
    chain = {'records': ['trajectory:decision:0', 'trajectory:decision:1', 'trajectory:decision:2']}

    assert source_chain_records(chain) == chain['records']


def _task():
    edge = {
        'kind': 'result',
        'writer_record_id': 'trajectory:decision:0',
        'reader_record_id': 'trajectory:decision:1',
        'write_name': 'prior',
        'body_sha256': 'a' * 64,
    }
    chain = {
        'trajectory': 'teacher-program:one',
        'source': 'fixture',
        'split': 'test',
        'records': [edge['writer_record_id'], edge['reader_record_id']],
    }
    records = {rid: {'id': rid, 'family': 'test', 'source_ids': []} for rid in chain['records']}
    return chain, records, [edge]


def test_conversion_keeps_dependency_for_deduped_reader(monkeypatch):
    chain, records, edges = _task()
    seen = {}
    monkeypatch.setattr(build_chain_graphs, 'STATE', {
        'fingerprint': 'fixture', 'prompts': ['program-header'], 'max_tokens': 100,
        'serving_entry': 'mix', 'reader_cap': 64, 'target_cap': 64,
    })
    monkeypatch.setattr(build_chain_graphs, 'read_records', lambda _locations: records)
    monkeypatch.setattr(build_chain_graphs, 'pieces_for', lambda _path: [])
    monkeypatch.setattr(build_native_semantic_pairs_e2e, 'writer_payload', lambda *_args: 'body')
    monkeypatch.setattr(build_native_semantic_pairs_e2e, 'source_occurrences', lambda *_args, **_kwargs: {})

    def render(subchain, subrecords, subedges, _pieces, _prompt):
        seen['records'] = list(subchain['records'])
        seen['edges'] = list(subedges)
        return {'nodes': [
            {'name': f'n{i}', 'token_ids': [1], 'target_weights': [1.0], 'producing': [], 'imports': []}
            for i, _rid in enumerate(subchain['records'])
        ]}

    monkeypatch.setattr(build_chain_graphs, 'render', render)
    monkeypatch.setattr(native_graph_schema, 'parse_graph', lambda graph, **_kwargs: SimpleNamespace(plan=SimpleNamespace(token_count=2)))
    monkeypatch.setattr(native_graph_schema, 'communication_classification', lambda _graph: {'producer_consumer': True})
    monkeypatch.setattr(native_prefix_packing, 'pack_causal_prefixes', lambda _graphs, **_kwargs: None)
    plan = {'edges': dependency_edge_plan(edges), 'supervised': {chain['records'][0]}}

    kind, _split, result = build_chain_graphs.convert_unit((chain, {}, edges, 'pieces', plan))

    assert seen == {'records': chain['records'], 'edges': edges}
    assert kind == 'graph'
    assert result['graph']['chain_conversion']['dependency_retention'] == 'all-source-edges-authenticated'
    assert result['graph']['nodes'][1]['target_weights'] == [0.0]


def test_conversion_omits_whole_chain_when_dependency_authentication_fails(monkeypatch):
    chain, records, edges = _task()
    monkeypatch.setattr(build_chain_graphs, 'STATE', {'current_worlds': None})
    monkeypatch.setattr(build_chain_graphs, 'read_records', lambda _locations: records)
    monkeypatch.setattr(build_native_semantic_pairs_e2e, 'writer_payload', lambda *_args: 'body')

    def fail_auth(*_args, **_kwargs):
        raise ValueError('fixture source pointer mismatch')

    monkeypatch.setattr(build_native_semantic_pairs_e2e, 'source_occurrences', fail_auth)
    plan = {'edges': dependency_edge_plan(edges), 'supervised': set(chain['records'])}

    kind, split, result = build_chain_graphs.convert_unit((chain, {}, edges, 'pieces', plan))

    assert (kind, split) == ('omission', 'test')
    assert result['reason'] == 'source-edge-unauthenticated'

