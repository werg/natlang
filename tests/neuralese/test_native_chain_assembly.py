import hashlib
import json

import pytest

from scripts.neuralese_data import native_chain_assembly as assembly


def test_joint_three_node_chain_preserves_both_import_and_export(monkeypatch):
    class Tokenizer:
        def decode(self, ids, **_): return ''.join({8: 'x', 9: 'y'}[x] for x in ids)
    monkeypatch.setattr(assembly, 'writer_payload', lambda writer, name, sha: {'first': 'y', 'second': 'x', 'third': 'y'}[name])
    monkeypatch.setattr(assembly, 'source_occurrences', lambda *args, **kw: {'reader_occurrence_ids': [args[2] + '-read'], 'reader_json_pointers': ['/messages/0'], 'writer_occurrence_id': args[2] + '-write'})
    ids = ['a', 'b', 'c']
    chain = {'source': 's', 'trajectory': 't', 'split': 'train', 'records': ids, 'edges': 3}
    records = {x: {'id': x, 'split': 'train', 'source_groups': ['g'], 'target': {'content': x},
                   'source_ref': {'trajectory_id': 't'}} for x in ids}
    def edge(w, r, name, body):
        return {'source': 's', 'trajectory': 't', 'split': 'train', 'writer_split': 'train',
                'writer_record_id': w, 'reader_record_id': r, 'write_name': name,
                'body_sha256': hashlib.sha256(body.encode()).hexdigest()}
    edges = [edge('a', 'b', 'first', 'y'), edge('b', 'c', 'second', 'x'), edge('a', 'c', 'third', 'y')]
    sha = lambda e: hashlib.sha256(json.dumps(e, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    reads = lambda e, start, stop: {'start': start, 'stop': stop, 'source_record_id': e['writer_record_id'],
        'source_start': 2 if e['writer_record_id'] == 'a' else 5, 'source_stop': 3 if e['writer_record_id'] == 'a' else 6,
        'edge_sha256': sha(e), 'occurrence_id': e['write_name'] + '-read', 'source_pointer': '/messages/0'}
    rendered = {
        'a': {'token_ids': [1, 10, 9, 11, 2], 'target_weights': [0, 1, 1, 1, 1], 'producing': [[2, 3]],
              'producer_bindings': [{'edge_sha256': sha(edges[0]), 'writer_occurrence_id': 'first-write', 'start': 2, 'stop': 3},
                                    {'edge_sha256': sha(edges[2]), 'writer_occurrence_id': 'third-write', 'start': 2, 'stop': 3}]},
        'b': {'token_ids': [1, 12, 9, 13, 10, 8, 11, 2], 'target_weights': [0, 1, 1, 1, 1, 1, 1, 1],
              'reads': [reads(edges[0], 2, 3)], 'producing': [[5, 6]],
              'producer_bindings': [{'edge_sha256': sha(edges[1]), 'writer_occurrence_id': 'second-write', 'start': 5, 'stop': 6}]},
        'c': {'token_ids': [1, 12, 9, 13, 12, 8, 13, 2], 'target_weights': [0, 1, 1, 1, 1, 1, 1, 1],
              'reads': [reads(edges[2], 2, 3), reads(edges[1], 5, 6)]},
    }
    canonical_sha = lambda value: hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    for rid, row in rendered.items():
        row['source_record_sha256'] = canonical_sha(records[rid])
        row['source_target_sha256'] = canonical_sha(records[rid]['target'])
        row['supervised_suffix_start'] = 1 if rid == 'a' else 4 if rid == 'b' else 7
    graph = assembly.assemble(chain, records, edges, rendered, tokenizer=Tokenizer(), tokenizer_fingerprint='fp',
                              role_ids=(10, 11, 12, 13), source_binding={'records_sha256': 'r'})
    assert graph['nodes'][1]['imports'][0]['at'] == 2
    assert graph['nodes'][1]['producing'] == [[4, 5]]
    assert [item['source'] for item in graph['nodes'][2]['imports']] == ['n0', 'n1']
    assert graph['nodes'][1]['target_weights'][2] == 0
    from natlang_neuralese.train.own_output_supervision import _node_suffixes
    assert _node_suffixes(graph, [node['target_weights'] for node in graph['nodes']]) == [1, 3, 5]
    with pytest.raises(ValueError):
        _node_suffixes({**graph, 'source_pairing': {'pair_id': 'bad'}}, [node['target_weights'] for node in graph['nodes']])
    import torch
    from test_shared_state import tiny
    from natlang_neuralese.data.native_graph_schema import parse_graph
    from natlang_neuralese.model.shared_state import SharedStateModel
    model = SharedStateModel(tiny('lfm'), (10, 11, 12, 13), rank=4)
    plan = parse_graph(graph, tokenizer_fingerprint='fp').plan
    flat_ids = plan.tensor([x for node in plan.invocations for x in node.token_ids], 'cpu')
    initial = model.embed(flat_ids[None]).detach()
    outputs, gradients = [], []
    for reference in (False, True):
        embeddings = initial.clone().requires_grad_()
        result = model(plan, embeddings=embeddings, reference=reference)['h_final']
        outputs.append(result.detach())
        gradients.append(torch.autograd.grad(result[:, plan.offsets[-1]:].square().sum(), embeddings)[0])
    torch.testing.assert_close(outputs[0], outputs[1], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(gradients[0], gradients[1], atol=1e-4, rtol=1e-4)
    assert torch.count_nonzero(gradients[0][0, 2:3]) > 0
    rendered['c']['reads'][0]['occurrence_id'] = 'unwitnessed'
    with pytest.raises(ValueError):
        assembly.assemble(chain, records, edges, rendered, tokenizer=Tokenizer(), tokenizer_fingerprint='fp',
                          role_ids=(10, 11, 12, 13), source_binding={'records_sha256': 'r'})
