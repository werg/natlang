import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('native_chain_sources', Path(__file__).resolve().parents[2] / 'scripts/neuralese_data/native_chain_sources.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_complete_chain_rejects_missing_cross_split_and_reversed_edges():
    chain = {'source': 's', 'trajectory': 't', 'split': 'train', 'records': ['a', 'b', 'c'], 'edges': 2}
    records = {x: {'id': x, 'split': 'train', 'source_ref': {'trajectory_id': 't'}} for x in chain['records']}
    def edge(w, r):
        return {'source': 's', 'trajectory': 't', 'split': 'train', 'writer_split': 'train',
                'writer_record_id': w, 'reader_record_id': r, 'write_name': 'n', 'body_sha256': 'f'*64}
    edges = [edge('a', 'b'), edge('b', 'c')]
    assert len(mod.bind_chain(chain, records, edges)['edge_sha256']) == 2
    with pytest.raises(ValueError): mod.bind_chain(chain, records, edges[:1])
    with pytest.raises(ValueError): mod.bind_chain(chain, records, [edges[0], edge('c', 'b')])
    with pytest.raises(ValueError): mod.bind_chain(chain, records, [edges[0], {**edges[1], 'split': 'test'}])
