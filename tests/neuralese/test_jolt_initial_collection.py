import pytest
from natlang_neuralese.rl.production import initial_capacity_collection


def test_small_initial_round_restores_full_curriculum_only_after_update():
    declaration={'schema':'natlang.initial-whole-graph-collection/1','rows':1,'group':2,'workers':1}
    config={'rows_per_round':32,'group':4,'workers':4,'initial_capacity_collection':declaration}
    initial=initial_capacity_collection(config,{'trained_rounds':0,'round':0})
    assert (initial['rows_per_round'],initial['group'],initial['workers'])==(1,2,1)
    assert initial_capacity_collection(config,{'trained_rounds':0,'round':0,'collection_attempt':2})==initial
    expanded=initial_capacity_collection(config,{'trained_rounds':1,'round':1})
    assert (expanded['rows_per_round'],expanded['group'],expanded['workers'])==(32,4,4)
    with pytest.raises(ValueError):initial_capacity_collection({**config,'initial_capacity_collection':{**declaration,'group':1}}, {})
