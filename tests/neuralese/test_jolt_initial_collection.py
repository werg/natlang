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


def test_initial_world_ranks_only_bound_train_reference_geometry():
    from natlang_neuralese.rl.jolt_handoff import initial_training_world
    def row(name,level,root,children):
        return {'id':name,'split':'train','curriculum':{'rl':{'level':level},'reference':{'root':[0]*root,'children':[{'calls':[0]*n} for n in children]}}}
    rows=[row('hard',1,3,[3]),row('hinted',0,3,[3]),row('long',0,4,[3])]
    choice=initial_training_world(rows,'a'*64)
    assert choice['case_id']=='hinted' and choice['reference_actions']==6 and choice['cases_sha256']=='a'*64
    with pytest.raises(ValueError):initial_training_world([dict(rows[1],split='test')],'a'*64)
