import copy
import json

import pytest

from natlang_neuralese.common.hashing import sha256_file_hex_cached as sha
from natlang_neuralese.train.native_pending_sources import declaration, pending_source_dependencies
from natlang_neuralese.train.recipe import (_pending_source_continuation, _data_continuation,
    unstarted_stage_input_changes, _without_unstarted_bindings)
from test_native_state_sft import graph


def declared(tmp_path, name, ids, *, evidence=(), mutate=None):
    graphs=tmp_path/(name+'.jsonl')
    graphs.write_text(''.join(json.dumps(dict(graph(i),tokenizer_fingerprint='1'*64))+'\n' for i in ids))
    value=declaration(graphs,graphs,tokenizer_fingerprint='1'*64,role_ids=(58,59,60,61),evidence=evidence)
    if mutate:mutate(value)
    path=tmp_path/(name+'.json');path.write_text(json.dumps(value))
    return {'path':str(path),'sha256':sha(path),'binding':'pending'}


def inputs():
    return {'native_graphs':{'path':'train','sha256':'a'*64,'binding':'train'},
        'continued_native_graphs':{'path':'previous','sha256':'b'*64,'binding':'previous'},
        'held_native_graphs':{'path':'held','sha256':'c'*64,'binding':'held'},
        'foundation_native_graphs':{'path':'foundation','sha256':'d'*64,'binding':'foundation'}}


def test_new_pending_obligations_keep_exact_active_sources_and_historical_predecessor(tmp_path):
    old=inputs();new={**old,'pending_native_sources':declared(tmp_path,'new',['long'])}
    assert _pending_source_continuation(old,new)
    assert not _data_continuation(old,new)  # Resident scope is deliberately unchanged.
    assert not _pending_source_continuation(new,new)
    assert not _pending_source_continuation(new,old)


@pytest.mark.parametrize('role',['native_graphs','held_native_graphs','foundation_native_graphs','continued_native_graphs'])
def test_pending_handoff_cannot_change_other_inputs(tmp_path,role):
    old=inputs();new={**copy.deepcopy(old),'pending_native_sources':declared(tmp_path,'new',['long'])}
    new[role]['sha256']='e'*64
    assert not _pending_source_continuation(old,new)


def test_expansion_retains_old_rows_quotas_and_dependency_evidence(tmp_path):
    prior=declared(tmp_path,'prior',['long']);old={**inputs(),'pending_native_sources':prior}
    evidence=pending_source_dependencies(prior['path'])
    added=declared(tmp_path,'added',['long','later'],evidence=evidence)
    # The shared semantic binding is independent of the leaf filename.
    assert _pending_source_continuation(old,{**inputs(),'pending_native_sources':added})
    reduced=declared(tmp_path,'reduced',['later'],evidence=evidence)
    assert not _pending_source_continuation(old,{**inputs(),'pending_native_sources':reduced})
    changed=declared(tmp_path,'changed',['long','later'],evidence=evidence,
        mutate=lambda v:v['sources'][0]['rows'][0].update(required_visits=2))
    assert not _pending_source_continuation(old,{**inputs(),'pending_native_sources':changed})
    missing=declared(tmp_path,'missing',['long','later'])
    assert not _pending_source_continuation(old,{**inputs(),'pending_native_sources':missing})


def test_corrupt_declaration_and_dependency_fail_before_handoff(tmp_path):
    pending=declared(tmp_path,'new',['long']);new={**inputs(),'pending_native_sources':pending}
    path=tmp_path/'new.json';original=path.read_text();path.write_text(original+'\n')
    with pytest.raises(ValueError,match='declaration changed'):_pending_source_continuation(inputs(),new)
    path.write_text(original);(tmp_path/'new.jsonl').write_text('changed')
    with pytest.raises(ValueError,match='immutable binding'):_pending_source_continuation(inputs(),new)


def test_started_recipe_uses_existing_data_continuation_projection(tmp_path):
    old=inputs();pending=declared(tmp_path,'pending',['long']);new={**old,'pending_native_sources':pending}
    def plan(resolved):
        names={k:v['binding'] for k,v in resolved.items()}
        return {'recipe':{'stages':[{'id':'full','kind':'native_state_sft','inputs':names,'parameters':{}}],
                'input_bindings':{v['binding']:{k:v[k] for k in ('path','sha256')} for v in resolved.values()}},
            'stage_inputs':{'full':resolved}}
    before,after=plan(old),plan(new);stage=tmp_path/'full';stage.mkdir();(stage/'checkpoint.pt').write_text('existing stage marker')
    changes=unstarted_stage_input_changes(before,after,tmp_path)
    assert changes['full']['kind']=='data_continuation'
    assert _without_unstarted_bindings(before['recipe'],after['recipe'],changes)[0]==\
        _without_unstarted_bindings(before['recipe'],after['recipe'],changes)[1]
