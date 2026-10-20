import copy
import json

import pytest

from scripts.neuralese_data.build_native_ce_supervision_policy import build
from natlang_neuralese.common.hashing import sha256_file_hex
from natlang_neuralese.data.native_graph_schema import parse_graph
from natlang_neuralese.train.native_supervision import apply_policy, load_policy


def native(ident='direct-native'):
    return {'schema':'natlang.neuralese-native-graph/1','id':ident,'split':'train',
        'source_groups':['train-only-native'],'tokenizer_fingerprint':'f'*64,
        'nodes':[{'name':'producer','token_ids':[1,2,3], 'target_weights':[0.,0.,1.],
                  'producing':[[1,3]]},
                 {'name':'consumer','token_ids':[4,5,6], 'target_weights':[0.,0.,2.],
                  'imports':[{'at':1,'source':'producer','start':1,'stop':3}]}]}


def source(tmp_path, rows):
    path=tmp_path/'source.jsonl'
    path.write_text(''.join(json.dumps(row)+'\n'for row in rows))
    return path


def test_direct_native_masks_are_source_bound_and_unchanged_at_runtime(tmp_path):
    row=native();before=copy.deepcopy(row);path=source(tmp_path,[row])
    policy=build([path],legacy_pair_context_weight=.05)
    assert policy['sources']==[]
    assert len(policy['retained_sources'])==1
    assert policy['retained_sources'][0]['sha256']==sha256_file_hex(path)
    assert 'explicit native node target masks' in policy['retained_sources'][0]['reason']
    example=parse_graph(row,tokenizer_fingerprint='f'*64)
    assert apply_policy(example,row,source_sha256=sha256_file_hex(path),policy=policy) is example
    assert row==before and example.target_weights==(0.,0.,1.,0.,0.,2.)


def test_semantic_pair_model_views_retain_direct_native_objectives(tmp_path):
    row={'schema':'natlang.native-semantic-pair/1','split':'train',
         'models':{model:{'graph':native(model)}for model in ('lfm25','mellum21')}}
    path=source(tmp_path,[row]);policy=build([path],legacy_pair_context_weight=.05)
    assert policy['sources']==[] and policy['retained_sources'][0]['sha256']==sha256_file_hex(path)


@pytest.mark.parametrize('mutation',[
    lambda r:r['nodes'][0].pop('target_weights'),
    lambda r:r['nodes'][0].update(target_weights=[0.,1.]),
    lambda r:r['nodes'][0].update(target_weights=[0.,float('nan'),1.]),
    lambda r:r.update(nodes=[]),
    lambda r:r.update(conversion={'role_interpretation':'ordinary-text'}),
    lambda r:r.update(conversion={'source_suffix_start':2}),
    lambda r:r.update(source_pairing={}),
])
def test_missing_masks_and_malformed_eligible_markers_still_reject(tmp_path,mutation):
    row=native();mutation(row);path=source(tmp_path,[row])
    with pytest.raises(ValueError):build([path],legacy_pair_context_weight=.05)


def test_incompatible_model_view_encoding_cannot_be_hidden_as_retained(tmp_path):
    direct=native('direct');ordinary=native('ordinary')
    ordinary['conversion']={'role_interpretation':'ordinary-text','source_suffix_start':2,'context_weight':.05}
    path=source(tmp_path,[{'schema':'natlang.native-semantic-pair/1','split':'train',
         'models':{'lfm25':{'graph':direct},'mellum21':{'graph':ordinary}}}])
    with pytest.raises(ValueError,match='incompatible model-view'):
        build([path],legacy_pair_context_weight=.05)


def test_later_eligible_row_still_requires_runtime_declaration(tmp_path):
    direct=native('direct');later=native('eligible')
    later['conversion']={'role_interpretation':'ordinary-text','source_suffix_start':2,'context_weight':.05}
    path=source(tmp_path,[direct,later]);policy=build([path],legacy_pair_context_weight=.05)
    example=parse_graph(later,tokenizer_fingerprint='f'*64)
    with pytest.raises(ValueError,match='eligible native CE source missing'):
        apply_policy(example,later,source_sha256=sha256_file_hex(path),policy=policy)


def test_mixed_eligible_and_direct_sources_seal_and_load_without_reweighting_native(tmp_path):
    from scripts.neuralese_data.build_native_ce_supervision_policy import main
    direct=native();path=source(tmp_path,[direct])
    ordinary=native('ordinary');ordinary['nodes']=ordinary['nodes'][:1]
    ordinary['nodes'][0].pop('producing')
    ordinary['nodes'][0]['target_weights']=[0.,.05,1.]
    ordinary['conversion']={'role_interpretation':'ordinary-text','source_suffix_start':2,'context_weight':.05}
    eligible=tmp_path/'ordinary.jsonl';eligible.write_text(json.dumps(ordinary)+'\n')
    out=tmp_path/'policy.json'
    assert main(['--source',str(path),'--source',str(eligible),
                 '--legacy-pair-context-weight','.05','--out',str(out)])==0
    policy=load_policy({'path':str(out),'sha256':sha256_file_hex(out)})
    assert len(policy['sources'])==1 and policy['sources'][0]['sha256']==sha256_file_hex(eligible)
    assert len(policy['retained_sources'])==1 and policy['retained_sources'][0]['sha256']==sha256_file_hex(path)
    example=parse_graph(direct,tokenizer_fingerprint='f'*64)
    assert apply_policy(example,direct,source_sha256=sha256_file_hex(path),policy=policy) is example
    assert example.target_weights==(0.,0.,1.,0.,0.,2.)
