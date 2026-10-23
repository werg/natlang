"""Exact operation evidence remains available without copying it into control."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from natlang_neuralese.common.hashing import canonical_json_sha256_hex
from natlang_neuralese.train import native_operation_report as report


def test_nested_source_receipt_preserves_totals_and_identities_with_linear_row_work(tmp_path):
    from natlang_neuralese.data.native_sources import NativeGraphSource
    visits=[]
    class CountedEntries(list):
        def __iter__(self):
            visits.append(len(self))
            return super().__iter__()
    def leaf(index):
        return SimpleNamespace(path=tmp_path/'staged'/str(index),source_identity_path=tmp_path/str(index),
            file_sha256=str(index),kind='native-graphs',
            entries=CountedEntries([{'groups':['all',str(index)],'supervised_tokens':index+1,
                                    'communication':bool(index%2)}]))
    source=leaf(0)
    for index in range(1,65):
        child=leaf(index)
        source=SimpleNamespace(path=tmp_path/'staged'/f'manifest-{index}',
            source_identity_path=tmp_path/f'manifest-{index}',file_sha256=f'm{index}',
            kind='manifest',children=[source,child],entries=CountedEntries(source.entries+child.entries))
    # Bind the genuine original method to this in-memory source tree.
    def original(node):
        node.receipt=lambda:NativeGraphSource.receipt(node)
        for child in getattr(node,'children',[]):original(child)
    original(source)
    legacy=source.receipt()
    visits.clear()
    bounded=report.source_receipt(source)
    assert sum(visits)==2*65  # root totals plus each leaf once; no ancestor scans
    assert bounded['by_source_group']==legacy['by_source_group']
    def identities(value):
        result={key:value[key] for key in ('path','sha256','rows','communication_graphs')}
        if 'sources' in value:result['sources']=[identities(child) for child in value['sources']]
        return result
    assert identities(bounded)==identities(legacy)
    assert bounded['group_counts_scope']=='root-only/1'
    assert 'by_source_group' not in bounded['sources'][0]
    view=report.SourceIdentityView(source)
    assert view.entries is source.entries
    assert view.receipt()==identities(legacy)


def test_actual_staged_manifest_keeps_original_identities_in_reports_and_classification(tmp_path,monkeypatch):
    from natlang_neuralese.data.native_sources import NativeGraphSource
    from natlang_neuralese.common.hashing import sha256_file_hex
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE',str(tmp_path/'cache'))
    staged=tmp_path/'staged';staged.mkdir()
    leaf=staged/'leaf.jsonl';manifest=staged/'manifest.json'
    original_leaf=tmp_path/'original'/'leaf.jsonl';original_manifest=tmp_path/'original'/'manifest.json'
    leaf.write_text(json.dumps({'schema':'natlang.neuralese-native-graph/1','id':'staged-row',
        'split':'train','source_groups':['staged-group'],'tokenizer_fingerprint':'f'*64,
        'nodes':[{'name':'answer','token_ids':[1,2,3],'target_weights':[0,1,1]}]})+'\n')
    manifest.write_text(json.dumps({'schema':'natlang.native-graph-sources/1','sources':[
        {'kind':'native-graphs','path':str(original_leaf),'sha256':sha256_file_hex(leaf)}]}))
    (staged/'.natlang-source-map.json').write_text(json.dumps({'schema':'natlang.staged-source-map/1',
        'paths':{str(original_leaf):str(leaf),str(original_manifest):str(manifest)}}))
    source=NativeGraphSource(manifest,'f'*64,required_split='train')
    compact=report.source_receipt(source);standard=source.receipt()
    assert compact['path']==standard['path']==str(original_manifest)
    assert compact['sources'][0]['path']==standard['sources'][0]['path']==str(original_leaf)
    assert compact['by_source_group']==standard['by_source_group']
    assert report.SourceIdentityView(source).receipt()['sources'][0]['path']==str(original_leaf)


def fixture(tmp_path):
    checkpoint=tmp_path/'checkpoint.pt'
    groups={'ordinary':{'examples':2,'supervised_tokens':8}}
    frame={'source_row_order':['a','b'],'source_row_bindings':{'a':'binding-a','b':'binding-b'},
        'required_visits':{'a':3,'b':3},'exposure_counts':{'a':2,'b':0},'options':{'epochs':3}}
    state={'training':{'sft_progress':{'full':frame},'source_receipts':[{'sha256':'source','by_source_group':groups}],
                      'source_row_bindings':frame['source_row_bindings'],'exposure_counts':frame['exposure_counts']},
           'model_config':{'tokenizer_fingerprint':'fp','marker_ids':[1,2,3,4]}}
    torch.save(state,checkpoint)
    from natlang_neuralese.train.native_continuation import progress_obligations
    value={'schema':'natlang.neuralese-native-sft-report/1','checkpoint':str(checkpoint),
        'checkpoint_sha256':report._binding(checkpoint)['sha256'],'completed':False,
        'mandatory_sft':progress_obligations(state['training']['sft_progress']),
        'train_source':state['training']['source_receipts'][0],
        'metrics':{'held':{'loss':1.}},'raw_communication_rows':{'g':{'qualified':False}},
        'communication_foundation':{'complete':False,'rows':{'g':{'graph_visits':0}},'sources':['source']},
        'future_owner_guard':{'must_retain':'exact'}}
    return value,state,checkpoint


def test_compact_report_retains_exact_checkpoint_backed_deficits_and_sources(tmp_path):
    value,state,checkpoint=fixture(tmp_path)
    original=copy.deepcopy(value)
    compact=report.compact_training_report(value,report._binding(checkpoint))
    assert value==original
    assert compact['mandatory_sft']['remaining_graph_visits']==4
    stage=compact['mandatory_sft']['stages']['full']
    assert 'missing_rows' not in stage
    assert report.resolve_detail(stage['missing_rows_detail'])=={'a':1,'b':3}
    source=compact['train_source']
    assert 'by_source_group' not in source
    assert report.resolve_detail(source['by_source_group_detail'])==state['training']['source_receipts'][0]['by_source_group']
    assert compact['future_owner_guard']==value['future_owner_guard']


@pytest.mark.parametrize('tamper',['checkpoint','hash','entries','stage'])
def test_detail_tampering_rejects(tmp_path,tamper):
    value,_,checkpoint=fixture(tmp_path)
    compact=report.compact_training_report(value,report._binding(checkpoint))
    proof=compact['mandatory_sft']['stages']['full']['missing_rows_detail']
    if tamper=='checkpoint':checkpoint.write_bytes(b'changed')
    elif tamper=='hash':proof['sha256']='0'*64
    elif tamper=='entries':proof['entries']+=1
    else:proof['stage']='unknown'
    with pytest.raises((ValueError,KeyError)):report.resolve_detail(proof)


def test_group_exposure_detail_uses_exact_source_and_current_row_visits(tmp_path):
    _,state,checkpoint=fixture(tmp_path)
    source=SimpleNamespace(file_sha256='source',bindings=state['training']['source_row_bindings'],entries=[
        {'id':'a','groups':['ordinary'],'supervised_tokens':4},
        {'id':'b','groups':['ordinary'],'supervised_tokens':4}])
    expected={'ordinary':{'graph_visits':2,'supervised_token_visits':8}}
    compact=report.compact_exposure({'by_source_group':expected},report._binding(checkpoint))
    assert report.resolve_detail(compact['by_source_group_detail'],source=source)==expected
    source.bindings={'a':'changed','b':'binding-b'}
    with pytest.raises(ValueError,match='source/rows'):report.resolve_detail(compact['by_source_group_detail'],source=source)


def test_control_progress_is_small_and_all_guards_survive_with_bound_detail_access(tmp_path):
    value,_,checkpoint=fixture(tmp_path)
    value['train_source']['by_source_group']={str(i):{'examples':i,'supervised_tokens':i*4} for i in range(10000)}
    path=tmp_path/'report.json';path.write_text(json.dumps(value))
    control=report.control_progress(value,path)
    assert len(json.dumps(control))<len(path.read_bytes())/20
    assert control['future_owner_guard']==value['future_owner_guard']
    assert control['mandatory_sft']['remaining_graph_visits']==4
    assert control['communication_foundation']['complete'] is False
    assert report.resolve_control_field(control['communication_foundation']['report_details']['rows'])=={'g':{'graph_visits':0}}
    assert report.resolve_control_field(control['report_details']['metrics'])==value['metrics']
    path.write_text('{}')
    with pytest.raises(ValueError,match='report changed'):report.resolve_control_field(control['report_details']['metrics'])


def test_streamed_canonical_hash_is_existing_exact_hash():
    value={'unicode':'é雪','float':1.25,'rows':{'b':3,'a':1},'nested':[False,None,'\\n']}
    assert report._digest(value)==canonical_json_sha256_hex(value)


def test_current_receipt_progress_avoids_checkpoint_unpickle_and_rejects_changed_report(tmp_path,monkeypatch):
    from natlang_neuralese.rl.native_raw_rounds import _progress
    from natlang_neuralese.train.checkpoint_policy import CheckpointPolicy
    _real_report(tmp_path)
    out=tmp_path/'run';binding=report._binding(out/'checkpoint.pt')
    with monkeypatch.context() as patch:
        patch.setattr(CheckpointPolicy,'load',lambda *a,**kw:pytest.fail('current receipt must avoid full checkpoint unpickle'))
        progress=_progress(binding)
    assert progress['remaining_graph_visits']==1
    assert report.resolve_detail(progress['stages']['native-graph-ce']['missing_rows_detail'])=={'train':1}
    receipt_path=out/'native-training-receipt.json'
    receipt=json.loads(receipt_path.read_text())
    receipt['checkpoint']['sha256']='0'*64
    receipt_path.write_text(json.dumps(receipt))
    assert _progress(binding)['remaining_graph_visits']==1  # stale receipt uses real full state
    receipt['checkpoint']['sha256']=binding['sha256']
    receipt_path.write_text(json.dumps(receipt))
    (out/'report.json').write_text('{}')
    with pytest.raises(ValueError,match='report changed'):_progress(binding)



def test_owned_state_compaction_preserves_cursor_and_old_report_bytes(tmp_path):
    value,_,_=fixture(tmp_path)
    path=tmp_path/'report.json';path.write_text(json.dumps(value));original=path.read_bytes()
    state={'last_learner_progress':copy.deepcopy(value),'mandatory_sft':copy.deepcopy(value['mandatory_sft']),
           'pending_rl_round1':{'selection':'sealed','keys':[1,2]},'checkpoint':value['checkpoint'],
           'future_owner_guard':{'must_stay':True}}
    compact=report.compact_controller_state(state,path)
    assert compact['pending_rl_round1']==state['pending_rl_round1']
    assert compact['checkpoint']==state['checkpoint']
    assert compact['future_owner_guard']==state['future_owner_guard']
    assert compact['mandatory_sft']==compact['last_learner_progress']['mandatory_sft']
    assert compact['mandatory_sft']['remaining_graph_visits']==4
    assert path.read_bytes()==original
    assert report.compact_controller_state(compact,path)==compact
    broken=copy.deepcopy(state);broken['last_learner_progress']['future_owner_guard']={'different':True}
    with pytest.raises(ValueError,match='differs from actual'):report.compact_controller_state(broken,path)



def _real_report(tmp_path):
    from test_native_model import tiny_source
    from natlang_neuralese.model.native_model import load_native_model
    from natlang_neuralese.data.native_graph import SCHEMA
    from natlang_neuralese.train.native_state_sft import main
    base=tmp_path/'base';tiny_source(base)
    _,_,_,config=load_native_model(base,dtype=torch.float32,rank=4)
    paths={}
    for split in ('train','test'):
        path=tmp_path/(split+'.jsonl');paths[split]=path
        path.write_text(json.dumps({'schema':SCHEMA,'id':split,'split':split,
            'source_groups':[split],'tokenizer_fingerprint':config['tokenizer_fingerprint'],
            'nodes':[{'name':'answer','token_ids':[1,2,3],'target_weights':[0,1,1]}]})+'\n')
    assert main(['--base',str(base),'--dtype','float32','--rank','4','--device','cpu',
        '--native-graphs',str(paths['train']),'--held-native-graphs',str(paths['test']),
        '--epochs','2','--checkpoint-every','1','--eval-every','1','--refresh-at-checkpoint',
        '--out',str(tmp_path/'run')])==0

def test_actual_trainer_proofs_resolve_without_changing_optimizer_or_source_credits(tmp_path,monkeypatch):
    _real_report(tmp_path)
    out=tmp_path/'run'
    value=json.loads((out/'report.json').read_text())
    ready=json.loads((out/'native-training-receipt.json').read_text())
    state=torch.load(out/'checkpoint.pt',weights_only=False)
    assert state['training']['optimizer'] and state['training']['rng']
    assert state['training']['exposure_counts']=={'train':1}
    assert ready['mandatory_sft']==value['mandatory_sft']
    proof=value['mandatory_sft']['stages']['native-graph-ce']['missing_rows_detail']
    assert report.resolve_detail(proof)=={'train':1}
    assert report.resolve_detail(value['train_source']['by_source_group_detail'])==state['training']['source_receipts'][0]['by_source_group']
    assert report.resolve_detail(ready['exposure']['by_source_group_detail'])=={
        'train':{'graph_visits':1,'supervised_token_visits':2}}


@pytest.mark.parametrize('value', [
    {}, [], (), None, True, 1.25, '雪\\n',
    {3: {'z': [False, None, -0.0, 1e-20], 'a': 'é'}, -2: {}},
    [[], {}, {'b': 2, 'a': 1}, ('nested', 3)],
    [{'id': i, 'label': '雪'} for i in range(2051)],
    {'source-'+str(i): {'examples': i, 'supervised_weight': i / 7,
                       'supervised_tokens': 4*i} for i in range(2000)},
])
def test_inventory_hash_preserves_existing_canonical_json_contract(value):
    assert report._digest(value)==canonical_json_sha256_hex(value)
