"""Real saved-row custody and pending-credit behavior during source recovery."""
import copy
import json

import pytest

from natlang_neuralese.train.native_availability import defer_unavailable_sources, SCHEMA
from natlang_neuralese.train.native_continuation import collect_sft_progress, progress_obligations


def inputs(tmp_path):
    root={'path':'/original/all.json','sha256':'a'*64,'rows':4,'sources':[
        {'path':'/original/available.zst','sha256':'b'*64,'rows':2},
        {'path':'/original/missing.zst','sha256':'c'*64,'rows':2}]}
    training={'schema':'natlang.neuralese-native-sft-training/1','step':3,'global_step':123,
        'epoch':0,'cursor':3,'identity':{'objective':None},'options':{'stage_id':'full','epochs':1},
        'source_receipts':[root,{'path':'/held','sha256':'d'*64}],
        'source_row_bindings':{key:str(index)*64 for index,key in enumerate('abcd',1)},
        'source_row_order':list('abcd'),'exposure_counts':{'a':1,'b':0,'c':1,'d':0},
        'optimizer':{'actual':'earned moments'},'rng':{'actual':'earned stream'},'fp8':{'actual':'scales'}}
    proof={'schema':SCHEMA,'model_id':'lfm:base','stage_id':'full','preserve_obligations':True,
        'reason':'Original external source unavailable; continue recovered rows and retain every missing obligation.',
        'prior_train_source':{key:root[key]for key in('path','sha256')},
        'unavailable_sources':[copy.deepcopy(root['sources'][1])]}
    path=tmp_path/'availability.json';path.write_text(json.dumps(proof))
    current={key:value for key,value in training['source_row_bindings'].items()if key in'ab'}
    return training,proof,path,current


def apply(training,path,current):
    return defer_unavailable_sources(training,path,current_bindings=current,
        model_id='lfm:base',current_source_sha256s={'b'*64})


def test_missing_leaf_keeps_actual_earned_and_required_visits_through_real_save_collect(tmp_path):
    training,proof,path,current=inputs(tmp_path);before=copy.deepcopy(training)
    result=apply(training,path,current)
    assert training==before
    for key in ('optimizer','rng','fp8'):assert result[key]is training[key]
    assert result['source_row_order']==['a','b']
    assert result['source_row_bindings']==current
    assert result['exposure_counts']==training['exposure_counts']
    # This is the shared trainer's save/collect shape: only active counts in
    # the new frame; the carried pending ledger retains absent earned counts.
    saved={**result,'source_receipts':[{'path':'/new/available.json','sha256':'e'*64},training['source_receipts'][1]],
           'exposure_counts':{'a':1,'b':1}}
    frames=collect_sft_progress(saved);report=progress_obligations(frames)['stages']['full']
    assert report['graph_visits']==3
    assert report['required_graph_visits']==4
    assert report['pending_source_graph_visits']==1
    assert report['active_remaining_graph_visits']==0
    assert not report['complete']
    assert frames['full']['exposure_counts']['c']==1
    assert apply({**saved,'sft_progress':frames},path,current)['global_step']==123
    # Later source restoration uses the existing activation mechanism, with
    # exact old bindings and counts rather than treating recovered bytes as visits.
    restored={**saved,'source_row_bindings':training['source_row_bindings'],
              'source_row_order':list('abcd'),'exposure_counts':{'a':1,'b':1,'c':0,'d':0},'sft_progress':frames}
    active=collect_sft_progress(restored)['full']
    assert active['exposure_counts']['c']==1 and active['exposure_counts']['d']==0
    assert active['pending_source_rows']=={}
    assert not progress_obligations({'full':active})['complete']


@pytest.mark.parametrize('mutation',[
    lambda t,p,c:p.update(model_id='other'),
    lambda t,p,c:p.update(stage_id='other'),
    lambda t,p,c:p.update(preserve_obligations=False),
    lambda t,p,c:p.update(reason=''),
    lambda t,p,c:p['prior_train_source'].update(sha256='f'*64),
    lambda t,p,c:p['unavailable_sources'][0].update(rows=1),
    lambda t,p,c:p['unavailable_sources'][0].update(path='/wrong'),
    lambda t,p,c:p['unavailable_sources'].append(copy.deepcopy(p['unavailable_sources'][0])),
    lambda t,p,c:t['identity'].update(objective={'kind':'DPO'}),
    lambda t,p,c:t['source_row_order'].pop(),
    lambda t,p,c:c.update(c=t['source_row_bindings']['c']),
    lambda t,p,c:c.pop('b'),
    lambda t,p,c:c.update(b='wrong'),
])
def test_availability_never_forgives_unenumerated_data_or_changes_saved_authority(tmp_path,mutation):
    training,proof,path,current=inputs(tmp_path);mutation(training,proof,current)
    path.write_text(json.dumps(proof))
    with pytest.raises(ValueError):apply(training,path,current)


def test_active_leaf_cannot_be_declared_missing(tmp_path):
    training,proof,path,current=inputs(tmp_path)
    with pytest.raises(ValueError,match='active'):
        defer_unavailable_sources(training,path,current_bindings=current,model_id='lfm:base',
                                  current_source_sha256s={'b'*64,'c'*64})


def test_real_trainer_continues_after_payload_loss_and_reactivates_exact_remaining_row(tmp_path):
    import torch
    from test_native_model import tiny_source
    from test_native_state_sft import graph
    from natlang_neuralese.model.native_model import load_native_model
    from natlang_neuralese.common.hashing import sha256_file_hex_cached
    from natlang_neuralese.train import native_state_sft as trainer
    base=tmp_path/'base';tiny_source(base)
    _,_,_,config=load_native_model(base,dtype=torch.float32,rank=4)
    leaves=[]
    for name in ('a','b','c','held'):
        row=dict(graph(name),split='test'if name=='held'else'train',
                 source_groups=['held'if name=='held'else'train'],tokenizer_fingerprint=config['tokenizer_fingerprint'])
        path=tmp_path/(name+'.jsonl');path.write_text(json.dumps(row)+'\n');leaves.append(path)
    def manifest(path,children):
        path.write_text(json.dumps({'schema':'natlang.native-graph-sources/1','sources':[
            {'kind':'native-graphs','path':str(child),'sha256':sha256_file_hex_cached(child)}for child in children]}))
        return path
    full=manifest(tmp_path/'all.json',leaves[:3]);available=manifest(tmp_path/'available.json',leaves[:2])
    argv=['--base',str(base),'--dtype','float32','--rank','4','--device','cpu',
        '--native-graphs',str(full),'--held-native-graphs',str(leaves[3]),'--epochs','1',
        '--eval-every','100','--out',str(tmp_path/'run'),'--forward-atol','1e-4','--gradient-atol','1e-5']
    assert trainer.main(argv+['--refresh-after-updates','1'])==0
    parent=torch.load(tmp_path/'run/checkpoint.pt',weights_only=False)
    t=parent['training'];assert t['exposure_counts']=={'a':1,'b':0,'c':0}
    proof={'schema':SCHEMA,'stage_id':'native-graph-ce',
        'model_id':config['architecture']+':'+config['base_inventory_sha256'],
        'preserve_obligations':True,'reason':'Actual payload temporarily disappeared.',
        'prior_train_source':{key:t['source_receipts'][0][key]for key in('path','sha256')},
        'unavailable_sources':[t['source_receipts'][0]['sources'][2]]}
    declaration=tmp_path/'unavailable.json';declaration.write_text(json.dumps(proof))
    missing=leaves[2].read_bytes();leaves[2].unlink();full.unlink()
    continued=argv.copy();continued[continued.index('--native-graphs')+1]=str(available)
    assert trainer.main(continued+['--unavailable-native-sources',str(declaration)])==0
    saved=torch.load(tmp_path/'run/checkpoint.pt',weights_only=False)
    assert saved['training']['global_step']==t['global_step']+1
    assert saved['training']['exposure_counts']=={'a':1,'b':1}
    frame=saved['training']['sft_progress']['native-graph-ce']
    assert frame['pending_source_rows']['c']['required_visits']==1
    assert frame['exposure_counts']['c']==0
    report=json.loads((tmp_path/'run/report.json').read_text())
    assert not report['completed'] and report['mandatory_sft']['remaining_graph_visits']==1
    # Restart with the same declaration and no work: no spurious optimizer step.
    assert trainer.main(continued+['--unavailable-native-sources',str(declaration)])==0
    assert torch.load(tmp_path/'run/checkpoint.pt',weights_only=False)['training']['global_step']==t['global_step']+1
    leaves[2].write_bytes(missing);full=manifest(full,leaves[:3])
    assert trainer.main(argv+['--continued-native-graphs',str(available)])==0
    final=torch.load(tmp_path/'run/checkpoint.pt',weights_only=False)
    assert final['training']['global_step']==t['global_step']+2
    assert final['training']['exposure_counts']=={'a':1,'b':1,'c':1}
    assert json.loads((tmp_path/'run/report.json').read_text())['mandatory_sft']['complete']
