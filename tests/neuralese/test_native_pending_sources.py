"""Deferred capacity is a mandatory CE deficit, never a separate earned counter."""
import copy
import json

import pytest

from natlang_neuralese.train.native_continuation import collect_sft_progress, progress_obligations, restore_sft_progress
from natlang_neuralese.train.native_pending_sources import declaration, load_pending_sources
from natlang_neuralese.common.hashing import sha256_file_hex_cached
from test_native_state_sft import graph


def frame():
    return {'schema':'natlang.neuralese-native-sft-training/1',
        'identity':{'objective':None}, 'source_row_bindings':{'fit':'f'*64},
        'source_row_order':['fit'],'exposure_counts':{'fit':1},
        'options':{'epochs':1,'stage_id':'full'}, 'optimizer':{'moments':17}, 'rng':{'seed':31}}


def pending():
    return {'binding':{'path':'/declared/source.json','sha256':'a'*64},
        'rows':{'long':{'binding':'b'*64,'required_visits':2}},'dependencies':[],'sources':[]}


def test_pending_survives_objective_switch_flag_omission_and_exact_activation_without_credit():
    training=frame();training['options']['_pending_native_source_declarations']=[pending()]
    saved=collect_sft_progress(training)
    report=progress_obligations(saved)['stages']['full']
    assert not report['complete'] and report['pending_source_graph_visits']==2
    assert report['active_remaining_graph_visits']==0
    joint={'optimizer':{'new':19},'rng':{'new':37},'global_step':10,'sft_progress':saved}
    restored=restore_sft_progress(joint,'full')
    assert restored['optimizer'] is joint['optimizer'] and restored['rng'] is joint['rng']
    restored['options'].pop('_pending_native_source_declarations')
    omitted=collect_sft_progress(restored)
    assert omitted['full']['pending_source_rows']==saved['full']['pending_source_rows']
    restored['source_row_bindings']['long']='b'*64
    restored['source_row_order'].append('long')
    restored['exposure_counts']['long']=1
    activated=collect_sft_progress(restored)
    assert activated['full']['pending_source_rows']=={}
    assert activated['full']['exposure_counts']['fit']==1
    assert activated['full']['exposure_counts']['long']==1
    assert progress_obligations(activated)['remaining_graph_visits']==1
    restored['sft_progress']=activated;restored['exposure_counts']['long']=2
    assert progress_obligations(collect_sft_progress(restored))['complete']


def test_pending_binding_conflicts_rejected_and_source_append_keeps_all_old_credits():
    training=frame();training['options']['_pending_native_source_declarations']=[pending()]
    saved=collect_sft_progress(training)
    training['sft_progress']=saved
    training['source_row_bindings']['long']='c'*64
    with pytest.raises(ValueError,match='pending admitted row'):
        collect_sft_progress(training)
    training['source_row_bindings'].pop('long')
    added=copy.deepcopy(pending());added['binding']['sha256']='d'*64
    added['rows']={'later':{'binding':'e'*64,'required_visits':1}}
    training['options']['_pending_native_source_declarations']=[added]
    continued=collect_sft_progress(training)
    assert continued['full']['exposure_counts']=={'fit':1,'long':0,'later':0}
    assert progress_obligations(continued)['remaining_graph_visits']==3


def test_declaration_uses_actual_shared_semantic_binding_and_hashes_full_dependency_closure(tmp_path):
    rows=[graph('long')];rows[0]['tokenizer_fingerprint']='1'*64
    graphs=tmp_path/'long.jsonl';graphs.write_text(json.dumps(rows[0])+'\n')
    source=tmp_path/'sources.json'
    source.write_text(json.dumps({'schema':'natlang.native-graph-sources/1','sources':[
        {'kind':'native-graphs','path':str(graphs),'sha256':sha256_file_hex_cached(graphs)}]}))
    value=declaration(source,source,tokenizer_fingerprint='1'*64,role_ids=(58,59,60,61))
    path=tmp_path/'pending.json';path.write_text(json.dumps(value))
    loaded=load_pending_sources(path,tokenizer_fingerprint='1'*64)
    from natlang_neuralese.data.native_sources import NativeGraphSource
    actual=NativeGraphSource(source,'1'*64,required_split='train',role_ids=(58,59,60,61))
    assert loaded['rows']['long']['binding']==actual.bindings['long']
    assert {item['path'] for item in loaded['dependencies']}=={str(path),str(source),str(graphs)}
    assert 'rows' not in loaded['sources'][0]
    graphs.write_text(graphs.read_text()+'\n')
    with pytest.raises(ValueError,match='content hash mismatch'):
        load_pending_sources(path,tokenizer_fingerprint='1'*64)


def test_pending_only_deficit_does_not_launch_repeated_empty_ce_but_new_source_does():
    from natlang_neuralese.rl.native_supervised import stage_pending
    progress={'complete':False,'active_remaining_graph_visits':0,'pending_source_graph_visits':2,
        'train_source_sha256':'train','held_source_sha256':'held','epochs':1}
    adapter={'models':{'student':{'mandatory_sft':{'stages':{'full':progress}}}}}
    sources={'native_graphs':{'sha256':'train'},'held_native_graphs':{'sha256':'held'}}
    assert not stage_pending(adapter,'student','full',sources,1)
    sources['pending_native_sources']={'sha256':'new-declaration'}
    assert stage_pending(adapter,'student','full',sources,1)
    sources.pop('pending_native_sources')
    sources['native_graphs']['sha256']='activated-long-source'
    assert stage_pending(adapter,'student','full',sources,1)


def test_actual_trainer_metadata_only_resume_preserves_numeric_state_then_activates_rows(tmp_path):
    import torch
    from test_native_model import tiny_source
    from natlang_neuralese.model.native_model import load_native_model
    from natlang_neuralese.train import native_state_sft as trainer
    base=tmp_path/'base';tiny_source(base)
    _,_,roles,config=load_native_model(base,dtype=torch.float32,rank=4)
    fp=config['tokenizer_fingerprint']
    def write_graph(path,name,split='train'):
        row=dict(graph(name),tokenizer_fingerprint=fp,split=split)
        path.write_text(json.dumps(row)+'\n')
        return path
    fit=write_graph(tmp_path/'fit.jsonl','fit')
    long=write_graph(tmp_path/'long.jsonl','long')
    held=write_graph(tmp_path/'held.jsonl','held','test')
    full=tmp_path/'full.json';full.write_text(json.dumps({'schema':'natlang.native-graph-sources/1',
        'sources':[{'kind':'native-graphs','path':str(path),'sha256':sha256_file_hex_cached(path)} for path in (fit,long)]}))
    declared=tmp_path/'pending.json';declared.write_text(json.dumps(declaration(full,long,
        tokenizer_fingerprint=fp,role_ids=roles.ids)))
    out=tmp_path/'run'
    argv=['--base',str(base),'--dtype','float32','--rank','4','--device','cpu',
        '--native-graphs',str(fit),'--held-native-graphs',str(held),'--epochs','1',
        '--eval-every','100','--out',str(out),'--forward-atol','1e-4','--gradient-atol','1e-5']
    assert trainer.main(argv)==0
    original=torch.load(out/'checkpoint.pt',weights_only=False)
    assert trainer.main(argv+['--pending-native-sources',str(declared)])==0
    pending_checkpoint=torch.load(out/'checkpoint.pt',weights_only=False)
    def exact(left,right):
        if torch.is_tensor(left):torch.testing.assert_close(left,right,rtol=0,atol=0)
        elif isinstance(left,dict):
            assert left.keys()==right.keys()
            for key in left:exact(left[key],right[key])
        elif isinstance(left,(list,tuple)):
            assert len(left)==len(right)
            for a,b in zip(left,right):exact(a,b)
        else:assert left==right
    for section in ('backbone_state','interface_state'):exact(original[section],pending_checkpoint[section])
    for section in ('optimizer','rng'):exact(original['training'][section],pending_checkpoint['training'][section])
    report=json.loads((out/'report.json').read_text())
    assert not report['completed'] and report['mandatory_sft']['remaining_graph_visits']==1
    assert report['mandatory_sft']['stages']['native-graph-ce']['pending_source_graph_visits']==1
    # Flag omission on process restart cannot erase the pending obligation.
    assert trainer.main(argv)==0
    omitted_report=json.loads((out/'report.json').read_text())
    assert not omitted_report['completed']
    assert omitted_report['pending_source_admission']==report['pending_source_admission']
    activated=argv.copy();activated[activated.index('--native-graphs')+1]=str(full)
    activated+=['--continued-native-graphs',str(fit)]
    assert trainer.main(activated)==0
    final=torch.load(out/'checkpoint.pt',weights_only=False)
    assert final['training']['global_step']==original['training']['global_step']+1
    assert final['training']['exposure_counts']=={'fit':1,'long':1}
    assert json.loads((out/'report.json').read_text())['mandatory_sft']['complete']


def test_whole_readiness_cannot_claim_complete_with_pending_source_deficits(tmp_path):
    from test_native_readiness import ready
    from natlang_neuralese.train.native_readiness import validate_ready
    path,receipt,requirement=ready(tmp_path,'lfm')
    obligations={'schema':'natlang.native-sft-progress/1','complete':False,
        'remaining_graph_visits':1,'stages':{'full':{'complete':False,
            'pending_source_graph_visits':1,'active_remaining_graph_visits':0}}}
    report_path=__import__('pathlib').Path(receipt['training_receipt']['path'])
    report=json.loads(report_path.read_text());report['mandatory_sft']=obligations
    report_path.write_text(json.dumps(report));receipt['training_receipt']['sha256']=sha256_file_hex_cached(report_path)
    receipt['mandatory_sft']=obligations;path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match='mandatory pending source'):
        validate_ready(path,requirement)
