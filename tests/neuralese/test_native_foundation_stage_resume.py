"""A current policy owns moments/RNG; its carried stage owns the finite cohort."""
import copy
import json

import pytest
import torch

from test_native_model import tiny_source
from test_native_state_sft import graph, authenticated_pair, source_manifest
from natlang_neuralese.model.native_model import load_native_model
from natlang_neuralese.train import native_state_sft as trainer
from natlang_neuralese.train.native_continuation import collect_sft_progress


@pytest.mark.parametrize('mode,stage', [
    ('policy', 'full'), ('ordinary', 'full'), ('policy', None),
    ('wrong_stage', 'full'), ('missing_flag', 'full'), ('unrelated_top_flag', 'full')])
def test_real_resume_uses_only_the_requested_carried_foundation(tmp_path, mode, stage):
    base=tmp_path/'base';tiny_source(base)
    _,_,roles,config=load_native_model(base,dtype=torch.float32,rank=4)
    fp=config['tokenizer_fingerprint']
    original=tmp_path/'original';original.mkdir()
    ordinary=[dict(graph('ordinary'),tokenizer_fingerprint=fp)]
    old_pair=authenticated_pair('old',roles=roles.ids,fingerprint=fp)
    manifest=source_manifest(original,ordinary,[old_pair])
    held=tmp_path/'held.jsonl'
    held.write_text(json.dumps(dict(graph('held'),split='test',tokenizer_fingerprint=fp))+'\n')
    args=['--base',str(base),'--dtype','float32','--rank','4','--device','cpu',
        '--native-graphs',str(manifest),'--foundation-native-graphs',str(manifest),
        '--communication-foundation','--held-native-graphs',str(held),'--epochs','1',
        '--eval-every','100','--out',str(tmp_path/'first'),
        '--forward-atol','1e-4','--gradient-atol','1e-5']
    if stage is not None:args+=['--stage-id',stage]
    assert trainer.main(args)==0
    first=torch.load(tmp_path/'first/checkpoint.pt',weights_only=False)
    source_frame=first['training']
    key=stage or 'native-graph-ce'
    assert source_frame['exposure_counts']=={'ordinary':0,'lfm:old':1}
    old_report=json.loads((tmp_path/'first/report.json').read_text())
    old_cohort=old_report['communication_foundation']['cohort_sha256']
    carried=collect_sft_progress(source_frame)
    carried[key]['completed_foundations']=[old_cohort]
    parent=copy.deepcopy(first)
    if mode!='ordinary':
        top=parent['training']
        top['identity']['objective']={'kind':'native-policy','sha256':'a'*64}
        top['options']['stage_id']='native_rl'
        top['options']['communication_foundation']=mode=='unrelated_top_flag'
        top['sft_progress']=carried
        if mode=='wrong_stage':top['sft_progress']={'another_stage':carried[key]}
        if mode in ('missing_flag','unrelated_top_flag'):
            top['sft_progress'][key]['options'].pop('communication_foundation')
    else:
        parent['training']['completed_foundations']=[old_cohort]
    parent_path=tmp_path/'parent.pt';torch.save(parent,parent_path)
    expanded=tmp_path/'expanded';expanded.mkdir()
    expanded_manifest=source_manifest(expanded,ordinary,[old_pair,
        authenticated_pair('new',roles=roles.ids,fingerprint=fp)])
    run=tmp_path/'resumed';continued=list(args)
    continued[continued.index('--native-graphs')+1]=str(expanded_manifest)
    continued[continued.index('--out')+1]=str(run)
    continued[continued.index('--eval-every')+1]='1'
    continued+=['--continue-from',str(parent_path),'--objective-continuation',
        '--continued-native-graphs',str(manifest),'--refresh-at-checkpoint','--checkpoint-every','1']
    if mode in ('wrong_stage','missing_flag','unrelated_top_flag'):
        with pytest.raises(ValueError,match='entire currently declared communication cohort'):
            trainer.main(continued)
        assert not (run/'checkpoint.pt').exists()
        return
    assert trainer.main(continued)==0
    saved=torch.load(run/'checkpoint.pt',weights_only=False)
    frame=saved['training'];report=json.loads((run/'report.json').read_text())
    assert frame['global_step']==source_frame['global_step']+1
    assert frame['exposure_counts']['lfm:old']==1
    assert sum(frame['exposure_counts'].values())==2
    assert frame['exposure_counts']['lfm:new']==1
    assert frame['exposure_counts']['ordinary']==0
    assert report['communication_foundation']['cohort_sha256']==old_cohort
    assert not report['foundation_ready']
    assert not report['completed']
    assert report['mandatory_sft']['remaining_graph_visits']==1
    assert any(not torch.equal(first['backbone_state'][name],value)
               for name,value in saved['backbone_state'].items())
    assert frame['optimizer']
    if mode=='policy' and stage=='full':
        # The same actual weights/moments/RNG/credits through a direct ordinary
        # parent must produce the identical next update. No fake policy update
        # is claimed by the metadata-only policy-top fixture above.
        control=copy.deepcopy(first)
        control['training']['completed_foundations']=[old_cohort]
        control_path=tmp_path/'ordinary-parent.pt';torch.save(control,control_path)
        control_args=list(continued)
        control_args[control_args.index('--continue-from')+1]=str(control_path)
        control_args[control_args.index('--out')+1]=str(tmp_path/'control')
        assert trainer.main(control_args)==0
        expected=torch.load(tmp_path/'control/checkpoint.pt',weights_only=False)
        def equal(left,right):
            if torch.is_tensor(left):torch.testing.assert_close(left,right,rtol=0,atol=0)
            elif isinstance(left,dict):
                assert left.keys()==right.keys()
                for name in left:equal(left[name],right[name])
            elif isinstance(left,(tuple,list)):
                assert len(left)==len(right)
                for a,b in zip(left,right):equal(a,b)
            else:assert left==right
        for section in ('backbone_state','interface_state'):equal(saved[section],expected[section])
        for section in ('optimizer','rng','exposure_counts'):
            equal(frame[section],expected['training'][section])


def test_fresh_incomplete_foundation_stays_strict(tmp_path):
    from types import SimpleNamespace
    from natlang_neuralese.train.native_continuation import verify_initial_foundation_cohort
    train=SimpleNamespace(entries=[{'id':'old','binding':'a','communication':True},
                                   {'id':'new','binding':'b','communication':True}])
    foundation=SimpleNamespace(entries=train.entries[:1])
    with pytest.raises(ValueError,match='entire currently declared communication cohort'):
        verify_initial_foundation_cohort(train,foundation)
