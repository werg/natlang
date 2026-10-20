"""Mixed admitted containers keep every row's explicit objective and graph."""
import copy
import json

import pytest
import torch

from scripts.neuralese_data.build_native_ce_supervision_policy import complete_source_contract, main
from natlang_neuralese.common.hashing import sha256_file_hex
from natlang_neuralese.data.native_graph_schema import parse_graph
from natlang_neuralese.train.native_supervision import apply_policy, held_measurements, load_policy
from test_native_supervision_mass import ordinary, declared


def mixed_policy():
    return complete_source_contract(declared(), [{'path':'/actual/source.jsonl','sha256':'a'*64}],
                                    context_weight=.05, inventory=[{'path':'/actual/dag.json','sha256':'b'*64}])


def segmented():
    row=ordinary();row['id']='segmented'
    row['conversion']['supervision_policy']={'weighting':'segmented_position_weights'}
    row['nodes'][0]['target_weights']=[0.]+[.25]*23+[1.,1.]
    return row


def pair():
    row=ordinary();row.pop('conversion')
    writer=row['nodes'][0];writer['name']='producer';writer['producing']=[[22,26]]
    reader={'name':'consumer','token_ids':[5]*20+[6,7],
            'target_weights':[0.,0.]+[.05]*18+[1.,1.],
            'imports':[{'at':1,'source':'producer','start':22,'stop':26}]}
    row['nodes'].append(reader)
    row['source_pairing']={'pair_id':'independent-fixture','payloads':[{'invocation':'producer'}],
        'consumed_occurrences':[{'consumer_import_at':1}],'source_ids':['writer','reader']}
    return row


@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('family', ['lfm','mellum'])
@pytest.mark.parametrize('kind', ['ordinary','pair'])
def test_mixed_wt_and_raw_loss_every_gradient_same_as_separate_row_contracts(family, reverse, kind):
    from natlang_neuralese.model.shared_state import SharedStateModel
    from natlang_neuralese.train.native_state_sft import packed_graph_loss
    from test_shared_state import tiny
    torch.manual_seed(125)
    model=SharedStateModel(tiny(family),marker_ids=(58,59,60,61),rank=4)
    candidate=copy.deepcopy(model)
    raw=ordinary() if kind=='ordinary' else pair()
    raw['nodes'][0]['target_weights'][3]=0.
    rows=[segmented(),raw]
    if reverse:rows.reverse()
    old=[];new=[]
    for row in rows:
        example=parse_graph(row,tokenizer_fingerprint='f'*64)
        old.append(apply_policy(example,row,source_sha256='a'*64,
                   policy=(declared('ordinary-suffix-raw-v1' if kind=='ordinary' else 'witnessed-pair-raw-v1')
                           if not row.get('conversion',{}).get('supervision_policy') else None)))
        new.append(apply_policy(example,row,source_sha256='a'*64,policy=mixed_policy()))
    for expected,actual in zip(old,new):
        assert actual.plan==expected.plan and actual.id==expected.id
        assert actual.target_weights==expected.target_weights
    losses=[]
    for engine,examples in [(model,old),(candidate,new)]:
        loss=packed_graph_loss(engine,examples,chunk_tokens=7)[0];loss.backward();losses.append(loss.detach())
    assert torch.equal(*losses)
    for (name,left),(other,right) in zip(model.named_parameters(),candidate.named_parameters()):
        assert name==other
        assert (left.grad is None)==(right.grad is None)
        if left.grad is not None:assert torch.equal(left.grad,right.grad),name


def test_contract_preserves_coefficients_held_masks_and_all_prior_metadata():
    prior=declared();prior['sources'][0]['context_weight']=.125
    prior.update(held_sources=[{'sha256':'c'*64,'encoding':'ordinary-suffix-raw-v1','context_weight':.02}],
                 failed_history_masks=[{'path':'/mask','sha256':'d'*64}],custom={'value':True})
    before=copy.deepcopy(prior)
    result=complete_source_contract(prior,[{'path':'/a','sha256':'a'*64},{'path':'/b','sha256':'b'*64}],
                                    context_weight=.05,inventory=[])
    assert prior==before
    assert result['sources'][0]['context_weight']==.125
    assert result['sources'][1]['context_weight']==.05
    for key in ('held_sources','failed_history_masks','custom'):assert result[key]==prior[key]
    with pytest.raises(ValueError,match='held'):
        complete_source_contract(prior,[{'path':'/c','sha256':'c'*64}],context_weight=.05,inventory=[])


@pytest.mark.parametrize('edit,error',[
    (lambda row:row['conversion'].update(context_weight=.1),'coefficient'),
    (lambda row:row['nodes'][0]['target_weights'].__setitem__(3,.1),'raw-weight'),
    (lambda row:row['conversion'].pop('source_suffix_start'),'suffix'),
    (lambda row:row.update(source_pairing={'pair_id':'ambiguous'}),'ambiguous'),
    (lambda row:row['conversion'].update(role_interpretation='unknown'),'ambiguous'),
    (lambda row:row.update(conversion={},source_pairing={'pair_id':'incomplete'}),'provenance'),
])
def test_mixed_unknown_or_malformed_raw_rows_fail_closed(edit,error):
    row=ordinary();example=parse_graph(row,tokenizer_fingerprint='f'*64);edit(row)
    with pytest.raises(ValueError,match=error):apply_policy(example,row,source_sha256='a'*64,policy=mixed_policy())


def test_direct_native_masks_and_explicit_segmented_held_are_unchanged():
    for row in (ordinary(),segmented()):
        if not row['conversion'].get('supervision_policy'):row.pop('conversion')
        example=parse_graph(row,tokenizer_fingerprint='f'*64)
        assert apply_policy(example,row,source_sha256='a'*64,policy=mixed_policy()) is example
        row['split']='test';example=parse_graph(row,tokenizer_fingerprint='f'*64)
        policy={'held_sources':mixed_policy()['sources']}
        assert held_measurements(example,row,source_sha256='a'*64,policy=policy) is example


def test_held_mixed_raw_uses_same_existing_target_measurements():
    row=ordinary();row['split']='test';example=parse_graph(row,tokenizer_fingerprint='f'*64)
    mixed=held_measurements(example,row,source_sha256='a'*64,policy={'held_sources':mixed_policy()['sources']})
    reference=held_measurements(example,row,source_sha256='a'*64,policy={'held_sources':declared()['sources']})
    assert mixed.target_weights==example.target_weights
    assert mixed.evaluation_weights==reference.evaluation_weights


def test_real_cli_mixed_first_row_and_exact_dag_count(tmp_path):
    source=tmp_path/'source.jsonl';source.write_text('\n'.join(json.dumps(row)for row in (segmented(),ordinary()))+'\n')
    prior=tmp_path/'old.json';prior.write_text(json.dumps(declared(digest=sha256_file_hex(source))))
    out=tmp_path/'new.json'
    args=['--prior-policy',str(prior),'--complete-source-contract','--source',str(source),
          '--legacy-pair-context-weight','.05','--expected-source-count','1','--out',str(out)]
    assert main(args)==0
    result=load_policy({'path':str(out),'sha256':sha256_file_hex(out)})
    assert result['source_contract']['active_source_count']==1
    assert result['sources'][0]['encoding']=='native-row-objectives-v1'
    for row in (segmented(),ordinary()):
        example=parse_graph(row,tokenizer_fingerprint='f'*64)
        apply_policy(example,row,source_sha256=sha256_file_hex(source),policy=result)
    with pytest.raises(ValueError,match='count'):
        wrong=list(args);wrong[wrong.index('--expected-source-count')+1]='2'
        wrong[-1]=str(tmp_path/'bad.json');main(wrong)


@pytest.mark.parametrize('reverse', [False,True])
def test_default_fresh_cli_never_assumes_first_row_retention(tmp_path,reverse):
    rows=[segmented(),ordinary()]
    if reverse:rows.reverse()
    source=tmp_path/'actual-mixed.jsonl';source.write_text('\n'.join(json.dumps(row)for row in rows)+'\n')
    out=tmp_path/'new-default.json'
    assert main(['--source',str(source),'--legacy-pair-context-weight','.05','--out',str(out)])==0
    result=load_policy({'path':str(out),'sha256':sha256_file_hex(out)})
    assert result['source_contract']['active_source_count']==1
    assert result['sources'][0]['encoding']=='native-row-objectives-v1'
    assert not result['retained_sources']
    for row in rows:
        before=parse_graph(row,tokenizer_fingerprint='f'*64)
        after=apply_policy(before,row,source_sha256=sha256_file_hex(source),policy=result)
        if row['conversion'].get('supervision_policy'):assert after is before
        else:assert after.target_weights!=before.target_weights
