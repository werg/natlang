"""Fixed-cohort scheduling accelerates readiness without deleting training debt."""
from collections import Counter
import json
import pytest
from natlang_neuralese.train.native_state_sft import pending_exposure_batches
from natlang_neuralese.train.native_source_streams import stream_pending_batches


def entries():
    return ([dict(id='expanded-'+str(i),communication=True,training_priority=9,forward_tokens=3) for i in range(3)]
        + [dict(id='fixed-'+str(i),communication=True,training_priority=0,forward_tokens=3) for i in range(6)]
        + [dict(id='teacher-'+str(i),communication=False,training_priority=1,forward_tokens=3) for i in range(12)]
        + [dict(id='ordinary',communication=False,training_priority=0,forward_tokens=3)])


def names(rows,batches):
    return [rows[index]['id'] for _,indices,_ in batches for index in indices]


def test_fixed_cohort_first_four_to_one_then_normal_mix_and_exact_all_quotas():
    rows=entries();fixed={e['id'] for e in rows if e['id'].startswith('fixed-')}
    quotas,batches=pending_exposure_batches(rows,{},epochs=1,foundation_ids=fixed,foundation_communication_batches=4)
    actual=names(rows,batches)
    assert actual[:10]==['fixed-0','fixed-1','fixed-2','fixed-3','teacher-0','fixed-4','fixed-5','teacher-1','expanded-0','teacher-2']
    assert actual[-1]=='ordinary'
    assert Counter(actual)==quotas
    _,legacy=pending_exposure_batches(rows,{},epochs=1)
    assert sorted(tuple(indices) for _,indices,_ in batches)==sorted(tuple(indices) for _,indices,_ in legacy)
    assert all(last==(index==len(batches)-1) for index,(_,_,last) in enumerate(batches))


def test_resume_and_new_high_priority_append_preserve_every_credit_and_packing():
    rows=entries();fixed={e['id'] for e in rows if e['id'].startswith('fixed-')};credits={'fixed-0':1,'teacher-0':1}
    quotas,batches=pending_exposure_batches(rows,credits,epochs=2,batch_tokens=7,foundation_ids=fixed,foundation_communication_batches=4)
    for _,indices,_ in batches[:3]:
        for i in indices:credits[rows[i]['id']]=credits.get(rows[i]['id'],0)+1
    rows.append(dict(id='new-expanded',communication=True,training_priority=100,forward_tokens=99))
    quotas,remaining=pending_exposure_batches(rows,credits,epochs=2,batch_tokens=5,foundation_ids=fixed,foundation_communication_batches=4)
    observed=Counter(names(rows,remaining))
    assert observed=={e['id']:quotas[e['id']]-credits.get(e['id'],0) for e in rows if quotas[e['id']]>credits.get(e['id'],0)}
    first=names(rows,remaining)[0];assert first in fixed
    assert any(indices==[len(rows)-1] for _,indices,_ in remaining)  # oversized graph stays whole
    for index,(visit,_,last) in enumerate(remaining):
        assert last==all(v!=visit for v,_,_ in remaining[index+1:])


def test_completed_foundation_automatically_restores_default_mix():
    rows=entries();fixed={e['id'] for e in rows if e['id'].startswith('fixed-')};credits=dict.fromkeys(fixed,1)
    assert pending_exposure_batches(rows,credits,epochs=1)==pending_exposure_batches(rows,credits,epochs=1,foundation_ids=fixed,foundation_communication_batches=4)


@pytest.mark.parametrize('kind',['no-ordinary','no-communication','empty'])
def test_zero_classes_remain_complete(kind):
    rows=[] if kind=='empty' else [e for e in entries() if bool(e.get('communication'))==(kind=='no-ordinary')]
    fixed={e['id'] for e in rows if e['id'].startswith('fixed-')}
    quotas,batches=pending_exposure_batches(rows,{},epochs=1,foundation_ids=fixed,foundation_communication_batches=4)
    assert Counter(names(rows,batches))==quotas


@pytest.mark.parametrize('ratio',[0,-1,1.5,True])
def test_invalid_ratios_reject(ratio):
    with pytest.raises(ValueError,match='positive integer'):
        pending_exposure_batches(entries(),{},epochs=1,foundation_communication_batches=ratio)


@pytest.mark.parametrize('fixed',[{'absent'},{'ordinary'}])
def test_noncommunication_or_absent_foundation_rejects(fixed):
    with pytest.raises(ValueError,match='foundation priority'):
        pending_exposure_batches(entries(),{},epochs=1,foundation_ids=fixed)


def test_reordered_foundation_uses_independent_monotone_compressed_readers(tmp_path,monkeypatch):
    from test_native_source_streams import compressed_source,count_readers
    source=compressed_source(tmp_path)
    pairs=[e['id'] for e in source.entries if e.get('communication')]
    fixed={pairs[2],pairs[4]}
    _,batches=pending_exposure_batches(source.entries,{},epochs=1,foundation_ids=fixed,foundation_communication_batches=4)
    counts,active=count_readers(monkeypatch)
    observed=[row.id for _,_,_,batch in stream_pending_batches(source,batches,priority_ids=fixed) for row in batch]
    assert observed==names(source.entries,batches)
    assert observed[:2]==[pairs[2],pairs[4]]
    assert sorted(counts.values())==[1,2]  # one ordinary reader; two monotone pair categories
    assert not active


def test_actual_tiny_stop_append_resume_keeps_optimizer_rng_and_all_row_visits(tmp_path,monkeypatch):
    import torch
    from test_native_model import tiny_source
    from test_native_state_sft import graph,authenticated_pair,source_manifest
    from natlang_neuralese.model.native_model import load_native_model
    from natlang_neuralese.train import native_state_sft as trainer
    from natlang_neuralese.common.hashing import sha256_file_hex
    base=tmp_path/'base';tiny_source(base)
    _,_,roles,config=load_native_model(base,dtype=torch.float32,rank=4);fp=config['tokenizer_fingerprint']
    ordinary=[dict(graph('ordinary-'+str(i)),tokenizer_fingerprint=fp) for i in range(4)]
    pairs=[authenticated_pair('foundation-'+str(i),roles=roles.ids,fingerprint=fp) for i in range(4)]
    first=tmp_path/'first';first.mkdir();source=source_manifest(first,ordinary,pairs)
    declaration=json.loads(source.read_text());declaration['sources'][0]['training_priority']=1;source.write_text(json.dumps(declaration))
    held=tmp_path/'held.jsonl';held.write_text(json.dumps(dict(graph('held'),split='test',tokenizer_fingerprint=fp))+'\n')
    out=tmp_path/'run';argv=['--base',str(base),'--dtype','float32','--rank','4','--device','cpu','--native-graphs',str(source),'--foundation-native-graphs',str(source),'--foundation-communication-batches','4','--held-native-graphs',str(held),'--epochs','1','--eval-every','100','--out',str(out),'--held-eval-tokens','4']
    def stop(policy,step,**kwargs):
        policy.stop.requested=step==2
        return False
    with monkeypatch.context() as patch:
        patch.setattr(trainer.CheckpointPolicy,'due',stop)
        assert trainer.main(argv)==0
    partial=torch.load(out/'checkpoint.pt',weights_only=False)['training']
    assert partial['global_step']==2 and sum(partial['exposure_counts'].values())==2
    assert all(partial['exposure_counts']['lfm:foundation-'+str(i)]==(i<2) for i in range(4))
    second=tmp_path/'second';second.mkdir();new=source_manifest(second,ordinary,[*pairs,*[authenticated_pair('expanded-'+str(i),roles=roles.ids,fingerprint=fp) for i in range(2)]])
    declaration=json.loads(new.read_text());declaration['sources'][0]['training_priority']=1;new.write_text(json.dumps(declaration))
    continued=argv.copy();continued[continued.index('--native-graphs')+1]=str(new);continued+=['--continued-native-graphs',str(source)]
    actual=trainer.native_training_update;observed=[]
    def update(model,optimizer,batch,**kwargs):
        if not observed:
            assert torch.equal(torch.get_rng_state(),partial['rng']['torch_rng'])
            assert optimizer.state_dict().keys()==partial['optimizer'].keys()
        observed.extend(row.id for row in batch)
        return actual(model,optimizer,batch,**kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(trainer,'native_training_update',update)
        assert trainer.main(continued)==0
    assert observed[:3]==['lfm:foundation-2','lfm:foundation-3','ordinary-0']
    final=torch.load(out/'checkpoint.pt',weights_only=False)['training']
    assert final['global_step']==10 and all(v==1 for v in final['exposure_counts'].values())
    assert final['optimizer'] and final['metrics']['held']['scope']=='full'
