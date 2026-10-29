from types import SimpleNamespace
from natlang_neuralese.train.native_state_sft import pending_exposure_batches
from natlang_neuralese.train.native_source_streams import stream_pending_batches


def test_rotation_prioritizes_appended_rows_preserves_all_credits_and_wraps():
    entries=[{'id':f'row-{i}','forward_tokens':1,'supervised_tokens':1,'communication':False,'training_priority':0} for i in range(12)]
    credits={'row-1':1,'row-8':1}
    q,normal=pending_exposure_batches(entries,credits,epochs=1,batch_graphs=1)
    rotated_q,rotated=pending_exposure_batches(entries,credits,epochs=1,batch_graphs=1,rotation_index=8)
    assert q==rotated_q
    assert normal[0][1]==[0] and rotated[0][1]==[9]
    assert sorted(tuple(x[1]) for x in normal)==sorted(tuple(x[1]) for x in rotated)
    class Source:
        def __init__(self):self.entries=entries
        def iter_indices(self,indices):
            assert indices==sorted(set(indices))
            return iter(SimpleNamespace(id=entries[i]['id']) for i in indices)
    consumed=[x.id for _,_,_,batch in stream_pending_batches(Source(),rotated,rotation_index=8) for x in batch]
    assert consumed==[entries[b[1][0]]['id'] for b in rotated]
    after={**credits,'row-9':1,'row-10':1}
    _,resumed=pending_exposure_batches(entries,after,epochs=1,batch_graphs=1,rotation_index=8)
    assert [b[1] for b in resumed]==[b[1] for b in rotated if b[1][0] not in (9,10)]
    assert set(consumed)=={f'row-{i}' for i in range(12)}-set(credits)


def test_rotation_keeps_joint_communication_and_urgent_interleave():
    entries=[]
    for i in range(96):
        entries.append({'id':f'row-{i}','forward_tokens':1,'supervised_tokens':1,
            'communication':i%3==0,'training_priority':1 if i%3==1 else 0})
    quotas,batches=pending_exposure_batches(entries,{},epochs=1,batch_graphs=1,rotation_index=48)
    first=[b[1][0] for b in batches[:32]]
    assert any(i>=48 for i in first)
    assert any(entries[i]['communication'] for i in first)
    assert any(entries[i]['training_priority']>0 for i in first)
    assert len({i for b in batches for i in b[1]})==len(entries)
    assert all(value==1 for value in quotas.values())


def test_rotation_keeps_fixed_foundation_before_other_communication():
    entries=[{'id':f'row-{i}','forward_tokens':1,'supervised_tokens':1,'communication':True,'training_priority':0} for i in range(6)]
    _,batches=pending_exposure_batches(entries,{},epochs=1,batch_graphs=1,rotation_index=3,foundation_ids={'row-0','row-1'})
    assert {entries[b[1][0]]['id'] for b in batches[:2]}=={'row-0','row-1'}
