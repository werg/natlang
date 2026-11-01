import pytest
from natlang_neuralese.train.native_source_frontier import extend,pending,interleave_batches
from natlang_neuralese.train.native_pending_focus import promote


def test_append_frontier_promotes_only_first_visit_and_resumes():
    entries=[{'id':'old'},{'id':'new-a'},{'id':'new-b'}];bindings={'old':'o','new-a':'a','new-b':'b'}
    history=extend([],['new-a','new-b'],bindings,source_sha256='s',global_step=58)
    batches=[(0,[0],False),(0,[1],False),(0,[2],True),(1,[1],True),(1,[2],True)]
    focus=pending(history,bindings,{'old':3,'new-a':0,'new-b':0})
    actual=promote(batches,entries,focus)
    assert [b[1] for b in actual[:2]]==[[1],[2]]
    assert sorted((v,tuple(i)) for v,i,_ in actual)==sorted((v,tuple(i)) for v,i,_ in batches)
    assert pending(history,bindings,{'new-a':1,'new-b':0})=={'new-b'}
    assert pending(history,bindings,{'new-a':1,'new-b':1})==set()
    with pytest.raises(ValueError,match='binding'):pending(history,bindings|{'new-b':'changed'}, {})
    assert pending(history,{'old':'o','new-a':'a'}, {'new-a':1},[{'rows':[{'id':'new-b','binding':'b'}]}])==set()


def test_explicit_focus_first_visits_then_interleaves_remaining_communication():
    from natlang_neuralese.train.native_diversity import cohort_head_merge, sustained_communication_interleave
    entries=[{'id':f'old-{i}','forward_tokens':1,'communication':False} for i in range(10)]
    entries += [{'id':'new-ordinary','forward_tokens':1,'communication':False},
                {'id':'new-comm','forward_tokens':1,'communication':True}]
    bindings={row['id']:str(i) for i,row in enumerate(entries)}
    history=extend([],['new-ordinary','new-comm'],bindings,source_sha256='source',global_step=58)
    visits=[(0,[i],False) for i in range(12)] + [(i,[11],False) for i in range(1,7)]
    focused=pending(history,bindings,{})
    batches=promote(visits,entries,focused)
    batches=cohort_head_merge(entries,batches,list(range(len(entries))),focus_ids=focused,batch_graphs=1)
    batches=sustained_communication_interleave(entries,batches,focus_ids=focused)
    assert {entries[b[1][0]]['id'] for b in batches[:2]}==focused
    assert sorted((v,tuple(indices)) for v,indices,_ in batches)==sorted((v,tuple(indices)) for v,indices,_ in visits)
    assert any(not entries[b[1][0]]['communication'] for b in batches[2:7])
    assert pending(history,bindings,{'new-ordinary':1,'new-comm':1})==set()


def test_fresh_frontier_interleave_preserves_whole_batches_and_resume_phase():
    entries=[{'id':f'old-{i}'} for i in range(12)]
    entries += [{'id':f'fresh-{i}'} for i in range(4)]
    batches=[(0,[i],False) for i in range(12)]+[(0,[i],True) for i in range(12,16)]
    original=list(batches)
    frontier={row['id'] for row in entries[12:]}
    actual=interleave_batches(batches,entries,frontier,schedule_offset=0,period=5)
    assert len(actual)==len(original)
    assert {id(batch) for batch in actual}=={id(batch) for batch in original}
    assert [batch for batch in actual if batch[1][0]<12]==[batch for batch in original if batch[1][0]<12]
    assert [batch for batch in actual if batch[1][0]>=12]==[batch for batch in original if batch[1][0]>=12]
    assert [batch[1][0]>=12 for batch in actual[:10]]==[False,False,False,False,True,False,False,False,False,True]
    resumed=interleave_batches(batches,entries,frontier,schedule_offset=4,period=5)
    assert resumed[0][1][0]>=12
    assert sorted(index for _,indices,_ in actual for index in indices)==list(range(16))


def test_frontier_interleave_keeps_explicit_focus_and_mixed_batches_ahead():
    entries=[{'id':f'old-{i}'} for i in range(24)]
    entries += [{'id':f'fresh-{i}'} for i in range(8)]
    # Model already-packed four-graph optimizer batches, including one batch
    # containing both an explicit-focus row and a frontier row.
    batches=[(0,[0,1,2,3],False),(0,[4,5,6,7],False),
             (0,[8,9,10,24],False),(0,[12,13,14,15],False),
             (0,[16,17,18,19],False),(0,[20,21,22,23],False),
             (0,[25,26,27,28],False),(0,[29,30,31,11],False)]
    frontier={f'fresh-{i}' for i in range(8)}
    focused={'old-8'}
    actual=interleave_batches(batches,entries,frontier,schedule_offset=1,period=5,priority_ids=focused)
    assert actual[0] is batches[2]  # mixed explicit-focus/frontier batch stays first
    assert actual[1] is batches[0]  # established batches retain stream order
    assert actual[4] is batches[6]  # one fresh slot at global phase 4
    assert [batch for batch in actual if not any(entries[i]['id'] in frontier for i in batch[1])
            and batch is not batches[2]] == [batches[0],batches[1],batches[3],batches[4],batches[5]]
    assert [batch for batch in actual if any(entries[i]['id'] in frontier for i in batch[1])
            and batch is not batches[2]] == [batches[6],batches[7]]
    assert sorted(index for _,indices,_ in actual for index in indices)==list(range(32))
