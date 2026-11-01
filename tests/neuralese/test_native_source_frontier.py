import pytest
from natlang_neuralese.train.native_source_frontier import extend,pending
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


def test_full_scheduler_first_visits_then_interleaves_remaining_communication():
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
