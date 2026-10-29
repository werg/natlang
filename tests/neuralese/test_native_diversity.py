from collections import Counter
from natlang_neuralese.train.native_diversity import cohort_head_merge

def row(i, program, tokens=10):
    return {'id':f'text:teacher-program:{program}:decision:{i}', 'forward_tokens':tokens, 'communication':False, 'training_priority':0}

def test_diverse_stream_uses_monotone_single_reader_and_exact_rows():
    from types import SimpleNamespace
    from natlang_neuralese.train.native_diversity import cohort_head_merge,stream_windows
    entries=[row(0,'a'),row(1,'a'),row(2,'b'),row(3,'b')]
    class Source:
        def __init__(self):self.entries=entries;self.requests=[]
        def iter_indices(self,indices):
            self.requests.append(indices)
            assert indices==sorted(set(indices))
            yield from (SimpleNamespace(id=entries[i]['id']) for i in indices)
    source=Source();grouped=[[b] for b in cohort_head_merge(entries,[(0,[0,1],False),(0,[2,3],True)],['a','a','b','b'],batch_tokens=20)]
    result=list(stream_windows(source,grouped,cohort_ids=['a','a','b','b']))
    assert source.requests==[[0,1],[2,3]]
    assert [[x.id for x in row[-1]] for row in result]==[[entries[i]['id'] for i in row[1]] for row in result]
    assert [row[2] for row in result]==[False,True]

def test_resume_preserves_earned_credit_and_cross_cohort_streams():
    from types import SimpleNamespace
    from natlang_neuralese.train.native_diversity import cohort_head_merge,stream_windows
    entries=[row(0,'a'),row(1,'a'),row(2,'b'),row(3,'b')]
    class Source:
        def __init__(self):self.entries=entries;self.requests=[]
        def iter_indices(self,indices):
            self.requests.append(indices);assert indices==sorted(set(indices))
            yield from (SimpleNamespace(id=entries[i]['id']) for i in indices)
    cohorts=['old','old','new','new'];original=[(0,[0,1],False),(0,[2,3],True)]
    source=Source();schedule=list(stream_windows(source,[[b] for b in cohort_head_merge(entries,original,cohorts,batch_tokens=20)],cohort_ids=cohorts))
    earned=set(schedule[0][1]);pending=[i for i in range(4) if i not in earned]
    resumed=Source();remaining=list(stream_windows(resumed,[[(0,pending,True)]],cohort_ids=cohorts))
    assert earned.isdisjoint(remaining[0][1]) and earned|set(remaining[0][1])==set(range(4))
    assert all(x==sorted(set(x)) for x in source.requests+resumed.requests)

def test_disabled_fairness_keeps_physical_readers_and_iterator_windows():
    from types import SimpleNamespace
    from natlang_neuralese.train.native_diversity import stream_windows
    entries=[row(0,'a'),row(1,'a'),row(2,'b'),row(3,'b')]
    class Source:
        def __init__(self):self.entries=entries;self.requests=[]
        def iter_indices(self,indices):
            self.requests.append(indices)
            assert indices==sorted(set(indices))
            yield from (SimpleNamespace(id=entries[i]['id']) for i in indices)
    source=Source()
    windows=iter([[(0,[0,2],False)],[(0,[1,3],True)]])
    result=list(stream_windows(source,windows,cohort_ids=['a','a','b','b'],
        urgent_cohort_fairness=False,ordinary_cohort_fairness=False,
        communication_cohort_fairness=False))
    assert [x[1] for x in result]==[[0,2],[1,3]]
    assert source.requests==[[0,1],[2,3]]

def test_template_ids_are_authenticated_group_fields():
    from natlang_neuralese.train.native_diversity import task_key
    e=row(0,'x');e['groups']=['inline-curriculum:iterate_records:world-1:request']
    assert task_key(e)=='iterate_records'
    e['groups']=['inline:iterate_records']
    assert task_key(e) is None

def test_monotone_pair_mixes_without_source_rewind():
    from natlang_neuralese.train.native_diversity import cohort_head_merge
    entries=[row(0,'a'),row(1,'a'),row(2,'b'),row(3,'b')]
    original=[(0,[0,1],False),(0,[2,3],True)];cohorts=['a','a','b','b']
    result=cohort_head_merge(entries,original,cohorts,batch_tokens=20)
    assert result==[(0,[0,2],False),(0,[1,3],True)]
    for group in set(cohorts):
        selected=[k for _,ix,_ in result for k in ix if cohorts[k]==group]
        assert selected==sorted(selected)
    assert Counter((v,k) for v,ix,_ in result for k in ix)==Counter((v,k) for v,ix,_ in original for k in ix)

def test_focused_batch_uses_one_cursor_for_all_its_rows():
    from types import SimpleNamespace
    from natlang_neuralese.train.native_diversity import stream_windows
    entries=[row(i,'a') for i in range(4)]
    class Source:
        def __init__(self):self.entries=entries;self.requests=[]
        def iter_indices(self,indices):
            self.requests.append(indices);assert indices==sorted(set(indices))
            yield from (SimpleNamespace(id=entries[i]['id']) for i in indices)
    source=Source();grouped=[[(0,[2,3],False)],[(0,[0,1],True)]]
    result=list(stream_windows(source,grouped,cohort_ids=['same']*4,focus_ids={entries[2]['id']}))
    assert [x[1] for x in result]==[[2,3],[0,1]]
    assert source.requests==[[2,3],[0,1]]

def test_cohort_head_repacking_preserves_slots_and_monotone_heads():
    from natlang_neuralese.train.native_diversity import cohort_head_merge
    entries=[row(0,'a'),row(1,'a'),row(2,'b'),row(3,'b')]
    source=[(0,[0,1],False),(0,[2,3],True)];cohorts=['a','a','b','b']
    result=cohort_head_merge(entries,source,cohorts,batch_tokens=20)
    assert result==[(0,[0,2],False),(0,[1,3],True)]
    for group in set(cohorts):
        indices=[i for _,batch,_ in result for i in batch if cohorts[i]==group]
        assert indices==sorted(indices)
    assert Counter((v,i) for v,ix,_ in result for i in ix)==Counter((v,i) for v,ix,_ in source for i in ix)

def test_repacked_schedule_enforces_four_to_one_while_both_lanes_pending():
    from natlang_neuralese.train.native_diversity import cohort_head_merge
    entries=[];source=[]
    for i in range(20):
        e=row(i,str(i));e['communication']=(i%5!=4);entries.append(e);source.append((0,[i],i==19))
    result=cohort_head_merge(entries,source,[str(i) for i in range(20)],batch_graphs=1,schedule_offset=0)
    assert [entries[ix[0]]['communication'] for _,ix,_ in result]==[i%5!=4 for i in range(20)]

def test_reader_checks_actual_descriptor_headroom(monkeypatch):
    import pytest
    from natlang_neuralese.train import native_diversity as diversity
    entries=[row(0,'a')]
    class Source:
        def __init__(self):self.entries=entries
        def iter_indices(self,indices):raise AssertionError('should fail before opening reader')
    monkeypatch.setattr(diversity.resource,'getrlimit',lambda _: (10,10))
    monkeypatch.setattr(diversity.os,'listdir',lambda _: ['0','1'])
    with pytest.raises(ValueError,match='file descriptors'):
        list(diversity.stream_windows(Source(),[[(0,[0],True)]],cohort_ids=['a']))
