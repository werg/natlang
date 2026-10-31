from collections import Counter

from natlang_neuralese.train.native_diversity import sustained_communication_interleave


def test_sustains_communication_and_preserves_batches_on_resume():
    entries = ([{'id': f'c{i}', 'communication': True} for i in range(10)]
               + [{'id': f'u{i}', 'training_priority': 1} for i in range(80)]
               + [{'id': f'd{i}'} for i in range(20)])
    batches = [(v, [i], False) for v in range(7) for i in range(10)]
    batches += [(0, [i], False) for i in range(10, 110)]
    for remaining in (batches, batches[15:]):
        result = sustained_communication_interleave(entries, remaining, schedule_offset=52)
        assert Counter((v, tuple(i)) for v, i, _ in result) == Counter((v, tuple(i)) for v, i, _ in remaining)
        width = 20
        for window in (result[:width], result[len(result)//2-width//2:len(result)//2+width//2], result[-width:]):
            assert any(entries[i[0]].get('communication') for _, i, _ in window)
            assert any(not entries[i[0]].get('communication') for _, i, _ in window)
        for visit in {v for v, _, _ in result}:
            positions = [n for n, (v, _, _) in enumerate(result) if v == visit]
            assert [n for n in positions if result[n][2]] == [positions[-1]]


def test_focused_communication_promotes_one_visit_and_spreads_remaining():
    from natlang_neuralese.train.native_diversity import cohort_head_merge
    entries=[{'id':'new','communication':True,'training_priority':1,'forward_tokens':100,'groups':[]},
             {'id':'ordinary','communication':False,'training_priority':0,'forward_tokens':1,'groups':[]}]
    batches=[(visit,[0],False) for visit in range(7)] + [(visit,[1],False) for visit in range(20)]
    merged=cohort_head_merge(entries,batches,['comm','ordinary'],focus_ids={'new'},batch_tokens=10,batch_graphs=1)
    actual=sustained_communication_interleave(entries,merged,focus_ids={'new'})
    assert actual[0][1]==[0]
    assert sum(batch[1]==[0] for batch in actual)==7
    assert any(batch[1]==[1] for batch in actual[1:5])
    assert any(batch[1]==[0] for batch in actual[-8:])
    assert sorted((visit,tuple(indices)) for visit,indices,_ in actual)==sorted((visit,tuple(indices)) for visit,indices,_ in batches)
