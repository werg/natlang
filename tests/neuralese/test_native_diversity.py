from collections import Counter
from natlang_neuralese.train.native_diversity import cohort_head_merge


def row(i, program, tokens=10):
    return {'id':f'text:teacher-program:{program}:decision:{i}', 'forward_tokens':tokens, 'communication':False, 'training_priority':0}

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
