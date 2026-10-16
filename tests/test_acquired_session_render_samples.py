"""Source spotchecks compare full actual graph values, not metadata alone."""
import copy
import pytest
from test_acquired_session_admission import admitted
from scripts.neuralese_data.qualify_acquired_session_samples import compare_views


def test_both_production_tokenizer_views_match_exact_and_reject_changed_tokens_or_weights(admitted):
    _,graphs,_=admitted
    assert len(compare_views(graphs,copy.deepcopy(graphs)))==len(graphs)
    for field in ('token_ids','target_weights'):
        changed=copy.deepcopy(graphs)
        changed[0]['nodes'][0][field][-1]+=1
        with pytest.raises(ValueError,match='tokens/weights/provenance'):compare_views(graphs,changed)
    with pytest.raises(ValueError,match='inventory'):compare_views(graphs,graphs+graphs[:1])
    if len(graphs)>1:
        with pytest.raises(ValueError,match='inventory'):compare_views(graphs,graphs[:-1])
