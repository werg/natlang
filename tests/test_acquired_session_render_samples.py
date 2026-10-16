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


def test_final_registry_proposal_transports_exact_sample_artifact_without_writing_registry(tmp_path):
    import json
    from scripts.neuralese_data.qualify_acquired_session_samples import proposal_with_samples
    from natlang_neuralese.common.hashing import sha256_file_hex
    directory=tmp_path/'native'/'admission';directory.mkdir(parents=True)
    audit=directory/'audit.json';audit.write_text('{}')
    proposal=directory/'registry-proposal.json';proposal.write_text(json.dumps({'id':'cohort',
        'include':['admission/audit.json'],'build':{'audit_sha256':sha256_file_hex(audit)}}))
    sample=directory/'render-samples.json';sample.write_text(json.dumps({'passed':True,
        'cohort_audit':{'path':str(audit),'sha256':sha256_file_hex(audit)}}))
    before=proposal.read_bytes();target=proposal_with_samples(audit,sample);actual=json.loads(target.read_text())
    assert proposal.read_bytes()==before and actual['include']==['admission/audit.json','admission/render-samples.json']
    assert actual['build']['source_render_samples']['sha256']==sha256_file_hex(sample)
    audit.write_text('changed')
    with pytest.raises(ValueError,match='cohort proposal'):proposal_with_samples(audit,sample)
