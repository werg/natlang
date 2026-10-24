import json

import pytest

from test_native_failed_history import fixture
from natlang_neuralese.train import text_supervision
from natlang_neuralese.train.native_failed_history import binding,load_manifest,apply_masks
from natlang_neuralese.common.hashing import canonical_json_sha256_hex as digest
from natlang_neuralese.data.native_graph_schema import parse_graph
from natlang_neuralese.train.native_supervision import load_policy
from scripts.neuralese_data.continue_failed_history_mask_law import continue_law


def historical(tmp_path, change):
    graph,path,reference,policy_path,_=fixture(tmp_path/'source')
    helper=tmp_path/'old-helper.py';helper.write_text(change(open(text_supervision.__file__).read()))
    manifest=json.loads(open(reference['path']).read())
    manifest['mask_law_sha256']=digest({'helper_sha256':binding(helper)['sha256'],
        'failure_patterns':[(p.pattern,p.flags)for p in text_supervision.FAILED_TOOL_FEEDBACK]})
    open(reference['path'],'w').write(json.dumps(manifest))
    reference=binding(reference['path'])
    return graph,path,reference,policy_path,helper,manifest


def test_unrelated_helper_change_preserves_real_mask_and_current_consumer(tmp_path):
    graph,path,reference,policy_path,helper,original=historical(tmp_path,lambda s:s+'\n# unrelated helper revision\n')
    result=continue_law(reference,helper,tmp_path/'continued')
    value,rows,_=load_manifest(result)
    assert value['rows']==original['rows']
    assert value['sources']==original['sources']
    assert not value['mask_law_continuation']['mask_values_recomputed']
    policy=json.loads(policy_path.read_text());policy['failed_history_masks']=[result]
    policy_path.write_text(json.dumps(policy));loaded=load_policy(binding(policy_path))
    example=parse_graph(graph,tokenizer_fingerprint=graph['tokenizer_fingerprint'])
    changed,_=apply_masks(example,graph,source_sha256=binding(path)['sha256'],policy=loaded)
    assert changed.id==example.id and changed.target_weights!=example.target_weights
    assert continue_law(reference,helper,tmp_path/'continued')==result


@pytest.mark.parametrize('change',[
    lambda s:s.replace("body = body.lstrip()","body = body.strip()"),
    lambda s:s.replace("r'\\Arejected\\b'","r'\\Aaccepted\\b'"),
    lambda s:s.replace("role = 'other'","role = 'user'"),
    lambda s:s.replace("'assistant_reasoning', 'assistant_reply'","'assistant_reply', 'assistant_reasoning'"),
])
def test_changed_role_detector_or_global_law_is_rejected(tmp_path,change):
    _,_,reference,_,helper,_=historical(tmp_path,change)
    with pytest.raises(ValueError,match='callable law changed'):
        continue_law(reference,helper,tmp_path/'continued')
