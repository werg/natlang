import copy
import pytest
import torch
from natlang_neuralese.model.native_model import CHECKPOINT_SCHEMA
from natlang_neuralese.train.native_continuation import TRAINING_SCHEMA
from natlang_neuralese.train.native_execution_transition import derive, verify_states


def parent():
    return {'schema': CHECKPOINT_SCHEMA, 'model_config': {'dtype':'torch.bfloat16','architecture':'lfm3'},
        'interface_state': {'adapter': {'up':torch.tensor([1.])}},
        'backbone_state': {'weight':torch.tensor([3.])},
        'training': {'schema':TRAINING_SCHEMA,'global_step':67,'optimizer':{'state':torch.tensor([2.])},
                     'rng':{'torch':torch.tensor([42])},'exposure_counts':{'row':4},
                     'sft_progress':{'stage':{'quota':99}},'options':{'source':'bound'}}}


def test_exact_transition_and_tamper_rejection():
    before=parent(); source=copy.deepcopy(before); child=derive(before,parent_sha256='a'*64)
    report=verify_states(source,child,parent_sha256='a'*64)
    assert report['optimizer_updates']==0 and report['fresh_collection_required'] is True
    assert before==source
    assert child['training']['exposure_counts']=={'row':4}
    for key in ('backbone_state','interface_state','training'):
        altered=copy.deepcopy(child)
        if key=='backbone_state':altered[key]['weight'][0]=4
        elif key=='interface_state':altered[key]['adapter']['up'][0]=3
        else:altered[key]['optimizer']['state'][0]=9
        with pytest.raises(ValueError):verify_states(source,altered,parent_sha256='a'*64)
    with pytest.raises(ValueError):derive(child,parent_sha256='a'*64)


def test_file_verification_reloads_and_rejects_changed_bytes(tmp_path, monkeypatch):
    import hashlib
    from natlang_neuralese.rl import native_raw_policy
    from natlang_neuralese.train.native_execution_transition import verify_files
    source=parent();child=derive(source,parent_sha256='a'*64)
    def ref(path):
        return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}
    a=tmp_path/'parent.pt';b=tmp_path/'child.pt';torch.save(source,a)
    parent_ref=ref(a)
    child=derive(source,parent_sha256=parent_ref['sha256']);torch.save(child,b)
    monkeypatch.setattr(native_raw_policy,'saved_native_identity',lambda state,checkpoint_sha256:
        {'checkpoint_sha256':checkpoint_sha256,'profile':state['model_config'].get('native_shared_execution','source-v1')})
    proof=verify_files(parent_ref,ref(b))
    assert proof['actor_identity']['old']['profile']=='source-v1'
    assert proof['actor_identity']['new']['profile']=='ffn-stable-v1'
    bad=copy.deepcopy(child);bad['training']['exposure_counts']['row']+=1;torch.save(bad,b)
    with pytest.raises(ValueError):verify_files(parent_ref,ref(b))
