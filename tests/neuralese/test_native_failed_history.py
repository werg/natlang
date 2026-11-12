"""Same-ID failed-token repair with portable exact custody and resumed state."""
import copy
import json
from pathlib import Path
import pytest
import torch

from natlang_neuralese.common.hashing import sha256_hex
from natlang_neuralese.train.native_failed_history import binding,load_manifest
from natlang_neuralese.train.native_supervision import load_policy,apply_policy
from natlang_neuralese.data.native_graph import parse_graph


def tokenizer_fixture(path):
    from tokenizers import Tokenizer,decoders
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import PreTrainedTokenizerFast
    words=['unknown','<|im_start|>','<|im_sep|>','<|im_end|>','system','user','assistant','tool',
           'input','bad','good','target','rejected','ok','<tool_response>','</tool_response>']
    vocab={word:index for index,word in enumerate(words)}
    vocab.update({f't{index}':index for index in range(len(vocab),64)})
    backend=Tokenizer(WordLevel(vocab,unk_token='unknown'));backend.pre_tokenizer=WhitespaceSplit()
    backend.decoder=decoders.Fuse()
    tokenizer=PreTrainedTokenizerFast(tokenizer_object=backend,unk_token='unknown',
        additional_special_tokens=['<|im_start|>','<|im_sep|>','<|im_end|>'])
    tokenizer.save_pretrained(path)
    return tokenizer


def fixture(tmp_path,*,whole=False,tokenizer=None):
    from natlang_neuralese.data.native_graph_conversion import text_row_graph
    from natlang_neuralese.data.text_corpus import tokenizer_fingerprint
    from scripts.neuralese_data.build_native_failed_history_masks import build
    tmp_path.mkdir(parents=True,exist_ok=True)
    if tokenizer is None:tokenizer=tokenizer_fixture(tmp_path/'original-tokenizer')
    fp=tokenizer_fingerprint(tokenizer)
    prefix=('<|im_start|> user <|im_sep|> input <|im_end|> '
            '<|im_start|> assistant <|im_sep|> bad <|im_end|> '
            '<|im_start|> tool <|im_sep|> rejected <|im_end|> '
            '<|im_start|> assistant <|im_sep|> good <|im_end|> '
            '<|im_start|> tool <|im_sep|> ok <|im_end|> '
            '<|im_start|> assistant <|im_sep|> ')
    text=prefix+'target <|im_end|>';ids=tokenizer(text,add_special_tokens=False)['input_ids']
    source={'id':'already-earned','split':'train','source_groups':['train-only'],
        'tokenizer_sha256':fp,'text':tokenizer.decode(ids,skip_special_tokens=False),'token_ids':ids,
        'supervised_suffix_start':len(tokenizer(prefix,add_special_tokens=False)['input_ids'])}
    graph=text_row_graph(source,tokenizer=tokenizer,source_tokenizer_fingerprint=fp,expanded_tokenizer_fingerprint=fp)
    raw={'id':'original','trajectory':[{'assistant':{'content':'target'},
        'model_response':{'raw_calls':[]},'context':[{'role':'user','content':'input'}],'tools_offered':[]}]}
    raw_path=tmp_path/'raw.jsonl';line=json.dumps(raw);raw_path.write_text(line+'\n')
    graph['conversion']['teacher_source']={'source_row_id':raw['id'],'source_row_sha256':sha256_hex(line.encode()),
        'source_artifact':str(raw_path),'source_artifact_sha256':binding(raw_path)['sha256'],'teacher_turn_index':0}
    if whole:
        graph['conversion'].update(role_interpretation='whole-trajectory-chat-roles',
            supervision_policy={'failed_assistant_weight':0.,'weighting':'segmented_position_weights'})
    path=tmp_path/'graphs.jsonl';path.write_text(json.dumps(graph)+'\n')
    reference,counts=build([path],tokenizer=tokenizer,out=tmp_path/'mask')
    value={'schema':'natlang.native-ce-supervision-policy/1','sources':[]if whole else[
        {'sha256':binding(path)['sha256'],'encoding':'ordinary-suffix-raw-v1','context_weight':.05}],
        'failed_history_masks':[reference]}
    policy_path=tmp_path/'policy.json';policy_path.write_text(json.dumps(value))
    return graph,path,reference,policy_path,counts


@pytest.mark.parametrize('whole',[False,True])
def test_original_source_and_every_unfailed_raw_weight_survive(tmp_path,whole):
    graph,path,reference,policy_path,counts=fixture(tmp_path,whole=whole)
    from natlang_neuralese.train.native_failed_history import apply_masks
    from natlang_neuralese.data.native_sources import NativeGraphSource
    before=copy.deepcopy(graph);policy=load_policy(binding(policy_path))
    example=parse_graph(graph,tokenizer_fingerprint=graph['tokenizer_fingerprint'])
    masked,raw=apply_masks(example,graph,source_sha256=binding(path)['sha256'],policy=policy)
    assert graph==before and masked.id==example.id and masked.plan is example.plan
    spans=load_manifest(reference)[0]['rows'][0]['zero_spans'];positions={i for a,b in spans for i in range(a,b)}
    assert positions and counts['affected_rows']==1
    assert all(b==(0. if i in positions else a)for i,(a,b)in enumerate(zip(example.target_weights,masked.target_weights)))
    assert raw['nodes'][0]['target_weights']==list(masked.target_weights)
    source=NativeGraphSource(path,graph['tokenizer_fingerprint'],required_split='train')
    bindings=copy.deepcopy(source.bindings);source.set_ce_policy(policy)
    result=next(iter(source));assert source.bindings==bindings and result.id==graph['id']
    assert all(result.target_weights[i]==0 for i in positions)
    if not whole:
        boundary=graph['conversion']['source_suffix_start']
        assert sum(result.target_weights[boundary:])/sum(result.target_weights)>.95
    else:assert result.target_weights==masked.target_weights
    # Held has no failed-mask path, even for an explicit manifest on the policy.
    held=copy.deepcopy(graph);held['split']='test'
    held_example=parse_graph(held,tokenizer_fingerprint=graph['tokenizer_fingerprint'])
    assert apply_policy(held_example,held,source_sha256=binding(path)['sha256'],policy=policy) is held_example


@pytest.mark.parametrize('damage',['span','feedback','raw-row','guard','groups','fingerprint','source-split','asset'])
def test_tampered_masks_or_decoder_fail_before_training(tmp_path,damage):
    graph,path,reference,policy_path,_=fixture(tmp_path)
    manifest_path=Path(reference['path']);value=json.loads(manifest_path.read_text());row=value['rows'][0]
    if damage=='span':row['zero_spans'][0][0]+=1
    elif damage=='feedback':row['failure_evidence'][0]['feedbacks'][0]['tokens_sha256']='a'*64
    elif damage=='raw-row':row['raw_row_sha256']='a'*64
    elif damage=='guard':row['source_evidence']['structured_input_guard']['policy']='invented'
    elif damage=='groups':row['source_groups']=['held']
    elif damage=='fingerprint':value['tokenizer']['fingerprint']='a'*64
    elif damage=='source-split':value['sources'][0]['split']='test'
    else:
        asset=manifest_path.parent/value['tokenizer']['files'][0]['path'];asset.write_bytes(asset.read_bytes()+b' ')
    manifest_path.write_text(json.dumps(value));policy=json.loads(policy_path.read_text());policy['failed_history_masks']=[binding(manifest_path)]
    policy_path.write_text(json.dumps(policy))
    with pytest.raises(ValueError):
        loaded=load_policy(binding(policy_path));example=parse_graph(graph,tokenizer_fingerprint=graph['tokenizer_fingerprint'])
        apply_policy(example,graph,source_sha256=binding(path)['sha256'],policy=loaded)


def test_identical_verified_decoder_snapshots_reuse_one_load_without_skipping_asset_checks(tmp_path,monkeypatch):
    from transformers import AutoTokenizer
    from natlang_neuralese.train.native_failed_history import prepare_masks,apply_masks,_decoder
    tokenizer=tokenizer_fixture(tmp_path/'original-tokenizer')
    fixtures=[fixture(tmp_path/f'family-{index}',tokenizer=tokenizer)for index in range(4)]
    policy={'failed_history_masks':[item[2]for item in fixtures]}
    _decoder.cache_clear();calls=[];original=AutoTokenizer.from_pretrained
    def load(*args,**kwargs):
        calls.append(args[0]);return original(*args,**kwargs)
    monkeypatch.setattr(AutoTokenizer,'from_pretrained',load)
    prepare_masks(policy)
    for graph,source,*_ in fixtures*2:
        example=parse_graph(graph,tokenizer_fingerprint=graph['tokenizer_fingerprint'])
        result,_=apply_masks(example,graph,source_sha256=binding(source)['sha256'],policy=policy)
        assert result.target_weights!=example.target_weights
    assert len(calls)==1
    # Equivalent content never excuses checking each independent bound copy.
    manifest,_,_=load_manifest(fixtures[-1][2])
    asset=Path(fixtures[-1][2]['path']).parent/manifest['tokenizer']['files'][0]['path']
    asset.write_bytes(asset.read_bytes()+b' ')
    with pytest.raises(ValueError):prepare_masks({'failed_history_masks':policy['failed_history_masks']})
    _decoder.cache_clear()


def test_evidence_addition_leaves_old_mask_identical():
    from test_failed_assistant_context import fixture as old_fixture
    from natlang_neuralese.train.text_supervision import chat_roles,failed_assistant_mask
    source,options=old_fixture();tok=options['tokenizer'];ids=source['token_ids']
    labels=chat_roles(ids,start_id=tok.convert_tokens_to_ids('<|im_start|>'),
        role_ids={tok.convert_tokens_to_ids(role):role for role in ('system','user','assistant','tool')})
    decode=lambda ids:tok.decode(ids,skip_special_tokens=False)
    evidence=[];assert failed_assistant_mask(ids,labels,decode)==failed_assistant_mask(ids,labels,decode,evidence=evidence)
    assert len(evidence)==1 and evidence[0]['feedbacks']


@pytest.mark.parametrize('family',['lfm'])
def test_both_port_loss_and_every_derivative_equal_explicit_zero_before_mass(tmp_path,family):
    from dataclasses import replace
    from test_shared_state import tiny
    from natlang_neuralese.model.shared_state import SharedStateModel
    from natlang_neuralese.train.native_graph_loss import packed_graph_loss
    from natlang_neuralese.train.native_supervision import _apply_declaration
    graph,path,reference,policy_path,_=fixture(tmp_path)
    example=parse_graph(graph,tokenizer_fingerprint=graph['tokenizer_fingerprint'])
    policy=load_policy(binding(policy_path))
    actual=apply_policy(example,graph,source_sha256=binding(path)['sha256'],policy=policy)
    raw=copy.deepcopy(graph)
    for start,stop in load_manifest(reference)[0]['rows'][0]['zero_spans']:
        raw['nodes'][0]['target_weights'][start:stop]=[0.]*(stop-start)
    masked=replace(example,target_weights=tuple(raw['nodes'][0]['target_weights']))
    oracle=_apply_declaration(masked,raw,policy['sources'][0])
    assert actual.target_weights==oracle.target_weights
    model=SharedStateModel(tiny(family),marker_ids=(58,59,60,61),rank=4)
    parameters=tuple(p for p in model.parameters()if p.requires_grad)
    losses=[packed_graph_loss(model,[item],checkpoint_layers=True,chunk_tokens=3)[0]for item in (actual,oracle)]
    assert torch.equal(losses[0],losses[1])
    gradients=[torch.autograd.grad(loss,parameters,allow_unused=True)for loss in losses]
    for left,right in zip(*gradients):
        if left is None:assert right is None
        else:assert torch.equal(left,right)


# Written for one optimizer update per packed batch; accumulation is tested in test_native_update_accumulation.py.
pytestmark = pytest.mark.usefixtures('per_batch_native_updates')
