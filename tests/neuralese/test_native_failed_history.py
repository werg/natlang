"""Same-ID failed-token repair with portable exact custody and resumed state."""
import copy
import json
from pathlib import Path
import pytest
import torch

from natlang_neuralese.common.hashing import canonical_json_sha256_hex as digest,sha256_hex
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


def test_portable_decoder_closure_needs_no_original_capture_at_consumer(tmp_path):
    import shutil
    from natlang_neuralese.train.native_supervision import ce_policy_dependencies
    from scripts.coreweave.provider_dependencies import dependency_closure
    graph,path,reference,policy_path,_=fixture(tmp_path)
    moved=tmp_path/'moved';shutil.copytree(tmp_path/'mask',moved)
    shutil.rmtree(tmp_path/'mask');shutil.rmtree(tmp_path/'original-tokenizer');(tmp_path/'raw.jsonl').unlink()
    policy=json.loads(policy_path.read_text());policy['failed_history_masks']=[binding(moved/'manifest.json')]
    policy_path.write_text(json.dumps(policy));loaded=load_policy(binding(policy_path))
    example=parse_graph(graph,tokenizer_fingerprint=graph['tokenizer_fingerprint'])
    assert apply_policy(example,graph,source_sha256=binding(path)['sha256'],policy=loaded).target_weights!=example.target_weights
    config=tmp_path/'config.json';config.write_text(json.dumps({'ce_supervision_policy':binding(policy_path)}))
    closure=dependency_closure({'config':binding(config)},namespace=str(tmp_path))
    expected={str(config)}|{item['path']for item in ce_policy_dependencies(binding(policy_path))}
    assert {item['path']for item in closure['files']}==expected
    asset=moved/json.loads((moved/'manifest.json').read_text())['tokenizer']['files'][0]['path'];asset.unlink()
    with pytest.raises((ValueError,FileNotFoundError)):ce_policy_dependencies(binding(policy_path))


def test_capture_custody_refuses_wrong_raw_line(tmp_path):
    from scripts.neuralese_data.build_native_failed_history_masks import capture_evidence
    graph,*_=fixture(tmp_path)
    raw=json.loads((tmp_path/'raw.jsonl').read_text())
    with pytest.raises(ValueError,match='capture row identity'):capture_evidence(graph,raw,'a'*64)


def test_evidence_addition_leaves_old_mask_identical():
    from test_failed_assistant_context import fixture as old_fixture
    from natlang_neuralese.train.text_supervision import chat_roles,failed_assistant_mask
    source,options=old_fixture();tok=options['tokenizer'];ids=source['token_ids']
    labels=chat_roles(ids,start_id=tok.convert_tokens_to_ids('<|im_start|>'),
        role_ids={tok.convert_tokens_to_ids(role):role for role in ('system','user','assistant','tool')})
    decode=lambda ids:tok.decode(ids,skip_special_tokens=False)
    evidence=[];assert failed_assistant_mask(ids,labels,decode)==failed_assistant_mask(ids,labels,decode,evidence=evidence)
    assert len(evidence)==1 and evidence[0]['feedbacks']


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_both_port_loss_and_every_derivative_equal_explicit_zero_before_mass(tmp_path,family):
    from dataclasses import replace
    from test_shared_state import tiny
    from natlang_neuralese.model.shared_state import SharedStateModel
    from natlang_neuralese.train.native_state_sft import packed_graph_loss
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


def test_real_shared_trainer_resume_keeps_optimizer_rng_original_ids_and_credits(tmp_path,monkeypatch):
    from test_native_serve import _tiny_source
    from natlang_neuralese.model.native_model import load_native_model
    from natlang_neuralese.train import native_state_sft as trainer
    from natlang_neuralese.train.checkpoint_offload import _same_values
    base=tmp_path/'base';_tiny_source(base);tokenizer_fixture(base)
    _,tokenizer,_,_=load_native_model(base,dtype=torch.float32,rank=4)
    graph,path,_,policy_path,_=fixture(tmp_path/'data',tokenizer=tokenizer)
    held=copy.deepcopy(graph);held.update(id='held-only',split='test',source_groups=['protected-test'])
    held_path=tmp_path/'held.jsonl';held_path.write_text(json.dumps(held)+'\n')
    out=tmp_path/'out'
    argv=['--base',str(base),'--dtype','float32','--rank','4','--device','cpu',
        '--native-graphs',str(path),'--held-native-graphs',str(held_path),
        '--epochs','3','--eval-every','100','--out',str(out)]
    def after_one(policy,step,**kwargs):policy.stop.requested=True;return False
    monkeypatch.setattr(trainer.CheckpointPolicy,'due',after_one)
    assert trainer.main(argv)==0
    before=torch.load(out/'checkpoint.pt',weights_only=False)['training']
    factory=trainer.make_native_optimizer;restored=[]
    def observe(*args,**kwargs):
        optimizer,names=factory(*args,**kwargs);restore=optimizer.load_state_dict
        def load(state,**options):
            _same_values(state,before['optimizer']);restored.append(True)
            return restore(state,**options)
        optimizer.load_state_dict=load
        return optimizer,names
    original_restore=trainer.restore_training_rng_state;rng_restored=[]
    def restore_rng(state,device):
        if not rng_restored:
            _same_values(state,before['rng']);rng_restored.append(True)
        return original_restore(state,device)
    monkeypatch.setattr(trainer,'make_native_optimizer',observe)
    monkeypatch.setattr(trainer,'restore_training_rng_state',restore_rng)
    assert trainer.main(argv+['--ce-supervision-policy',str(policy_path)])==0
    after=torch.load(out/'checkpoint.pt',weights_only=False)['training']
    assert restored==[True]and rng_restored==[True]
    assert after['step']==before['step']+1==2
    assert after['exposure_counts']=={graph['id']:2}
    assert after['source_row_bindings']==before['source_row_bindings']
    assert after['options']['ce_supervision_policy_history'][-1]['from'] is None
    monkeypatch.setattr(trainer,'make_native_optimizer',factory)
    monkeypatch.setattr(trainer,'restore_training_rng_state',original_restore)
    assert trainer.main(argv)==0
    inherited=torch.load(out/'checkpoint.pt',weights_only=False)['training']
    assert inherited['step']==3 and inherited['exposure_counts']=={graph['id']:3}
    assert inherited['options']['ce_supervision_policy_binding']==after['options']['ce_supervision_policy_binding']


def test_append_masks_to_existing_policy_without_rescanning_original_sources(tmp_path):
    from scripts.neuralese_data.build_native_ce_supervision_policy import main
    graph,path,reference,policy_path,_=fixture(tmp_path)
    original=json.loads(policy_path.read_text());original.pop('failed_history_masks')
    prior=tmp_path/'prior.json';prior.write_text(json.dumps(original))
    out=tmp_path/'continued.json'
    # Source is intentionally absent: attaching an exact mask does not parse it.
    path.unlink()
    assert main(['--prior-policy',str(prior),'--failed-history-mask-manifest',reference['path'],'--out',str(out)])==0
    continued=json.loads(out.read_text())
    assert continued['sources']==original['sources']and continued['failed_history_masks']==[reference]


def test_stream_preparation_uses_one_derivation_and_constant_fingerprint_work(tmp_path,monkeypatch):
    from scripts.neuralese_data import build_native_failed_history_masks as producer
    from natlang_neuralese.train.native_failed_history import row_mask_record
    graph,path,reference,_,_=fixture(tmp_path)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(tmp_path/'original-tokenizer',local_files_only=True)
    graph2=copy.deepcopy(graph);graph2['id']='second-earned-row'
    path.write_text(json.dumps(graph)+'\n'+json.dumps(graph2)+'\n')
    fingerprint=producer.tokenizer_fingerprint;derive=producer._mask;calls={'fingerprint':0,'mask':0}
    def measured_fingerprint(tokenizer):calls['fingerprint']+=1;return fingerprint(tokenizer)
    def measured_mask(row,tokenizer):calls['mask']+=1;return derive(row,tokenizer)
    monkeypatch.setattr(producer,'tokenizer_fingerprint',measured_fingerprint)
    monkeypatch.setattr(producer,'_mask',measured_mask)
    out=tmp_path/'optimized';actual,counts=producer.build([path],tokenizer=tokenizer,out=out)
    value=load_manifest(actual)[0]
    assert calls=={'fingerprint':3,'mask':2}and counts['affected_rows']==2
    # Exact former per-row path is the reference, with unchanged full source binding.
    for row,record in zip((graph,graph2),value['rows']):
        expected=row_mask_record(row,tokenizer,source_sha256=binding(path)['sha256'],
            source_evidence=record['source_evidence'])
        assert record==expected
    bad=copy.deepcopy(graph);bad['tokenizer_fingerprint']='a'*64
    path.write_text(json.dumps(bad)+'\n')
    with pytest.raises(ValueError,match='actual source row tokenizer'):
        producer.build([path],tokenizer=tokenizer,out=tmp_path/'bad-fingerprint')
