"""Original approved native records retain exact rendering and mask custody."""
import copy
import json
from pathlib import Path
import pytest

from natlang_neuralese.common.hashing import sha256_hex
from natlang_neuralese.train.native_failed_history import binding,load_manifest,prepare_masks,apply_masks
from natlang_neuralese.data.native_graph import parse_graph
from scripts.neuralese_data.build_native_failed_history_masks import build


def fixture(tmp_path):
    from tokenizers import Tokenizer,models,pre_tokenizers,decoders,AddedToken
    from transformers import PreTrainedTokenizerFast
    from natlang_neuralese.data.text_corpus import native_gold_packet,tokenizer_fingerprint
    from natlang_neuralese.data.native_graph_conversion import text_row_graph
    vocab={token:index for index,token in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    vocab['[UNK]']=len(vocab)
    backend=Tokenizer(models.BPE(vocab,merges=[],unk_token='[UNK]'))
    backend.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False);backend.decoder=decoders.ByteLevel()
    backend.add_tokens([AddedToken(role,single_word=True,normalized=False)for role in('system','user','assistant','tool')])
    template="{% for m in messages %}<|im_start|>{{ m['role'] }}\n{% if m.get('reasoning_content') %}<think>{{ m['reasoning_content'] }}</think>{% endif %}{{ m['content'] }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    tokenizer=PreTrainedTokenizerFast(tokenizer_object=backend,unk_token='[UNK]',
        additional_special_tokens=['<|im_start|>','<|im_sep|>','<|im_end|>'],chat_template=template)
    record={'version':'natlang.teacher_training_turn.native/1','id':'native-decision','split':'train',
        'source_groups':['synthetic-train-only'],'training_admission':{'approved':True,'kind':'synthetic-exact-oracle'},
        'messages':[{'role':'user','content':'Return target.'},{'role':'assistant','content':'bad'},
                    {'role':'tool','content':'rejected action'},{'role':'assistant','content':'good'},
                    {'role':'tool','content':'ok'}], 'target':{'role':'assistant','content':'target'},'tools':[]}
    raw=tmp_path/'native.jsonl';raw.write_text(json.dumps(record)+'\n')
    text,ids,boundary=native_gold_packet(tokenizer,record['messages'],record['target'],[])
    assert tokenizer.decode(ids,skip_special_tokens=False)==text
    fp=tokenizer_fingerprint(tokenizer);identifier='native-view-qualified:full:synthetic-native:native-decision'
    graph=text_row_graph({'id':identifier,'split':'train','source_groups':record['source_groups'],
        'source_record_ids':[record['id']],'tokenizer_sha256':fp,'text':text,'token_ids':ids,
        'supervised_suffix_start':boundary},tokenizer=tokenizer,source_tokenizer_fingerprint=fp,expanded_tokenizer_fingerprint=fp)
    source=tmp_path/'graphs.jsonl';source.write_text(json.dumps(graph)+'\n')
    receipt={'schema':'natlang.native-turn-render/1','corpus_id':'synthetic-native',
        'sources':{str(raw):{key:value for key,value in binding(raw).items()if key!='path'}},
        'tokenizers':{'synthetic':{'files':{'chat_template.jinja':sha256_hex(template.encode())},
            'history_reasoning':{'policy':'keep'}}}}
    receipt_path=tmp_path/'render-receipt.json';receipt_path.write_text(json.dumps(receipt))
    return tokenizer,record,raw,graph,source,receipt,receipt_path


def test_native_record_reconstructs_whole_source_and_preserves_every_target_weight(tmp_path):
    tok,record,raw,graph,source,receipt,receipt_path=fixture(tmp_path)
    ref,counts=build([source],tokenizer=tok,out=tmp_path/'mask',native_render_receipts=[receipt_path])
    manifest=load_manifest(ref)[0];evidence=manifest['rows'][0]['source_evidence']
    assert counts['affected_rows']==1 and evidence['kind']=='verified-native-record-render'
    assert evidence['native_record']['id']==record['id']
    policy={'failed_history_masks':[ref]};prepare_masks(policy)
    example=parse_graph(graph,tokenizer_fingerprint=graph['tokenizer_fingerprint'])
    result,_=apply_masks(example,graph,source_sha256=binding(source)['sha256'],policy=policy)
    boundary=graph['conversion']['source_suffix_start']
    assert result.plan is example.plan and result.id==example.id
    assert result.target_weights[boundary:]==example.target_weights[boundary:]
    positions={i for a,b in manifest['rows'][0]['zero_spans']for i in range(a,b)}
    assert positions and all(b==(0. if i in positions else a)for i,(a,b)in enumerate(zip(example.target_weights,result.target_weights)))


@pytest.mark.parametrize('damage',['unapproved','groups','view','template','source-sha','held','missing-record','tokens'])
def test_native_record_custody_never_infers_missing_approval_or_rendering(tmp_path,damage):
    tok,record,raw,graph,source,receipt,receipt_path=fixture(tmp_path)
    if damage=='unapproved':record['training_admission']['approved']=False
    elif damage=='groups':record['source_groups']=['other-train-group']
    elif damage=='view':graph['id']=graph['id'].replace(':full:',':unknown:')
    elif damage=='template':receipt['tokenizers']['synthetic']['files']['chat_template.jinja']='a'*64
    elif damage=='held':record['split']='test'
    elif damage=='missing-record':record['id']='other-record'
    elif damage=='tokens':
        graph['nodes'][0]['token_ids'][-3]=tok('x',add_special_tokens=False)['input_ids'][0]
        graph['conversion']['source_text_sha256']=sha256_hex(tok.decode(graph['nodes'][0]['token_ids'],skip_special_tokens=False).encode())
    raw.write_text(json.dumps(record)+'\n');source.write_text(json.dumps(graph)+'\n')
    receipt['sources']={str(raw):{key:value for key,value in binding(raw).items()if key!='path'}}
    if damage=='source-sha':receipt['sources'][str(raw)]['sha256']='a'*64
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):build([source],tokenizer=tok,out=tmp_path/'mask',native_render_receipts=[receipt_path])


def test_native_consumer_rejects_changed_proof_and_teacher_fallback(tmp_path):
    tok,*_,graph,source,receipt,receipt_path=fixture(tmp_path)
    ref,_=build([source],tokenizer=tok,out=tmp_path/'mask',native_render_receipts=[receipt_path])
    from natlang_neuralese.train.native_failed_history import validate_source_evidence
    evidence=load_manifest(ref)[0]['rows'][0]['source_evidence']
    for field in('group','boundary','id','split','approval'):
        changed=copy.deepcopy(evidence)
        if field=='group':changed['native_record']['source_groups']=['other']
        elif field=='boundary':changed['native_render']['binding']['source_suffix_start']+=1
        elif field=='id':changed['native_record']['id']='other'
        elif field=='split':changed['native_record']['split']='test'
        else:changed['native_record']['training_admission']['approved']=False
        with pytest.raises(ValueError):validate_source_evidence(graph,changed)
    graph['conversion']['teacher_source']={'source_row_id':'declared-teacher'}
    with pytest.raises(ValueError):validate_source_evidence(graph,evidence)
