"""Gold text uses the serving chat path and pins exact token coordinates."""
import json
from types import SimpleNamespace

import pytest

from natlang_neuralese.data.text_corpus import (
    gold_text_rows, native_gold_document, tokenizer_fingerprint,
)
from natlang_neuralese.train.text_warmup import load_text_rows


class Tokenizer:
    all_special_tokens = ['<role>', '</role>', '<|neuralese|>']
    chat_template = 'fixture-native-chat'
    backend_tokenizer = SimpleNamespace(to_str=lambda: '{"normalizer":null}')

    def get_vocab(self):
        return {'<role>': 10000, '</role>': 10001, '<|neuralese|>': 10002}

    def __len__(self):
        return 20000

    def __call__(self, text, *, add_special_tokens=False, split_special_tokens=False):
        assert not add_special_tokens
        ids = []
        while text:
            marker = next((s for s in self.all_special_tokens if text.startswith(s)), None)
            if marker and not split_special_tokens:
                ids.append(self.get_vocab()[marker]); text = text[len(marker):]
            else:
                ids.append(ord(text[0])); text = text[1:]
        return {'input_ids': ids}

    def apply_chat_template(self, messages, *, tools=None, tokenize=False, add_generation_prompt=False):
        assert not tokenize and not add_generation_prompt
        return json.dumps(tools or [], sort_keys=True) + ''.join(
            '<role>' + m['role'] + '\n' + (m.get('content') or '') +
            json.dumps(m.get('tool_calls') or [], sort_keys=True) + '</role>' for m in messages)


def record(rid, split):
    return {'id':rid, 'split':split, 'source_groups':[rid],
            'training_admission':{'approved':True},
            'messages':[{'role':'user','content':'Quoted <|neuralese|> is text.'}],
            'target':{'role':'assistant','content':rid}}


def test_native_content_markers_are_plain_tokens_and_structure_is_special():
    tokenizer=Tokenizer()
    messages=[{'role':'user','content':'Quoted <|neuralese|>.'}]
    target={'role':'assistant','content':'done','tool_calls':[{
        'type':'function','function':{'name':'eval','arguments':'{"code":"1 + 2"}'}}]}
    text, ids=native_gold_document(tokenizer,messages,target,[{'type':'function','function':{'name':'eval'}}])
    assert '<|neuralese|>' in text
    assert 10002 not in ids
    assert ids.count(10000)==2 and ids.count(10001)==2
    assert '1 + 2' in text and 'eval' in text
    assert native_gold_document(tokenizer,messages,target,[])==native_gold_document(tokenizer,messages,target,[])


def test_native_rows_keep_splits_gold_and_token_provenance(tmp_path):
    tokenizer=Tokenizer()
    rows, receipt, omissions, provenance=gold_text_rows([record('train-world','train'),record('test-world','test')],{},tokenizer=tokenizer)
    assert not omissions
    assert receipt['rendering']=='natlang.native_gold_chat/1'
    assert all(r['tokenizer_sha256']==tokenizer_fingerprint(tokenizer) for r in rows)
    assert all(p['token_ids_sha256'] for p in provenance)
    path=tmp_path/'text.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    actual,_=load_text_rows(tmp_path/'unused',text_data=path,tokenizer=tokenizer)
    assert actual==rows
    tokenizer.chat_template='different'
    with pytest.raises(ValueError,match='fingerprint mismatch'):
        load_text_rows(tmp_path/'unused',text_data=path,tokenizer=tokenizer)


def test_invalid_native_token_ids_are_not_silently_retokenized(tmp_path):
    tokenizer=Tokenizer()
    rows,*_=gold_text_rows([record('a','train'),record('b','test')],{},tokenizer=tokenizer)
    rows[0]['token_ids']=[True]
    path=tmp_path/'bad.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError,match='token IDs'):
        load_text_rows(tmp_path/'unused',text_data=path,tokenizer=tokenizer)


def test_dedup_preserves_all_source_groups_and_records():
    a=record('a','train'); b=record('b','train')
    b['target']=a['target']
    rows,receipt,_,provenance=gold_text_rows([a,b,record('held','test')],{},tokenizer=Tokenizer())
    train=next(r for r in rows if r['split']=='train')
    assert train['source_groups']==['a','b']
    assert train['source_record_ids']==['a','b']
    assert [r['id'] for r in next(p for p in provenance if p['split']=='train')['source_records']]==['a','b']
    assert receipt['duplicate_same_split_documents_deduplicated']==1


def test_new_corpus_resets_plain_text_reference_without_resetting_phase():
    from natlang_neuralese.train.text_warmup import same_alignment_data
    old={'options':{'records':'r','pieces':'p','text_data':'t'},'inputs':{'r':'R','p':'P','t':'T'}}
    moved={'options':{'records':'r2','pieces':'p2','text_data':'t2'},'inputs':{'r2':'R','p2':'P','t2':'T'}}
    assert same_alignment_data(old,moved)
    moved['inputs']['t2']='native-chat'
    assert not same_alignment_data(old,moved)
