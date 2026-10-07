"""Gold text uses the serving chat path and pins exact token coordinates."""
import json
from types import SimpleNamespace

import pytest

from natlang_neuralese.data.text_corpus import (
    gold_text_rows, native_gold_document, native_gold_packet, tokenizer_fingerprint,
)
from natlang_neuralese.train.text_warmup import load_text_rows


class Tokenizer:
    all_special_tokens = ['<role>', '</role>', '<|neuralese|>']
    chat_template = 'fixture-native-chat'
    backend_tokenizer = SimpleNamespace(to_str=lambda: '{"normalizer":null}')

    def get_vocab(self):
        return {'<role>': 10000, '</role>': 10001, '<|neuralese|>': 10002}

    def __len__(self):
        self.length_calls=getattr(self,'length_calls',0)+1
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
    assert receipt['rendering']=='natlang.native_gold_chat/2'
    assert all(r['tokenizer_sha256']==tokenizer_fingerprint(tokenizer) for r in rows)
    assert all(p['token_ids_sha256'] for p in provenance)
    path=tmp_path/'text.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    actual,_=load_text_rows(tmp_path/'unused',text_data=path,tokenizer=tokenizer)
    assert actual==rows
    assert tokenizer.length_calls==1
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


def test_suffix_is_exact_native_prefix_divergence_for_tool_target():
    tokenizer=Tokenizer()
    messages=[{'role':'user','content':'quoted <|neuralese|>'}]
    target={'role':'assistant','content':'','tool_calls':[{'type':'function','function':{'name':'eval','arguments':'{"code":"2+2"}'}}]}
    text,ids,start=native_gold_packet(tokenizer,messages,target,[])
    assert (text,ids)==native_gold_document(tokenizer,messages,target,[])
    assert start>0 and start<len(ids)
    assert ''.join(chr(i) for i in ids[start:] if i<10000).startswith('assistant')

def test_invalid_suffix_coordinate_fails_admission(tmp_path):
    tokenizer=Tokenizer()
    rows,*_=gold_text_rows([record('a','train'),record('b','test')],{},tokenizer=tokenizer)
    for bad in [True,-1,len(rows[0]['token_ids'])]:
        rows[0]['supervised_suffix_start']=bad
        path=tmp_path/'bad.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        with pytest.raises(ValueError,match='suffix coordinate'):
            load_text_rows(tmp_path/'unused',text_data=path,tokenizer=tokenizer)


def _record_with_attested_message_body(rid, split, *, block_id='nz1_body', body='Judge each item.'):
    import hashlib
    row=record(rid,split)
    row['messages']=[{'role':'user','content':[{'type':'neuralese','id':block_id}]}]
    row['source_ref']={'inline_instruction_site':{'site':{
        'soft_body_id':'nz1_body', 'raw_body_source':body,
        'raw_body_source_sha256':hashlib.sha256(body.encode()).hexdigest(),
    }}}
    return row


def test_gold_text_expands_only_matching_hash_bound_message_body_once():
    rows,receipt,omissions,_=gold_text_rows([
        _record_with_attested_message_body('a','train'),record('b','test')],{},tokenizer=Tokenizer())
    assert not omissions
    text=next(row['text'] for row in rows if row['id']=='a')
    assert text.count('<|neuralese|>')==1
    assert text.count('<|/neuralese|>')==1
    assert 'Judge each item.' in text
    assert receipt['omitted_records']==0


@pytest.mark.parametrize('mutate',[
    lambda row: row['messages'][0]['content'][0].update(id='nz1_wrong'),
    lambda row: row['source_ref']['inline_instruction_site']['site'].update(raw_body_source_sha256='0'*64),
    lambda row: row['source_ref']['inline_instruction_site']['site'].pop('raw_body_source'),
    lambda row: row['messages'][0]['content'].append({'type':'neuralese','id':'nz1_body'}),
])
def test_gold_text_holds_mismatched_missing_or_ambiguous_message_bodies(mutate):
    bad=_record_with_attested_message_body('bad','train')
    mutate(bad)
    rows,receipt,omissions,_=gold_text_rows([bad,record('valid-train','train'),record('held','test')],{},tokenizer=Tokenizer())
    assert [item['id'] for item in omissions]==['bad']
    assert omissions[0]['reason']=='unresolved_or_malformed_crisp_reference'
    assert receipt['omitted_records']==1


def _soft_state_writer(rid, name, body, *, approved=True):
    row=record(rid,'train')
    row['training_admission']={'approved':approved}
    row['decision']={'training_approved':True,'failed_action':False}
    row['target']={'role':'assistant','tool_calls':[{
        'type':'function','function':{'name':'return_result','arguments':json.dumps({
            'status':'success','value':{'$write':{'name':f'soft-state:{name}',
                'type':'Neuralese<string>','source':body}}})}}]}
    row['source_ref']={'source_row_sha256':'source-row-hash'}
    return row


def test_approved_writer_source_hydrates_reader_context_only_with_provenance():
    writer=_soft_state_writer('writer','nz1_state','Exact generated note text.')
    reader=record('reader','train')
    reader['messages']=[
        {'role':'user','content':[{'type':'text','text':'Prior: '},
                                  {'type':'neuralese','id':'nz1_state'}]},
        {'role':'assistant','tool_calls':[{'type':'function','function':{
            'name':'eval','arguments':[{'type':'text','text':'{"code":"const note = \\"'},
                                        {'type':'neuralese','id':'nz1_state'},
                                        {'type':'text','text':'\\";"}'}]}}]},
    ]
    rows,receipt,omissions,provenance=gold_text_rows(
        [writer,reader,record('held','test')],{},tokenizer=Tokenizer())
    assert not omissions
    assert len(rows)==3  # the input body did not become a separate target row
    reader_row=next(row for row in rows if row['id']=='reader')
    assert reader_row['text'].count('Exact generated note text.')==2
    proof=next(row for row in provenance if row['id']=='reader')['neuralese_context_attestations']
    assert len(proof)==1
    assert proof[0]['block_id']=='nz1_state'
    assert proof[0]['writer_record_id']=='writer'
    assert proof[0]['writer_source_row_sha256']=='source-row-hash'
    assert proof[0]['body_sha256']
    assert receipt['hash_bound_reader_context_blocks']==1


def test_unapproved_or_failed_writer_cannot_hydrate_context():
    writer=_soft_state_writer('writer','nz1_state','Must not leak.',approved=False)
    reader=record('reader','train')
    reader['messages']=[{'role':'user','content':[{'type':'neuralese','id':'nz1_state'}]}]
    rows,receipt,omissions,_=gold_text_rows(
        [writer,reader,record('valid-train','train'),record('held','test')],{},tokenizer=Tokenizer())
    assert [x['id'] for x in omissions]==['writer','reader']
    assert 'Must not leak.' not in ''.join(row['text'] for row in rows)
