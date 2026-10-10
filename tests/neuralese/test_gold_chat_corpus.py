"""Gold text uses the serving chat path and pins exact token coordinates."""
import json
from types import SimpleNamespace

import pytest

from natlang_neuralese.data.text_corpus import (
    gold_text_preview_rows, gold_text_rows, native_gold_document, native_gold_packet, tokenizer_fingerprint,
)
from natlang_neuralese.serve.chat import render_messages, split_escaped
from natlang_neuralese.train.text_warmup import load_text_rows


def serving_prompt_ids(tokenizer, messages, tools):
    rendered=render_messages(messages,tools,lambda turns,schemas:tokenizer.apply_chat_template(
        turns,tools=schemas or None,tokenize=False,add_generation_prompt=True),
        specials=(*tokenizer.all_special_tokens,'<|neuralese|>','<|/neuralese|>'))
    return [token for segment in rendered.segments for run,escaped in split_escaped(segment,rendered.escape_nonce)
            for token in tokenizer(run,add_special_tokens=False,split_special_tokens=escaped)['input_ids']]


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
        assert not tokenize
        rendered = json.dumps(tools or [], sort_keys=True) + ''.join(
            '<role>' + m['role'] + '\n' + (m.get('content') or '') +
            json.dumps(m.get('tool_calls') or [], sort_keys=True) + '</role>' for m in messages)
        return rendered + ('<role>assistant\n' if add_generation_prompt else '')


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
    assert receipt['rendering']=='natlang.native_gold_chat/3'
    assert receipt['history_reasoning']=={'policy':'undeclared','backbone':None,'template_kwargs':{},'probe':'not-rendered'}
    assert all(r['tokenizer_sha256']==tokenizer_fingerprint(tokenizer) for r in rows)
    assert all(p['token_ids_sha256'] for p in provenance)
    path=tmp_path/'text.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    actual,_=load_text_rows(tmp_path/'unused',text_data=path,tokenizer=tokenizer)
    # Loaded token IDs are stored compactly as unsigned arrays (881ff257); the coordinates are unchanged.
    assert all(r['token_ids'].typecode=='I' for r in actual)
    assert [{**r,'token_ids':list(r['token_ids'])} for r in actual]==rows
    assert tokenizer.length_calls==1
    tokenizer.chat_template='different'
    with pytest.raises(ValueError,match='fingerprint mismatch'):
        load_text_rows(tmp_path/'unused',text_data=path,tokenizer=tokenizer)


def test_preview_renderer_reuses_gold_format_but_never_marks_held_rows_eligible():
    tokenizer = Tokenizer()
    approved_train = record('same-train', 'train')
    approved_test = record('same-test', 'test')
    held_train = record('same-train', 'train')
    held_test = record('same-test', 'test')
    for held in (held_train, held_test):
        held.pop('training_admission')
        held['review_disposition'] = 'held_for_root_review'
        held['decision'] = {'training_approved': False, 'failed_action': False}

    approved_rows, approved_receipt, approved_omissions, _ = gold_text_rows(
        [approved_train, approved_test], {}, tokenizer=tokenizer)
    preview_rows, preview_receipt, preview_omissions, _ = gold_text_preview_rows(
        [held_train, held_test], {}, tokenizer=tokenizer)

    assert not approved_omissions and not preview_omissions
    assert [row['text'] for row in preview_rows] == [row['text'] for row in approved_rows]
    assert preview_receipt['format'] == 'natlang.gold_text_preview_receipt/1'
    assert preview_receipt['status'] == 'held-review-only'
    assert preview_receipt['review_only'] is True
    assert preview_receipt['sft_eligible'] is False
    assert preview_receipt['training_admission_granted'] is False
    assert approved_receipt['format'] == 'natlang.gold_text_packet_receipt/1'
    assert approved_receipt['sft_eligible'] is True

    unmarked = record('unmarked', 'train')
    unmarked.pop('training_admission')
    _, _, rejected, _ = gold_text_preview_rows(
        [unmarked, held_train, held_test], {}, tokenizer=tokenizer)
    assert rejected[0]['reason'] == 'not_explicitly_held_review_record'


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
    policy={'mask_system_prompt':True,'held_documents':16,'tokens':16384,'prefix_tokens':32,
            'qualification_cohort':'native','rollout_passes':0,'rollout_start_passes':4,'max_sequence_passes':3}
    old={'options':{'records':'r','pieces':'p','text_data':'t',**policy},
         'inputs':{'r':'R','p':'P','t':'T'},'target':'gold-token','text_history':'gold-history',
         'supervision_policy':'token-ce'}
    moved={'options':{'records':'r2','pieces':'p2','text_data':'t2',**policy},
           'inputs':{'r2':'R','p2':'P','t2':'T'},'target':'gold-token',
           'text_history':'gold-history','supervision_policy':'token-ce'}
    assert same_alignment_data(old,moved)
    moved['inputs']['t2']='native-chat'
    assert not same_alignment_data(old,moved)


def test_text_ce_baseline_is_not_reused_across_mask_or_held_window_changes():
    from natlang_neuralese.train.text_warmup import same_alignment_data
    options={'records':'r','pieces':'p','text_data':'t','mask_system_prompt':True,
             'held_documents':16,'tokens':16384,'prefix_tokens':32,'qualification_cohort':'native',
             'rollout_passes':4,'rollout_start_passes':4,'max_sequence_passes':3}
    identity={'options':dict(options),'inputs':{'r':'R','p':'P','t':'T'},
              'target':'gold-token','text_history':'gold-history',
              'supervision_policy':'token-ce'}
    for changed in ({'mask_system_prompt':False}, {'held_documents':8}, {'tokens':8192},
                    {'qualification_cohort':'tools'}, {'text_history':'greedy-history'}):
        other={'options':dict(options),'inputs':dict(identity['inputs']),
               'target':identity['target'],'text_history':identity['text_history'],
               'supervision_policy':identity['supervision_policy']}
        if 'text_history' in changed:
            other.update(changed)
        else:
            other['options'].update(changed)
        assert not same_alignment_data(identity,other)


def test_legacy_identity_without_baseline_policy_remeasures_only_baseline():
    from natlang_neuralese.train.text_warmup import same_alignment_data
    legacy={'options':{'records':'r','pieces':'p','text_data':'t'},
            'inputs':{'r':'R','p':'P','t':'T'}}
    current={'options':{'records':'r','pieces':'p','text_data':'t','mask_system_prompt':True,
                        'held_documents':16,'tokens':16384,'prefix_tokens':32,
                        'rollout_passes':0,'rollout_start_passes':4,'max_sequence_passes':3},
             'inputs':{'r':'R','p':'P','t':'T'},'target':'gold-token',
             'text_history':'gold-history','supervision_policy':'token-ce'}
    assert not same_alignment_data(legacy,current)


def test_suffix_is_serving_generation_prompt_boundary_for_tool_target():
    tokenizer=Tokenizer()
    messages=[{'role':'user','content':'quoted <|neuralese|>'}]
    target={'role':'assistant','content':'','tool_calls':[{'type':'function','function':{'name':'eval','arguments':'{"code":"2+2"}'}}]}
    text,ids,start=native_gold_packet(tokenizer,messages,target,[])
    assert text.startswith(tokenizer.apply_chat_template(messages,tools=None,tokenize=False,
                                                         add_generation_prompt=True))
    assert start>0 and start<len(ids)
    assert start == len(serving_prompt_ids(tokenizer,messages,[]))


def test_target_mask_starts_after_serving_prompt_with_history_and_quoted_role_markers():
    tokenizer=Tokenizer()
    messages=[{'role':'user','content':'Quoted <role>assistant\\n is ordinary.'},
              {'role':'assistant','content':'historical reasoning'},
              {'role':'tool','content':'historical tool result'}]
    target={'role':'assistant','content':'answer quotes <role>tool\\n literally'}
    text,ids,start=native_gold_packet(tokenizer,messages,target,[])
    prompt=tokenizer.apply_chat_template(messages,tools=None,tokenize=False,add_generation_prompt=True)
    assert text.startswith(prompt)
    assert start==len(serving_prompt_ids(tokenizer,messages,[]))
    assert 'historical tool result' in text[:len(prompt)]
    assert 'answer quotes <role>tool\\n literally' in text[len(prompt):]
    # Escaped content spellings are tokenized as ordinary characters; only the
    # generated prompt/reply structure becomes the fixture's role token ID.
    assert ids.count(10000)==len(messages)+1
    quoted=tokenizer('<role>tool\\n',add_special_tokens=False,split_special_tokens=True)['input_ids']
    suffix=ids[start:]
    assert any(suffix[i:i+len(quoted)]==quoted for i in range(len(suffix)-len(quoted)+1))
    assert suffix.count(10000)==0 and suffix.count(10001)==1


def test_target_bound_packet_survives_history_reasoning_template_rewrite():
    class RewritingHistoryTokenizer(Tokenizer):
        def apply_chat_template(self, messages, *, tools=None, tokenize=False, add_generation_prompt=False):
            copied=[dict(message) for message in messages]
            if not add_generation_prompt and len(copied)>1:
                for message in copied[:-1]:
                    if message.get('role')=='assistant':
                        message['content']='template-rewritten historical reasoning'
            return super().apply_chat_template(copied,tools=tools,tokenize=tokenize,
                                               add_generation_prompt=add_generation_prompt)

    tokenizer=RewritingHistoryTokenizer()
    messages=[{'role':'user','content':'Question'},
              {'role':'assistant','content':'original reasoning'},
              {'role':'tool','content':'result'}]
    target={'role':'assistant','content':'final answer'}
    full_text,_=native_gold_document(tokenizer,messages,target,[])
    packet_text,packet_ids,boundary=native_gold_packet(tokenizer,messages,target,[])
    prompt=tokenizer.apply_chat_template(messages,tools=None,tokenize=False,add_generation_prompt=True)
    assert 'template-rewritten historical reasoning' in full_text
    assert 'original reasoning' in packet_text[:len(prompt)]
    assert 'template-rewritten historical reasoning' not in packet_text
    assert boundary==len(serving_prompt_ids(tokenizer,messages,[]))
    assert boundary<len(packet_ids)

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
    writer['source_groups']=reader['source_groups']=['shared-world']
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


def test_writer_context_hydrates_encoded_tool_argument_when_group_and_split_match():
    writer=_soft_state_writer('writer','nz1_state','Encoded context.')
    writer['source_groups']=['world-a']
    reader=record('reader','train')
    reader['source_groups']=['world-a']
    reader['source_ref']={'source_row_sha256':'different-raw-row'}
    encoded=json.dumps({'code':'const note = ', 'priorNotes':{'type':'neuralese','id':'nz1_state'}})
    reader['messages']=[{'role':'assistant','tool_calls':[{'type':'function','function':{
        'name':'eval','arguments':encoded}}]}]
    rows,_,omissions,provenance=gold_text_rows(
        [writer,reader,record('other-train','train'),record('held','test')],{},tokenizer=Tokenizer())
    assert not omissions
    reader_row=next(row for row in rows if row['id']=='reader')
    assert 'Encoded context.' in reader_row['text']
    proof=next(row for row in provenance if row['id']=='reader')['neuralese_context_attestations']
    assert proof[0]['writer_source_groups']==['world-a']


@pytest.mark.parametrize('reader_split,reader_groups,reader_source_row',[
    ('test',['world-a'],'different-raw-row'),
    ('train',['world-b'],'different-raw-row'),
])
def test_writer_context_never_crosses_split_or_source_group(reader_split,reader_groups,reader_source_row):
    writer=_soft_state_writer('writer','nz1_state','Must stay scoped.')
    writer['source_groups']=['world-a']
    reader=record('reader',reader_split)
    reader['source_groups']=reader_groups
    reader['source_ref']={'source_row_sha256':reader_source_row}
    reader['messages']=[{'role':'user','content':[{'type':'neuralese','id':'nz1_state'}]}]
    inputs=[writer,reader,record('train-extra','train'),record('test-extra','test')]
    rows,_,omissions,_=gold_text_rows(inputs,{},tokenizer=Tokenizer())
    assert any(item['id']=='reader' and item['reason']=='unresolved_or_malformed_crisp_reference'
               for item in omissions)
    assert not any(row['id']=='reader' and 'Must stay scoped.' in row['text'] for row in rows)


def test_multiple_eligible_writers_for_same_block_are_ambiguous():
    first=_soft_state_writer('writer-a','nz1_state','First exact body.')
    second=_soft_state_writer('writer-b','nz1_state','Second exact body.')
    first['source_groups']=second['source_groups']=['shared-world']
    reader=record('reader','train')
    reader['source_groups']=['shared-world']
    reader['source_ref']={'source_row_sha256':'reader-row'}
    reader['messages']=[{'role':'user','content':[{'type':'neuralese','id':'nz1_state'}]}]
    rows,_,omissions,_=gold_text_rows(
        [first,second,reader,record('other-train','train'),record('held','test')],{},tokenizer=Tokenizer())
    assert any(item['id']=='reader' for item in omissions)
    assert not any(row['id']=='reader' for row in rows)


def test_unapproved_or_failed_writer_cannot_hydrate_context():
    writer=_soft_state_writer('writer','nz1_state','Must not leak.',approved=False)
    reader=record('reader','train')
    reader['messages']=[{'role':'user','content':[{'type':'neuralese','id':'nz1_state'}]}]
    rows,receipt,omissions,_=gold_text_rows(
        [writer,reader,record('valid-train','train'),record('held','test')],{},tokenizer=Tokenizer())
    assert [x['id'] for x in omissions]==['writer','reader']
    assert 'Must not leak.' not in ''.join(row['text'] for row in rows)
