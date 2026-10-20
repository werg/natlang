"""Shared failed-history mask: exact source view, preserved success and suffix."""
import copy
import pytest
from test_native_graph_conversion import ChatRoleAsciiTokenizer
from natlang_neuralese.data.native_graph_conversion import text_row_graph,unambiguous_chat_input_guard
from natlang_neuralese.train.text_supervision import ROLE_CODES,chat_roles,failed_assistant_mask


def fixture(tool_role='tool'):
    tok=ChatRoleAsciiTokenizer()
    feedback=(('<|im_start|>user\n<tool_response>rejected: wrong argument</tool_response>' if tool_role=='user'
               else '<|im_start|>tool<|im_sep|>rejected: wrong argument')+'<|im_end|>')
    prefix=('<|im_start|>user<|im_sep|>input<|im_end|>'
            '<|im_start|>assistant<|im_sep|>bad action<|im_end|>'+feedback+
            '<|im_start|>assistant<|im_sep|>good action<|im_end|>'
            '<|im_start|>tool<|im_sep|>ok<|im_end|><|im_start|>assistant<|im_sep|>')
    text=prefix+'target<|im_end|>';ids=tok(text)['input_ids'];boundary=len(tok(prefix)['input_ids'])
    source={'id':'new-view','split':'train','source_groups':['train-only'],'tokenizer_sha256':'a'*64,
            'text':text,'token_ids':ids,'supervised_suffix_start':boundary}
    options=dict(tokenizer=tok,source_tokenizer_fingerprint='a'*64,expanded_tokenizer_fingerprint='b'*64,
        assistant_history_guard=unambiguous_chat_input_guard([],{'content':'target'},[]))
    return source,options


@pytest.mark.parametrize('tool_role',['tool','user'])
def test_only_explicitly_failed_prior_assistant_is_zero_weight(tool_role):
    source,options=fixture(tool_role);tok=options['tokenizer'];before=copy.deepcopy(source)
    old=text_row_graph(source,**options)
    new=text_row_graph(source,**options,mask_failed_assistant_history=True)
    ids=source['token_ids'];boundary=source['supervised_suffix_start']
    labels=chat_roles(ids,start_id=tok.convert_tokens_to_ids('<|im_start|>'),
        role_ids={tok.convert_tokens_to_ids(role):role for role in ('system','user','assistant','tool')},
        tool_response_prefixes=tuple(tuple(tok(text)['input_ids'])for text in ('\n<tool_response>','\n\n<tool_response>')))
    mask=failed_assistant_mask(ids,labels,lambda values:tok.decode(values,skip_special_tokens=False))
    assert any(mask) and not any(mask[boundary:])
    expected=[0. if index<boundary and mask[index] else weight
        for index,weight in enumerate(old['nodes'][0]['target_weights'])]
    assert new['nodes'][0]['target_weights']==expected
    assert new['nodes'][0]['token_ids']==ids and source==before
    assert new['nodes'][0]['target_weights'][boundary:]==old['nodes'][0]['target_weights'][boundary:]
    assert all(new['nodes'][0]['target_weights'][index]==old['nodes'][0]['target_weights'][index]
               for index,label in enumerate(labels)if label==ROLE_CODES.index('tool'))
    assert any(new['nodes'][0]['target_weights'][index]==.05 for index in range(boundary)
               if labels[index]==ROLE_CODES.index('assistant_reply')and not mask[index])
    proof=new['conversion']['assistant_history_target_mask']
    assert proof['policy']=='runtime-failed-prior-assistant-context-only/1'
    assert proof['masked_tokens']==sum(mask)
    assert text_row_graph(source,**options,mask_failed_assistant_history=False)==old
    assert text_row_graph(source,**options,mask_assistant_history=True)==text_row_graph(
        source,**options,mask_assistant_history=True,mask_failed_assistant_history=True)


def test_failed_mask_requires_exact_guard_and_boolean():
    source,options=fixture();options.pop('assistant_history_guard')
    with pytest.raises(ValueError,match='structured input protection'):
        text_row_graph(source,**options,mask_failed_assistant_history=True)
    with pytest.raises(ValueError,match='explicitly boolean'):
        text_row_graph(source,**options,mask_failed_assistant_history='yes')


def test_source_view_identity_preserves_legacy_and_has_separate_mask_authority():
    from scripts.neuralese_data.build_verified_teacher_trajectory_graphs import teacher_action_graph_id
    old=teacher_action_graph_id('source','a'*64,0)
    new=teacher_action_graph_id('source','a'*64,0,failed_assistant_history_policy='mask-runtime-failures')
    assert new==old+':assistant-history:mask-runtime-failures'
    assert teacher_action_graph_id('source','a'*64,0,'keep')==old+':history-reasoning:keep'
    assert teacher_action_graph_id('source','a'*64,0,'keep','mask-runtime-failures')==old+':history-reasoning:keep:assistant-history:mask-runtime-failures'
    with pytest.raises(ValueError):teacher_action_graph_id('source','a'*64,0,failed_assistant_history_policy='unknown')


def test_strict_source_verifier_reconstructs_failed_mask_and_protects_old_ids(monkeypatch):
    from scripts.coreweave.teacher_tranche_admission import verify_teacher_model_actions
    from scripts.neuralese_data import build_verified_teacher_trajectory_graphs as converter
    from natlang_neuralese.data import text_corpus
    source,options=fixture();tok=options['tokenizer']
    monkeypatch.setattr(text_corpus,'tokenizer_fingerprint',lambda _:'b'*64)
    monkeypatch.setattr(converter,'render_teacher_packet',lambda *a:(((source['text'],source['token_ids'],source['supervised_suffix_start']),None),None))
    raw={'row':{'task':{'program_ir':{'source_groups':['train-only']}},'trajectory':[
        {'assistant':{'content':'target'},'model_response':{'raw_calls':[]},'context':[],'tools_offered':[]}]}}
    captured=raw['row']['trajectory'][0]
    options['assistant_history_guard']=converter.reviewed_teacher_context_guard(captured,[],{'role':'assistant','content':'target','tool_calls':[]},[])
    source['id']=converter.teacher_action_graph_id('row','a'*64,0,failed_assistant_history_policy='mask-runtime-failures')
    graph=text_row_graph(source,**options,mask_failed_assistant_history=True)
    graph['conversion']['teacher_source']={'source_row_id':'row','teacher_turn_index':0,'source_row_sha256':'a'*64,
        'source_artifact_sha256':'d'*64,'source_groups':['train-only'],'decision_admission':{'materialized_decision_id':'row:decision:0000'}}
    receipt={'failed_assistant_history_policy':'mask-runtime-failures','tokenizer':{'source_fingerprint':'a'*64,'expanded_fingerprint':'b'*64},
        'outputs':{'train':{'rows':1}},'turns':1,'deduplication':{'emitted_rows':1,'omitted_rows':0,'omissions':[]},'action_omissions':[]}
    kwargs=dict(raw_rows=raw,raw_row_shas={'row':'a'*64},source_sha256='d'*64,decision_keys={('row',0)},hold_ids=set(),partial_tokenizer=tok)
    assert verify_teacher_model_actions([graph],receipt,**kwargs)['eligible_actions']=={('row','a'*64,0)}
    for damage in ('mask','weight','identity','tokenizer','policy','held-legacy'):
        bad=copy.deepcopy(graph);rec=copy.deepcopy(receipt);args=dict(kwargs)
        if damage=='mask':bad['conversion']['assistant_history_target_mask']['masked_tokens']+=1
        elif damage=='weight':bad['nodes'][0]['target_weights']=[.05 if v==0 else v for v in bad['nodes'][0]['target_weights']]
        elif damage=='identity':bad['id']=bad['id'].split(':assistant-history:')[0]
        elif damage=='tokenizer':args['partial_tokenizer']=None
        elif damage=='policy':rec.pop('failed_assistant_history_policy')
        else:args['hold_ids']={'text:'+converter.teacher_action_graph_id('row','a'*64,0)}
        with pytest.raises(ValueError):verify_teacher_model_actions([bad],rec,**args)
