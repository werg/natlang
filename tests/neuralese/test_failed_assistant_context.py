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
