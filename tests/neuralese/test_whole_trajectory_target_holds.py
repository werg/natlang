"""Explicit quality target holds retain complete recovery context."""
import copy
import json

import pytest

from natlang_neuralese.data.native_graph_conversion import whole_trajectory_text_row_graph
from test_native_graph_conversion import ChatRoleAsciiTokenizer
from test_render_held_split_alias import load_renderer


def held_source():
    render = load_renderer()
    user = {'role': 'user', 'content': 'solve'}
    bad = {'role': 'assistant', 'content': 'bad action'}
    feedback = {'role': 'user', 'content': 'try again'}
    common = {'split': 'train', 'source_groups': ['case']}
    earlier = {**common, 'id': 'held', 'messages': [user], 'target': bad}
    final = {**common, 'id': 'good', 'messages': [user, bad, feedback],
             'target': {'role': 'assistant', 'content': 'good action'}}
    records = {'held': earlier, 'good': final}
    hashes = {'held': '1'*64, 'good': '2'*64}
    metadata = render.whole_trajectory_target_holds(final, list(records), final['messages'], records, hashes, {'held'})
    tok = ChatRoleAsciiTokenizer()
    text = ('<|im_start|>user<|im_sep|>solve<|im_end|>'
            '<|im_start|>assistant<|im_sep|><think>bad reasoning</think>bad action<|im_end|>'
            '<|im_start|>user<|im_sep|>try again<|im_end|>'
            '<|im_start|>assistant<|im_sep|><think>good reasoning</think>good action<|im_end|>')
    source = {'id': 'whole-trajectory:good', 'text': text, 'token_ids': tok(text)['input_ids'],
              'tokenizer_sha256': 'a'*64, 'split': 'train', 'source_groups': ['case'],
              'supervision': 'whole-trajectory', 'source_record_ids': metadata['target_source_record_ids'],
              'source_records': [{'id': key, 'sha256': value} for key, value in hashes.items()], **metadata}
    return tok, source


def convert(tok, source):
    return whole_trajectory_text_row_graph(source, source_tokenizer_fingerprint='a'*64,
        tokenizer=tok, expanded_tokenizer_fingerprint='b'*64)


def test_held_reasoning_and_action_have_zero_loss_with_exact_context_and_later_targets():
    tok, source = held_source()
    graph = convert(tok, source)
    node = graph['nodes'][0]
    assert node['token_ids'] == source['token_ids']
    assert tok.decode(node['token_ids']) == source['text']
    # The same role parser defines both conversion and these structural spans.
    from natlang_neuralese.train.text_supervision import chat_roles, ROLE_CODES
    labels = chat_roles(node['token_ids'], start_id=tok.convert_tokens_to_ids('<|im_start|>'),
        role_ids={tok.convert_tokens_to_ids(role): role for role in ('system', 'user', 'assistant', 'tool')},
        think_open=tok.convert_tokens_to_ids('<think>'), think_close=tok.convert_tokens_to_ids('</think>'))
    assistant = {ROLE_CODES.index('assistant_reasoning'), ROLE_CODES.index('assistant_reply')}
    starts = [i for i, role in enumerate(labels) if role in assistant and (i == 0 or labels[i-1] not in assistant)]
    assert len(starts) == 2
    held = [i for i in range(starts[0], starts[1]) if labels[i] in assistant]
    later = [i for i in range(starts[1], len(labels)) if labels[i] in assistant]
    assert held and later
    assert all(node['target_weights'][i] == 0 for i in held)
    assert all(node['target_weights'][i] > 0 for i in later)
    provenance = graph['conversion']
    assert provenance['source_record_ids'] == ['good']
    assert provenance['context_only_source_record_ids'] == ['held']
    assert provenance['target_exclusion_bindings'][0]['source_record_sha256'] == '1'*64
    assert provenance['supervision_policy']['context_only_assistant_tokens_masked'] == len(held)


@pytest.mark.parametrize('change', ['count', 'ordinal', 'digest', 'target_overlap', 'partial', 'all_held'])
def test_invalid_target_hold_binding_fails_closed(change):
    tok, source = held_source()
    if change == 'count': source['assistant_turn_count'] = 3
    elif change == 'ordinal': source['context_only_assistant_turns'] = [2]
    elif change == 'digest': source['target_exclusion_bindings'][0]['source_record_sha256'] = '3'*64
    elif change == 'target_overlap': source['target_source_record_ids'].append('held')
    elif change == 'partial': source.pop('context_only_assistant_turns')
    else:
        source['context_only_assistant_turns'] = [0, 1]
        source['target_exclusion_bindings'].append({'id': 'good', 'source_record_sha256': '2'*64, 'assistant_turn': 1})
        source['target_source_record_ids'] = []
        source['context_only_source_record_ids'] = ['held', 'good']
    with pytest.raises(ValueError): convert(tok, source)


def test_renderer_dedup_exclusion_is_distinct_from_target_holds():
    render = load_renderer()
    earlier = {'id': 'old', 'messages': [{'role': 'user', 'content': 'question'}],
               'target': {'role': 'assistant', 'content': 'answer'}}
    final = {'id': 'new', 'messages': earlier['messages']+[earlier['target']],
             'target': {'role': 'assistant', 'content': 'next'}}
    records = {'old': earlier, 'new': final};hashes = {key: '1'*64 for key in records}
    assert render.whole_trajectory_target_holds(final, list(records), final['messages'], records, hashes, set()) == {}
    with pytest.raises(ValueError, match='no independently admitted'):
        render.whole_trajectory_target_holds(final, list(records), final['messages'], records, hashes, set(records))
    with pytest.raises(ValueError, match='assistant count'):
        render.whole_trajectory_target_holds(final, list(records), [], records, hashes, {'old'})


@pytest.mark.parametrize('change', ['wrong_target', 'duplicate_ordinal', 'split', 'groups'])
def test_member_authentication_rejects_false_targets_and_cross_source_credit(change):
    render = load_renderer()
    prefix = [{'role': 'user', 'content': 'question'}]
    bad = {'role': 'assistant', 'content': 'wrong answer'}
    common = {'split': 'train', 'source_groups': ['case']}
    earlier = {**common, 'id': 'held', 'messages': prefix, 'target': bad}
    final = {**common, 'id': 'good', 'messages': prefix+[bad, {'role': 'user', 'content': 'retry'}],
             'target': {'role': 'assistant', 'content': 'right answer'}}
    records = {'held': earlier, 'good': final}
    if change == 'wrong_target': earlier['target'] = {'role': 'assistant', 'content': 'unrelated'}
    elif change == 'duplicate_ordinal': records['alias'] = {**earlier, 'id': 'alias'}
    elif change == 'split': earlier['split'] = 'test'
    else: earlier['source_groups'] = ['different-case']
    with pytest.raises(ValueError):
        render.whole_trajectory_target_holds(final, list(records), final['messages'], records,
                                           {key: '1'*64 for key in records}, {'held'})


@pytest.mark.parametrize('change', ['held_ordinal', 'duplicate', 'missing', 'target_digest'])
def test_conversion_requires_positive_target_ordinal_bindings(change):
    tok, source = held_source()
    if change == 'held_ordinal': source['target_source_bindings'][0]['assistant_turn'] = 0
    elif change == 'duplicate': source['target_source_bindings'] *= 2
    elif change == 'missing': source.pop('target_source_bindings')
    else: source['target_source_bindings'][0]['source_target_sha256'] = 'not-a-sha'
    with pytest.raises(ValueError): convert(tok, source)


def test_target_hold_control_is_bound_before_resume(tmp_path):
    render = load_renderer();source = tmp_path/'turns.jsonl';source.write_text('{}\n')
    hold = tmp_path/'holds.jsonl';hold.write_text('{"id":"old"}\n')
    binding = render.bind_target_hold_policy(tmp_path, [source], [hold])
    assert render.bind_target_hold_policy(tmp_path, [source], [hold]) == binding
    hold.write_text('{"id":"new"}\n')
    with pytest.raises(ValueError, match='policy changed'):
        render.bind_target_hold_policy(tmp_path, [source], [hold])
    with pytest.raises(ValueError, match='policy changed'):
        render.bind_target_hold_policy(tmp_path, [source], [])


def test_source_identity_and_merge_statistics_keep_target_mask_semantics():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts/neuralese_data'))
    from text_tokenizer_twin import document_identity
    from build_native_turn_union import _row_identity
    tok, source = held_source()
    other = copy.deepcopy(source);other['context_only_assistant_turns'] = [1]
    assert document_identity(source) != document_identity(other)
    assert _row_identity(source)['target_mask'] != _row_identity(other)['target_mask']
