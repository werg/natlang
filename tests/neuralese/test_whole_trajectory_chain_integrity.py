"""Whole-document source credit follows actual answers, with held context protected."""
import copy

from test_render_held_split_alias import load_renderer


def chain():
    user = {'role': 'user', 'content': 'question'}
    answer = {'role': 'assistant', 'content': 'answer'}
    common = {'split': 'train', 'source_groups': ['source-a'], 'program_id': 'program',
              'teacher_trajectory_id': 'trajectory', 'source_ref': {'invocation_id': 'root'}}
    early = {**common, 'id': 'early', 'messages': [user], 'target': answer}
    late = {**common, 'id': 'late', 'messages': [user, answer, {'role': 'user', 'content': 'next'}],
            'target': {'role': 'assistant', 'content': 'last answer'}}
    return early, late


def test_only_actual_earlier_answers_are_covered_and_other_branches_survive():
    render = load_renderer();early, late = chain()
    branch = {**early, 'id': 'branch', 'target': {'role': 'assistant', 'content': 'different answer'}}
    rows = render.whole_trajectory_records([early, branch, late], {})
    by_id = {row['id']: members for row, members in rows}
    assert by_id == {'branch': ['branch'], 'late': ['early', 'late']}


def test_exact_teacher_reasoning_target_is_a_valid_history_member():
    render = load_renderer();early, late = chain()
    early['teacher_reasoning'] = 'derive the answer'
    late['messages'][1] = {**early['target'], 'reasoning_content': early['teacher_reasoning']}
    assert render.source_decision_in_history(early, late)
    assert render.whole_trajectory_records([early, late], {})[0][1] == ['early', 'late']


def test_mixed_splits_are_separate_even_when_answer_is_exact():
    render = load_renderer();early, late = chain();early['split'] = 'test'
    rows = render.whole_trajectory_records([early, late], {})
    assert {(row['id'], tuple(members)) for row, members in rows} == {
        ('early', ('early',)), ('late', ('late',))}


def test_held_context_is_removed_but_valid_alternate_branch_is_preserved():
    render = load_renderer();early, late = chain()
    early['source_groups'] = ['held-source'];late['source_groups'] = ['other-source']
    alternative = copy.deepcopy(late);alternative['id'] = 'alternative'
    alternative['messages'][1]['content'] = 'different answer'
    kept, omitted = render.decontaminate_source_records([early, late, alternative], {'held-source'})
    assert [row['id'] for row in kept] == ['alternative']
    by_id = {item['id']: item for item in omitted}
    assert by_id['early']['reason'] == 'train_decision_shares_held_source_group'
    assert by_id['late']['reason'] == 'train_trajectory_contains_held_source_decision'
    assert by_id['late']['held_source_record_id'] == 'early'


def test_held_split_context_never_becomes_train_context_under_a_new_group():
    render = load_renderer();early, late = chain();early['split'] = 'test'
    kept, omitted = render.decontaminate_source_records([early, late], set())
    assert [row['id'] for row in kept] == ['early']
    assert omitted[0]['id'] == 'late'


def test_split_local_omission_does_not_truncate_a_held_chain():
    render = load_renderer();early, late = chain()
    held = copy.deepcopy(late);held.update(id='held', split='test')
    blocked = {render.split_chain_key(early): 0}
    assert [(row['id'], members) for row, members in render.whole_trajectory_records([early, late, held], blocked)] == [
        ('held', ['held'])]
    # Legacy selection states retain their previous conservative interpretation.
    assert render.whole_trajectory_records([early, late, held], {render.chain_key(early): 0}) == []


def test_call_id_transport_is_bound_but_never_hides_a_changed_action():
    render = load_renderer();early, late = chain()
    early['target'] = {'role': 'assistant', 'content': '', 'tool_calls': [
        {'id': 'teacher-1', 'type': 'function', 'function': {'name': 'eval', 'arguments': '{"code":"1+1"}'}}]}
    early['teacher_reasoning'] = 'add the numbers'
    late['messages'][1] = copy.deepcopy(early['target'])
    late['messages'][1]['tool_calls'][0]['id'] = 'runtime-9'
    late['messages'][1]['reasoning_content'] = early['teacher_reasoning']
    binding = render.source_assistant_binding(early, late['messages'][1])
    assert binding['tool_call_id_rebindings'] == [{'call_index': 0, 'source_id': 'teacher-1', 'history_id': 'runtime-9'}]
    assert render.source_decision_in_history(early, late)
    late['messages'][1]['tool_calls'][0]['function']['arguments'] = '{"code":"3+3"}'
    assert not render.source_decision_in_history(early, late)


def test_quality_masked_chain_preserves_mixed_source_groups_and_exact_union():
    from test_whole_trajectory_target_holds import held_source, convert
    render = load_renderer();early, late = chain()
    early['source_groups'] = ['earlier-group'];late['source_groups'] = ['later-group']
    metadata = render.whole_trajectory_target_holds(late, ['early', 'late'], late['messages'],
        {'early': early, 'late': late}, {'early': '1'*64, 'late': '2'*64}, {'early'})
    assert metadata['source_record_groups'] == {'early': ['earlier-group'], 'late': ['later-group']}
    tok, source = held_source()
    source['source_groups'] = ['earlier-group', 'later-group']
    source['source_record_groups'] = {'held': ['earlier-group'], 'good': ['later-group']}
    assert convert(tok, source)['source_groups'] == ['earlier-group', 'later-group']


def test_untrained_stock_reasoning_authenticates_history_but_is_explicitly_masked():
    render = load_renderer();early, late = chain()
    early.update(teacher_reasoning='stock action notes', teacher_reasoning_trained=False)
    late['messages'][1] = {**early['target'], 'reasoning_content': early['teacher_reasoning']}
    assert render.source_decision_in_history(early, late)
    metadata = render.whole_trajectory_reasoning_holds(late, ['early', 'late'], late['messages'], late['target'],
        {'early': early, 'late': late}, {'early': '1'*64, 'late': '2'*64})
    assert metadata['context_only_reasoning_turns'] == [0]
    assert metadata['reasoning_exclusion_bindings'][0]['teacher_reasoning_trained'] is False
    late['messages'][1]['reasoning_content'] = 'unrelated notes'
    assert not render.source_decision_in_history(early, late)


def reasoning_source():
    from test_whole_trajectory_target_holds import held_source
    tok, source = held_source()
    for key in ('assistant_turn_count', 'context_only_assistant_turns', 'target_exclusion_bindings',
                'target_source_bindings', 'target_source_record_ids', 'context_only_source_record_ids', 'source_record_groups'):
        source.pop(key)
    source.update(source_record_ids=['held', 'good'], reasoning_assistant_turn_count=2,
        context_only_reasoning_turns=[0], reasoning_exclusion_bindings=[{
            'id': 'held', 'source_record_sha256': '1'*64, 'assistant_turn': 0,
            'teacher_reasoning_trained': False, 'source_target_sha256': '3'*64,
            'history_assistant_sha256': '4'*64, 'untrained_reasoning_sha256': '5'*64,
            'tool_call_id_rebindings': []}])
    return tok, source


def test_converter_zeroes_only_untrained_reasoning_body_and_retains_action_and_boundaries():
    from test_whole_trajectory_target_holds import convert
    from natlang_neuralese.train.text_supervision import chat_roles
    tok, source = reasoning_source();graph = convert(tok, source);node = graph['nodes'][0]
    assert node['token_ids'] == source['token_ids']
    labels = chat_roles(node['token_ids'], start_id=tok.convert_tokens_to_ids('<|im_start|>'),
        role_ids={tok.convert_tokens_to_ids(role): role for role in ('system','user','assistant','tool')},
        think_open=tok.convert_tokens_to_ids('<think>'), think_close=tok.convert_tokens_to_ids('</think>'))
    boundaries = {tok.convert_tokens_to_ids('<think>'), tok.convert_tokens_to_ids('</think>')}
    # ASCII fixture has one token per ordinary character; use role-run ordinal
    # to test the actual parsed boundaries rather than character offsets.
    ordinal=-1;inside=False;masked=0;positive_action=0
    for index,code in enumerate(labels):
        if code in (4,5) and not inside:ordinal+=1
        inside=code in (4,5)
        if ordinal==0 and code==4 and node['token_ids'][index] not in boundaries:
            assert node['target_weights'][index]==0;masked+=1
        elif ordinal==0 and code in (4,5):
            assert node['target_weights'][index]>0;positive_action+=1
        elif ordinal==1 and code in (4,5):assert node['target_weights'][index]>0
    assert masked>0 and positive_action>0
    assert graph['conversion']['supervision_policy']['untrained_reasoning_tokens_masked']==masked
    tok.encode = lambda text, add_special_tokens=False: tok(text, add_special_tokens=add_special_tokens)['input_ids']
    render=load_renderer();count=render.role_counter(tok)
    assert render.whole_trajectory_target_tokens(source,count)[0] == sum(code in (4,5) for code in labels)-masked


def test_reasoning_hold_metadata_is_required_to_close_and_fails_per_row():
    import pytest
    from test_whole_trajectory_target_holds import convert
    tok, source = reasoning_source();source['reasoning_exclusion_bindings'][0]['teacher_reasoning_trained'] = True
    with pytest.raises(ValueError,match='binding'):convert(tok,source)
    tok, source = reasoning_source();source['reasoning_exclusion_bindings'][0]['source_record_sha256'] = '9'*64
    with pytest.raises(ValueError,match='binding'):convert(tok,source)


def test_whole_ids_bind_source_groups_masks_and_preserve_the_original_id_crosswalk():
    from test_whole_trajectory_target_holds import convert
    render=load_renderer();tok,source=reasoning_source()
    first=render.qualified_whole_trajectory_row(source)
    assert first['source_row_id']==source['id'] and first['id'].startswith(source['id']+':source-sha256:')
    converted = convert(tok, first)['conversion']
    assert converted['source_row_id'] == first['id']
    assert converted['legacy_source_row_id'] == source['id']
    other=copy.deepcopy(source);other['source_groups']=['different-group']
    assert render.qualified_whole_trajectory_row(other)['id']!=first['id']
    other=copy.deepcopy(source);other['context_only_reasoning_turns']=[1]
    assert render.qualified_whole_trajectory_row(other)['id']!=first['id']
    # Templates differ; source identity and resulting twin IDs do not.
    twin={**source,'text':'another tokenizer rendering','token_ids':[123], 'tokenizer_sha256':'f'*64}
    assert render.qualified_whole_trajectory_row(twin)['id']==first['id']
