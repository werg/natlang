"""Whole-graph policy admission permits ordinary episodes and preserves writers."""
import copy
import pytest
from natlang_neuralese.rl.episode_graph import validate_model_turn_coverage,validate_rollout_graph,rollout_turn_names


def attach_execution_graph(episodes):
    """Test traces derived from actual tiny-actor captures, not training data."""
    for episode in episodes:
        turns=episode['turns'];names=rollout_turn_names(turns)
        events=[{'node':name,'kind':'model_turn','call_id':turn.get('invocation_id')}
                for name,turn in zip(names,turns)]
        writers={}
        for name,turn in zip(names,turns):
            for index,write in enumerate((turn.get('sampling') or {}).get('writes',[])):
                block=write['state_id'];node=name+f':write:{index}'
                if block in writers:raise ValueError('test fixture needs explicit repeated-writer occurrence evidence')
                writers[block]=node
                events.append({'node':node,'kind':'block_write','turn':name,'block':block})
        for name,turn in zip(names,turns):
            for index,part in enumerate((turn.get('sampling') or {}).get('prompt',[])):
                if part.get('kind')!='native_state':continue
                block=part['id'];events.append({'node':name+f':read:{index}','kind':'block_read',
                    'turn':name,'block':block,'inputs':[{'node':writers.get(block,'external'),'block':block}]})
        episode['execution_graph']=events
    return episodes


def test_ordinary_complete_execution_needs_no_neuralese_or_minimum_nodes():
    assert validate_model_turn_coverage(['one'],[])=={}
    events=[{'node':'one','kind':'model_turn'}]
    assert set(validate_model_turn_coverage(['one'],events))=={'one'}
    validate_rollout_graph({'turns':[{'invocation_id':'one','sampling':{'prompt':[], 'writes':[]}}]})


@pytest.mark.parametrize('identity',[None,'',17,{'invented':'call'}])
def test_missing_invocation_identity_is_not_replaced_with_a_fabricated_namespace(identity):
    with pytest.raises(ValueError,match='actual nonempty invocation identity'):
        rollout_turn_names([{'invocation_id':identity}])


def graph():
    return [{'node':'producer','kind':'model_turn'},{'node':'consumer','kind':'model_turn'},
        {'node':'write','kind':'block_write','turn':'producer','block':'state'},
        {'node':'read','kind':'block_read','turn':'consumer','block':'state',
         'inputs':[{'node':'write','block':'state'}]}]


def test_missing_captured_producer_cannot_be_reclassified_as_external():
    with pytest.raises(ValueError,match='model-turn inventory'):
        validate_model_turn_coverage(['consumer'],graph())
    assert len(validate_model_turn_coverage(['producer','consumer'],graph()))==4


def test_even_unused_ordinary_or_forced_turns_are_part_of_the_episode():
    events=graph()+[{'node':'ordinary','kind':'model_turn'}]
    with pytest.raises(ValueError,match='model-turn inventory'):
        validate_model_turn_coverage(['producer','consumer'],events)


def test_real_external_inputs_remain_valid_without_a_local_producer():
    events=graph()[1:2]+graph()[3:]
    events[1]['inputs'][0]['node']='older-global-writer'
    assert len(validate_model_turn_coverage(['consumer'],events))==2


@pytest.mark.parametrize('change',['writer-block','writer-turn','duplicate','two-sources'])
def test_bad_internal_occurrences_reject(change):
    events=copy.deepcopy(graph())
    if change=='writer-block':events[2]['block']='different-state'
    if change=='writer-turn':events[2]['turn']='missing'
    if change=='duplicate':events.append(events[0])
    if change=='two-sources':events[3]['inputs'].append({'node':'external','block':'state'})
    with pytest.raises(ValueError):validate_model_turn_coverage(['producer','consumer'],events)


def test_neuralese_sampling_requires_the_actual_graph_not_just_detached_cache_bytes():
    with pytest.raises(ValueError,match='actual complete execution graph'):
        validate_rollout_graph({'turns':[{'invocation_id':'consumer','sampling':{
            'prompt':[{'kind':'native_state','id':'state'}]}}]})


@pytest.mark.parametrize('change',['missing-write','changed-read','duplicate-write','duplicate-read'])
def test_graph_events_must_match_actual_state_capture_inventories(change):
    turns=[{'invocation_id':'producer','sampling':{'writes':[{'state_id':'state'}]}},
           {'invocation_id':'consumer','sampling':{'prompt':[{'kind':'native_state','id':'state'}]}}]
    episode=attach_execution_graph([{'turns':turns}])[0]
    if change=='missing-write':turns[0]['sampling']['writes']=[]
    if change=='changed-read':turns[1]['sampling']['prompt'][0]['id']='decoded-different-state'
    if change=='duplicate-write':turns[0]['sampling']['writes']*=2
    if change=='duplicate-read':turns[1]['sampling']['prompt']*=2
    with pytest.raises(ValueError,match='whole-episode'):
        validate_rollout_graph(episode)


def test_explicit_external_same_content_is_not_aliased_to_unrelated_local_writer():
    from test_native_rl_replay import actor_rollout
    from natlang_neuralese.rl.episode_graph import whole_episode_turns
    from natlang_neuralese.rl.native_replay import compile_replay
    _,roles,identity,_,block,actual=actor_rollout('lfm')
    turns=[{'invocation_id':turn['graph_turn'],'messages':[],'target':{},'sampling':turn['sampling']}
           for turn in actual]
    episode=attach_execution_graph([{'turns':turns}])[0]
    for event in episode['execution_graph']:
        if event['kind']=='block_read':event['inputs'][0]['node']='older-outside-episode-writer'
    replay=compile_replay(whole_episode_turns(episode),identity=identity,roles=roles,fetch_block=lambda _:block)
    assert len(replay.plan.invocations)==2
    consumer=next(node for node in replay.plan.invocations if node.name.startswith('consumer'))
    assert consumer.imports[0].source in replay.external_blocks
