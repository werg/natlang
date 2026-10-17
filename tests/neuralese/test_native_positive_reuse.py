import copy
import pytest
from test_native_rl_replay import actor_rollout
from test_rl_episode_graph import attach_execution_graph
from natlang_neuralese.common.hashing import canonical_json_sha256_hex as digest
from natlang_neuralese.rl.native_replay import prepare_objective
from natlang_neuralese.rl.native_positive_reuse import prepare_positive_reuse


def population(family='lfm'):
    _, roles, identity, _, block, turns = actor_rollout(family)
    raw = [{'messages': [], 'target': {}, 'invocation_id': turn['graph_turn'],
            'sampling': turn['sampling'], 'replay': {'raw_calls_retained': True}} for turn in turns]
    rows = [{'task_id':'train-task','sample':i,'split':'train','group':group,
             'reward':{'quality':quality}, 'policy':{'checkpoint_sha256':identity['checkpoint_sha256']},
             'turns':copy.deepcopy(raw)}
            for i,(group,quality) in enumerate([('flat',1.0),('flat',1.0),('mixed',1.0),('mixed',0.0)])]
    attach_execution_graph(rows)
    options=dict(identity=identity,roles=roles,fetch_block=lambda _:block)
    _, prior=prepare_objective(rows,**options)
    return rows,options,prior


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_real_whole_episode_reuse_preserves_graphs_and_occurrences(family):
    rows,options,prior=population(family)
    selected=[digest(row)for row in rows[:2]]
    before=copy.deepcopy(rows)
    graphs,proof=prepare_positive_reuse(rows,selected_episode_ids=selected,prior_objective=prior,**options)
    expected,_=prepare_objective(rows[:2],mode='rft',**options)
    assert graphs==expected and len(graphs)==2
    assert all(len(graph['nodes'])==2 and not graph['external_states']for graph in graphs)
    assert proof['sampled_actions']==14 and proof['current_policy_claim']is False
    assert proof['training_exposure']is False and rows==before


@pytest.mark.parametrize('problem',['used','missing_prior','wrong_actor','duplicate','unscored','held'])
def test_reuse_rejects_missing_coverage_wrong_split_and_occurrence_ambiguity(problem):
    rows,options,prior=population()
    selected=[digest(row)for row in rows[:2]]
    if problem=='used':selected=[digest(rows[2])]
    elif problem=='missing_prior':prior['rows']={}
    elif problem=='wrong_actor':prior['policy']={**prior['policy'],'checkpoint_sha256':'9'*64}
    elif problem=='duplicate':selected*=2
    elif problem=='unscored':rows[1]['reward']=None;selected=[digest(rows[0])]
    elif problem=='held':rows[0]['split']='test';selected=[digest(rows[0])]
    with pytest.raises(ValueError):
        prepare_positive_reuse(rows,selected_episode_ids=selected,prior_objective=prior,**options)


def test_old_external_state_is_not_frozen_into_new_learner_source():
    rows,options,prior=population()
    for row in rows[:2]:
        row['turns']=row['turns'][:1]  # actual captured consumer; producer belongs to old external actor
    attach_execution_graph(rows)
    selected=[digest(row)for row in rows[:2]]
    with pytest.raises(ValueError,match='closed whole graphs'):
        prepare_positive_reuse(rows,selected_episode_ids=selected,prior_objective=prior,**options)


def test_corrupt_prior_graph_proof_cannot_make_an_episode_unused():
    rows,options,prior=population()
    prior['rows'][next(iter(prior['rows']))]['graph_sha256']='9'*64
    with pytest.raises(ValueError,match='original scored population'):
        prepare_positive_reuse(rows,selected_episode_ids=[digest(rows[0])],prior_objective=prior,**options)
