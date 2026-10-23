"""Verify declared unused positive episodes using the existing whole-graph compiler.

This prepares an ordinary positive-token source, never a policy objective or a
claim that a later learner sampled these episodes. Artifact/split protection and
source-continuation admission remain the caller's independently bound checks.
"""
from __future__ import annotations


def prepare_positive_reuse(rollouts, *, selected_episode_ids, prior_objective,
                           identity, roles, fetch_block, block_bundle=None):
    from ..common.hashing import canonical_json_sha256_hex as digest
    from .native_replay import prepare_objective
    from .rollout_math import group_advantages

    selected = list(selected_episode_ids)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError('positive reuse requires distinct declared original episode identities')
    if (prior_objective.get('kind') != 'native-policy'
            or prior_objective.get('mode') != 'grpo'
            or prior_objective.get('policy') != identity):
        raise ValueError('positive reuse prior GRPO differs from the actual sampled actor')
    episodes = {digest(row): row for row in rollouts}
    if len(episodes) != len(rollouts) or any(key not in episodes for key in selected):
        raise ValueError('positive reuse lost or duplicated an original sampled occurrence')
    # Recompile the original scored population, rather than trusting an empty or
    # truncated prior objective as evidence that an episode has not been used.
    _, replayed = prepare_objective(rollouts, identity=identity, roles=roles,
                                   fetch_block=fetch_block, block_bundle=block_bundle)
    prior = prior_objective.get('rows', {})
    if set(prior) != set(replayed['rows']) or any(
            prior[key].get('graph_sha256') != row['graph_sha256']
            for key, row in replayed['rows'].items()):
        raise ValueError('positive reuse prior GRPO does not cover the original scored population')
    groups = {}
    for row in rollouts:
        groups.setdefault((row.get('task_id'), row.get('group')), []).append(row)
    for key in selected:
        row = episodes[key]
        members = groups[(row.get('task_id'), row.get('group'))]
        qualities = [(member.get('reward') or {}).get('quality') for member in members]
        if (key in prior or row.get('split') != 'train' or not qualities
                or any(type(value) not in (int, float) or value != 1 for value in qualities)
                or group_advantages(qualities)[1]['spread']):
            raise ValueError('positive reuse requires an unused fully scored correct flat TRAIN group')
    graphs, objective = prepare_objective([episodes[key] for key in selected],
        identity=identity, roles=roles, fetch_block=fetch_block,
        block_bundle=block_bundle, mode='rft')
    if [graph['id'] for graph in graphs] != selected:
        raise ValueError('positive reuse compiler changed the declared occurrence order')
    # Archived external KV belongs to the old actor. Only genuine producer
    # invocations present in these whole graphs may be recomputed by a new one.
    if any(graph.get('external_states') for graph in graphs):
        raise ValueError('positive reuse requires closed whole graphs, not old external actor state')
    return graphs, {'schema': 'natlang.native-positive-episode-reuse/1',
        'objective': 'ordinary-positive-token-sft', 'sampled_actor': dict(identity),
        'selected_episode_ids': selected, 'prior_grpo_sha256': digest(prior_objective),
        'source_rollouts_sha256': digest(rollouts),
        'compiled_rft_sha256': digest(objective),
        'graphs_sha256': digest(graphs), 'episodes': len(graphs),
        'sampled_actions': sum(len(row['actions']) for row in objective['rows'].values()),
        'current_policy_claim': False, 'training_exposure': False,
        'requires_current_protection_and_source_continuation': True}
