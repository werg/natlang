"""Whole-graph RL rows for program tuning (plans/HIERARCHICAL_LORA_TUNING.md §5.1a).

A `natlang.tune-rollout/v1` record is one student execution of one training root, scored by the root's oracle:

    {"schema": "natlang.tune-rollout/v1", "root": str, "component": str, "partition": "train",
     "group": str,            # rollouts of one root in one round share a group (group-relative advantages)
     "sample": int, "reward": {"quality": 0 | 1, "source": "root-oracle"},
     "behavior": {"release": str | null, "node_artifacts": {node: nnt1 id},
                  "gates": {json.dumps([binding, family, version]): float | float.hex()},
                  "soft_states": {json.dumps([binding, kind, target]): nzs1 id}, "temperature": float, "seed": int | null},
     "bindings": {invocation_id: tune binding id | null},   # each model-turn invocation's binding (null: runs the base)
     "episode": {production rollout: task_id, split "train", sample, group, reward, policy, turns (recorded native
                 turns with `sampling` captures and raw calls), execution_graph}}

Admission reuses the production recorded-policy path unchanged: `invocation_behavior_policy` binds every actual
per-invocation actor, `save_bundle` archives the exact behavior artifacts and states, and `prepare_objective` emits
the GRPO graphs and the registered `native-policy` objective. The trainer evaluates each episode through
`serve.native_grad.native_graph_gradient` with the CURRENT node coefficients and soft states substituted for the
behavior versions (same bases, recorded lineage), so every model turn of the actual graph, its producer/consumer
dependencies and the clipped importance-weighted policy loss are exactly the production ones.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..common.hashing import canonical_json_sha256_hex, sha256_file_hex_cached
from . import TuneMathError

ROLLOUT_SCHEMA = 'natlang.tune-rollout/v1'


def validate_rollout(record):
    if not isinstance(record, dict) or record.get('schema') != ROLLOUT_SCHEMA:
        raise TuneMathError('RL rows must be natlang.tune-rollout/v1 records')
    if record.get('partition') != 'train':
        raise TuneMathError('only training roots produce RL rewards')
    reward = record.get('reward')
    if not isinstance(reward, dict) or reward.get('quality') not in (0, 1) or reward.get('source') != 'root-oracle':
        raise TuneMathError('an RL reward is the root oracle verdict (quality 0 or 1)')
    behavior = record.get('behavior')
    if not isinstance(behavior, dict) or not isinstance(behavior.get('node_artifacts'), dict):
        raise TuneMathError('an RL rollout records its exact behavior node versions')
    episode = record.get('episode')
    if (not isinstance(episode, dict) or episode.get('split') != 'train' or episode.get('reward') != {'quality': reward['quality']}
            or episode.get('group') != record.get('group') or not isinstance(record.get('bindings'), dict)):
        raise TuneMathError('an RL rollout carries its scored production episode with its group and actor bindings')
    for turn in episode.get('turns') or []:
        if turn.get('invocation_id') not in record['bindings']:
            raise TuneMathError('every model-turn invocation of an RL episode needs its tune binding')
    return record


def call_of(node_name):
    """Graph node names are `<invocation_id>#turn<k>` (rl/episode_graph.rollout_turn_names)."""
    call, separator, _ = node_name.rpartition('#turn')
    if not separator:
        raise TuneMathError(f'RL graph node {node_name} is not a recorded model turn')
    return call


class RLEpisodes:
    """Admitted RL episodes for one trainer input: graphs, the registered objective and their rollouts."""

    def __init__(self, engine, rollouts, workdir):
        from ..rl.native_replay import invocation_behavior_policy, prepare_objective
        from ..rl.graph_replay import save_bundle
        from ..model.native_artifact import encode_native_tensor_artifact
        from ..serve.store import encode_native_state
        self.rollouts = [validate_rollout(record) for record in rollouts]
        if not self.rollouts:
            raise TuneMathError('no RL rollouts')
        identity, roles = engine.native_identity, engine.role_tokens
        fetch_block = lambda ident: engine.store.get_native_state(ident).block
        episodes = [record['episode'] for record in self.rollouts]
        self.by_episode = {canonical_json_sha256_hex(record['episode']): record for record in self.rollouts}
        if len(self.by_episode) != len(self.rollouts):
            raise TuneMathError('RL rollouts repeat an identical episode')
        policy = invocation_behavior_policy(episodes, identity=identity, roles=roles, fetch_block=fetch_block)
        owner = canonical_json_sha256_hex(policy)
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        fetch = lambda ident: (encode_native_tensor_artifact(engine.store.get_native_tensor_artifact(ident).artifact)
                               if ident.startswith('nnt1_') else encode_native_state(fetch_block(ident)))
        save_bundle(episodes, workdir / 'bundle', policy_id=owner, fetch=fetch)
        graphs, spec = prepare_objective(episodes, identity=identity, roles=roles, fetch_block=fetch_block,
                                         block_bundle=workdir / 'bundle' / 'manifest.json', behavior_policy=policy)
        path = workdir / 'objective.json'
        path.write_text(json.dumps(spec, sort_keys=True))
        self.objective = {'kind': 'native-policy', 'path': str(path), 'sha256': sha256_file_hex_cached(path)}
        # GRPO admits only groups whose rewards differ; the rest contribute nothing and are recorded as such.
        self.graphs = graphs
        self.advantages = {graph['id']: spec['rows'][graph['id']]['advantage'] for graph in graphs}
        self.skipped = len(self.rollouts) - len(graphs)

    def rollout(self, graph):
        return self.by_episode[graph['id']]

    def digest(self):
        return canonical_json_sha256_hex([canonical_json_sha256_hex(graph) for graph in self.graphs])
