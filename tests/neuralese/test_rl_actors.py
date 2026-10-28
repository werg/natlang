"""Declared stale/quantized RL actors (rl.actors): admission, importance requirement and objective custody."""
import copy
import json
from types import SimpleNamespace

import pytest
import torch

from natlang_neuralese.rl.actors import accepted_policy, actor_declaration, validate_capture_export

ACTOR = '1' * 64
LEARNER = '9' * 64


def stale_episode():
    """The whole producer/consumer fixture sampled by checkpoint ACTOR (top-k captured)."""
    from test_native_rl_replay import actor_rollout
    from test_shared_state import tiny
    from natlang_neuralese.common.sampling import behavior_top_k_declaration, encode_top_k
    from natlang_neuralese.model.shared_state import SharedStateModel
    torch.manual_seed(907)
    model = SharedStateModel(tiny('lfm'), marker_ids=(58, 59, 60, 61), rank=4)
    roles = SimpleNamespace(produce_open=58, produce_close=59, consume_open=60, consume_close=61)
    actor = {'checkpoint_sha256': ACTOR, 'tokenizer_fingerprint': '2' * 64,
             'embedding_weight_sha256': '3' * 64, 'architecture': model.port.config.model_type}
    seen = []
    original = model.logits
    def recording(hidden):
        value = original(hidden)
        seen.append(value.detach().float().div(.8).log_softmax(-1)[0])
        return value
    model.logits = recording
    _, _, _, _, block, turns = actor_rollout('lfm', loaded=(model, roles, actor))
    del model.logits
    ordered = [sample for name in ('producer', 'consumer') for sample in
               {turn['graph_turn']: turn for turn in turns}[name]['sampling']['samples']]
    for sample, log_probs in zip(ordered, seen):
        sample['top_k'] = encode_top_k(log_probs, 8)
    for turn in turns:
        turn['sampling']['behavior_top_k'] = behavior_top_k_declaration(8)
    raw_turns = [{'messages': [], 'target': {}, 'invocation_id': turn['graph_turn'], 'sampling': turn['sampling'],
                  'replay': {'raw_calls_retained': True}} for turn in turns]
    episodes = [{'task_id': 'train-task', 'sample': sample, 'split': 'train', 'group': 'g',
                 'reward': {'quality': reward}, 'policy': {'checkpoint_sha256': ACTOR},
                 'turns': copy.deepcopy(raw_turns)} for sample, reward in enumerate((1, 0))]
    from test_rl_episode_graph import attach_execution_graph
    attach_execution_graph(episodes)
    learner = {**actor, 'checkpoint_sha256': LEARNER}
    return model, roles, actor, learner, block, episodes


def declaration(export=None):
    return {'schema': 'natlang.rl-actors/1', 'learner_checkpoint_sha256': LEARNER,
            'actors': {ACTOR: {'lineage_step': 1200, 'export': export}}}


def test_declaration_is_strict_and_admits_only_the_learner_with_another_checkpoint():
    learner = {'checkpoint_sha256': LEARNER, 'tokenizer_fingerprint': '2' * 64}
    quantized = declaration({'kind': 'quantized', 'format': 'gguf-q8_0', 'sha256': 'a' * 64})
    assert actor_declaration(quantized, learner_identity=learner) == quantized
    for bad in ({**declaration(), 'learner_checkpoint_sha256': 'f' * 64},
                {**declaration(), 'actors': {LEARNER: {'lineage_step': 1, 'export': None}}},
                {**declaration(), 'actors': {ACTOR: {'lineage_step': -1, 'export': None}}},
                declaration({'kind': 'lossy', 'format': 'x', 'sha256': 'a' * 64})):
        with pytest.raises(ValueError):
            actor_declaration(bad, learner_identity=learner)
    decl = actor_declaration(declaration(), learner_identity=learner)
    assert accepted_policy(learner, learner, None)
    assert accepted_policy({**learner, 'checkpoint_sha256': ACTOR}, learner, decl)
    assert not accepted_policy({**learner, 'checkpoint_sha256': ACTOR}, learner, None)
    assert not accepted_policy({'checkpoint_sha256': ACTOR, 'tokenizer_fingerprint': '5' * 64}, learner, decl)


def test_quantized_same_checkpoint_requires_actual_bound_export():
    learner = {'checkpoint_sha256': LEARNER}
    export = {'kind': 'quantized', 'format': 'gguf-bf16-with-nvfp4', 'sha256': 'a' * 64}
    declaration = {'schema': 'natlang.rl-actors/1', 'learner_checkpoint_sha256': LEARNER,
                   'actors': {LEARNER: {'lineage_step': 10, 'export': export}}}
    declaration = actor_declaration(declaration, learner_identity=learner)
    capture = {'policy': learner, 'actor_export': export}
    validate_capture_export(capture, declaration)
    with pytest.raises(ValueError, match='requires its exact declaration'):
        validate_capture_export(capture, None)
    with pytest.raises(ValueError, match='differs from its exact declaration'):
        validate_capture_export({**capture, 'actor_export': {**export, 'sha256': 'b' * 64}}, declaration)
    with pytest.raises(ValueError, match='differs from its exact declaration'):
        validate_capture_export({'policy': learner}, declaration)


def test_quantized_same_checkpoint_requires_importance_and_is_retained_in_objective(tmp_path):
    from natlang_neuralese.rl.native_replay import NativePolicyObjective, prepare_objective
    model, roles, actor, _, block, episodes = stale_episode()
    export = {'kind': 'quantized', 'format': 'gguf-bf16-with-nvfp4', 'sha256': 'a' * 64}
    for episode in episodes:
        for turn in episode['turns']:
            turn['sampling']['actor_export'] = export
    actors = {'schema': 'natlang.rl-actors/1', 'learner_checkpoint_sha256': ACTOR,
              'actors': {ACTOR: {'lineage_step': 1200, 'export': export}}}
    with pytest.raises(ValueError, match='requires its exact declaration'):
        prepare_objective(copy.deepcopy(episodes), identity=actor, roles=roles, fetch_block=lambda _: block,
                          policy_loss={'schema': 'score-centered/1', 'k': 8})
    with pytest.raises(ValueError, match='importance-corrected'):
        prepare_objective(copy.deepcopy(episodes), identity=actor, roles=roles, fetch_block=lambda _: block,
                          policy_loss={'schema': 'score-centered/1', 'k': 8, 'importance': 'none'}, actors=actors)
    _, spec = prepare_objective(copy.deepcopy(episodes), identity=actor, roles=roles, fetch_block=lambda _: block,
                                policy_loss={'schema': 'score-centered/1', 'k': 8}, actors=actors)
    assert spec['actors'] == actors
    assert all(row['actor_checkpoint_sha256'] == ACTOR for row in spec['rows'].values())
    path = tmp_path / 'objective.json'
    path.write_text(json.dumps(spec))
    NativePolicyObjective(path)


def test_stale_actor_episodes_train_the_learner_with_importance_correction(tmp_path):
    from natlang_neuralese.rl.native_replay import NativePolicyObjective, prepare_objective
    from natlang_neuralese.data.native_graph import parse_graph
    model, roles, actor, learner, block, episodes = stale_episode()
    fetch = lambda key: block  # noqa: E731
    with pytest.raises(ValueError, match='exact scored train-only actor'):
        prepare_objective(copy.deepcopy(episodes), identity=learner, roles=roles, fetch_block=fetch,
                          policy_loss={'schema': 'score-centered/1', 'k': 8})
    with pytest.raises(ValueError, match='importance-corrected'):
        prepare_objective(copy.deepcopy(episodes), identity=learner, roles=roles, fetch_block=fetch,
                          policy_loss={'schema': 'score-centered/1', 'k': 8, 'importance': 'none'},
                          actors=declaration())
    foreign = copy.deepcopy(episodes)
    foreign[0]['turns'][0]['sampling']['policy'] = {**actor, 'tokenizer_fingerprint': '5' * 64}
    with pytest.raises(ValueError, match='different policy'):
        prepare_objective(foreign, identity=learner, roles=roles, fetch_block=fetch,
                          policy_loss={'schema': 'score-centered/1', 'k': 8}, actors=declaration())
    graphs, spec = prepare_objective(copy.deepcopy(episodes), identity=learner, roles=roles, fetch_block=fetch,
                                     policy_loss={'schema': 'score-centered/1', 'k': 8}, actors=declaration())
    assert spec['policy'] == learner and spec['policy_loss']['importance'] == 'tis'
    assert spec['actors'] == declaration()
    assert {row['actor_checkpoint_sha256'] for row in spec['rows'].values()} == {ACTOR}
    path = tmp_path / 'objective.json'
    path.write_text(json.dumps(spec))
    objective = NativePolicyObjective(path)
    example = parse_graph(graphs[0], tokenizer_fingerprint=learner['tokenizer_fingerprint'])
    context = {'examples': [example], 'row_offsets': {example.id: {'physical_start': 0, 'external_names': {}}}}
    loss, detail = objective(model, model(example.plan, retain_states=False), context)
    assert detail['sampled_tokens'] == 7 and torch.isfinite(loss)
    loss.backward()
    assert model.adapter.up.weight.grad.abs().sum() > 0
    tampered = copy.deepcopy(spec)
    next(iter(tampered['rows'].values()))['actor_checkpoint_sha256'] = 'e' * 64
    path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match='undeclared actor'):
        NativePolicyObjective(path)
    path.write_text(json.dumps({**spec, 'actors': {**declaration(), 'learner_checkpoint_sha256': 'f' * 64}}))
    with pytest.raises(ValueError, match='exact learner checkpoint'):
        NativePolicyObjective(path)
