"""Program-tuning recipe schedules and whole-graph RL on recorded native rollouts (plan §5.1a)."""
import copy
import json
import math
from pathlib import Path

import pytest
import torch

from test_native_serving_adapters import engine_fixture
from test_rl_episode_graph import attach_execution_graph
from test_tune_trainer import manifest_for, rows_for, tensors
from natlang_neuralese.rl.policy_math import default_policy_loss
from natlang_neuralese.serve.engine import GenerationRequest
from natlang_neuralese.tune import TuneMathError
from natlang_neuralese.tune.rl import ROLLOUT_SCHEMA
from natlang_neuralese.tune.soft import soft_key
from natlang_neuralese.tune.trainer import JointTrainer, Recipe, lr_factor

NEUTRAL = {'level_lr_scale': {'tree': 1., 'folder': 1., 'lambda': 1.}, 'schedule': {'kind': 'constant'},
           'depth_penalty': {'tree': 0., 'folder': 0., 'lambda': 0.}}


def test_schedule_math():
    assert lr_factor({'kind': 'constant'}, 7, 10) == 1.
    schedule = {'kind': 'warmup-cosine', 'warmup_ratio': .2, 'min_ratio': .1}
    factors = [lr_factor(schedule, step, 10) for step in range(10)]
    assert factors[:3] == [.5, 1., 1.]
    assert all(a >= b for a, b in zip(factors[1:], factors[2:]))
    assert factors[3] == pytest.approx(.1 + .9 * .5 * (1 + math.cos(math.pi / 7))) and factors[-1] == pytest.approx(.1)
    assert lr_factor(schedule, 10, 10) == pytest.approx(.1) == lr_factor(schedule, 50, 10)
    assert lr_factor({'kind': 'warmup-cosine', 'warmup_steps': 0, 'min_ratio': 0.}, 0, 4) == 1.
    for bad in ({'kind': 'linear'}, {'kind': 'constant', 'min_ratio': .1}, {'kind': 'warmup-cosine', 'min_ratio': 2},
                {'kind': 'warmup-cosine', 'warmup_ratio': 1.}):
        with pytest.raises(TuneMathError):
            Recipe.from_json({'schedule': bad})
    with pytest.raises(TuneMathError, match='level_lr_scale'):
        Recipe.from_json({'level_lr_scale': {'tree': 1.}})
    with pytest.raises(TuneMathError, match='objectives'):
        Recipe.from_json({'objectives': ['dpo']})
    # Recipes written before these fields stay valid.
    assert Recipe.from_json({'steps': 3, 'batch_rows': 2, 'optimizer': {'name': 'adam', 'learning_rate': 1e-3}}).steps == 3
    assert Recipe.from_json({'prompt_weight': 0}).prompt_weight == 0
    # Own outputs only by default (owner 2026-10-28); an in-flight fit keeps the weight it was bound with.
    from natlang_neuralese.tune.trainer import LEGACY_PROMPT_WEIGHT, resolve_prompt_weight
    assert Recipe().prompt_weight == 0 and Recipe.from_json({}).prompt_weight == 0
    assert resolve_prompt_weight({}) == 0 and resolve_prompt_weight({'prompt_weight': .3}) == .3
    assert resolve_prompt_weight({}, {'input_sha256': 'x'}) == LEGACY_PROMPT_WEIGHT == .1
    assert resolve_prompt_weight({}, {'recipe_prompt_weight': 0.}) == 0
    assert resolve_prompt_weight({'prompt_weight': .2}, {'input_sha256': 'x'}) == .2
    for weight in (-1, 1.01, float('nan'), float('inf'), True):
        with pytest.raises(TuneMathError, match='prompt_weight'):
            Recipe.from_json({'prompt_weight': weight})


def test_epochs_derive_the_step_budget_and_the_schedule_resumes_from_the_persisted_step(tmp_path):
    engine = engine_fixture(tmp_path, 'lfm')
    manifest, rows = manifest_for(engine), rows_for(engine, 'bx', 3)
    recipe = Recipe(epochs=2, batch_rows=2, schedule={'kind': 'warmup-cosine', 'warmup_steps': 1, 'min_ratio': .2})
    continuous = JointTrainer(engine, manifest, rows, recipe)
    assert continuous.total_steps == 3
    records = continuous.run()
    assert [record['lr_factor'] for record in records] == [lr_factor(recipe.schedule, step, 3) for step in range(3)]
    assert records[0]['lr_factor'] == 1. and records[-1]['lr_factor'] == pytest.approx(.2)
    receipt = continuous.receipt()
    assert receipt['schedule']['total_steps'] == 3 and receipt['schedule']['derived_from'] == 'epochs'
    assert receipt['exposure']['rows_seen'] == 6
    interrupted = JointTrainer(engine, manifest, rows, recipe)
    interrupted.run(1)
    resumed = JointTrainer(engine, manifest, rows, recipe, state=json.loads(json.dumps(interrupted.state)))
    resumed.run()
    assert resumed.state['node_artifacts'] == continuous.state['node_artifacts']
    assert [record['lr_factor'] for record in resumed.state['losses']] == [record['lr_factor'] for record in records]


def test_per_level_rates_are_applied(tmp_path):
    engine = engine_fixture(tmp_path, 'lfm')
    manifest, rows = manifest_for(engine), rows_for(engine, 'bx', 2)
    deltas = {}
    for name, scales in (('neutral', NEUTRAL['level_lr_scale']), ('scaled', {'tree': .25, 'folder': .5, 'lambda': 1.})):
        trainer = JointTrainer(engine, manifest, rows, Recipe(batch_rows=2, optimizer={'name': 'sgd', 'learning_rate': 1.},
                                                              **{**NEUTRAL, 'level_lr_scale': scales}))
        start = {node: tensors(engine, artifact) for node, artifact in trainer.state['node_artifacts'].items()}
        record = trainer.step()
        assert record['learning_rates'] == {node: scales[trainer.nodes[node]['level']] for node in ('folder:a', 'lambda:x', 'tree:')}
        deltas[name] = {node: {key: tensors(engine, trainer.state['node_artifacts'][node])[key] - value
                               for key, value in start[node].items()} for node in ('tree:', 'folder:a', 'lambda:x')}
    for node, scale in (('tree:', .25), ('folder:a', .5), ('lambda:x', 1.)):
        for key, value in deltas['neutral'][node].items():
            assert value.abs().max() > 0
            torch.testing.assert_close(deltas['scaled'][node][key], value * scale, rtol=1e-5, atol=1e-9)


# -- RL --------------------------------------------------------------------------------------------------------------
def sample(engine, monkeypatch, tokens, entries, messages, seed, **options):
    """A real engine generation with chosen actions and their actual normalized behavior log-probabilities."""
    selected = iter(tokens)
    def token(seq):
        value = next(selected)
        return value, float(seq.logits[0].float().div(seq.request.temperature).log_softmax(-1)[value])
    monkeypatch.setattr(engine, '_native_token', token)
    return engine.generate(GenerationRequest(messages=messages, adapters=entries, temperature=.8, seed=seed,
                                             max_tokens=8, rl_capture=True, rl_top_k=128, **options))


def stack(trainer, binding):
    return [{'id': trainer.state['node_artifacts'][node], 'scale': 1.} for node in trainer.bindings[binding]['ancestry']]


def rollout(engine, trainer, turns, bindings, sample_index, reward, soft_states=None):
    episode = {'task_id': 'root-1', 'split': 'train', 'sample': sample_index, 'group': 'root-1/round-0',
               'reward': {'quality': reward}, 'policy': {'checkpoint_sha256': engine.native_identity['checkpoint_sha256']},
               'turns': [{'messages': [], 'target': {}, 'invocation_id': call, 'sampling': capture,
                          'replay': {'raw_calls_retained': True}} for call, capture in turns]}
    attach_execution_graph([episode])
    return {'schema': ROLLOUT_SCHEMA, 'root': 'root-1', 'component': 'c', 'partition': 'train', 'group': 'root-1/round-0',
            'sample': sample_index, 'reward': {'quality': reward, 'source': 'root-oracle'},
            'behavior': {'release': None, 'node_artifacts': dict(trainer.state['node_artifacts']), 'gates': {},
                         'soft_states': soft_states or {}, 'temperature': .8, 'seed': 21 + sample_index},
            'bindings': bindings, 'episode': episode}


def two_invocation_rollouts(engine, trainer, monkeypatch, *, writer_binding='bx'):
    """Writer (binding bx) exports a Neuralese block that the reader (binding by) imports; two samples, rewards 1/0."""
    roles, records = engine.role_tokens, []
    for index, (written, read) in enumerate((((5, 6), 7), ((9, 10), 11))):
        writer = sample(engine, monkeypatch, [roles.produce_open, *written, roles.produce_close, 2],
                        stack(trainer, writer_binding) if writer_binding is not None else None,
                        [{'role': 'user', 'content': 't4 t5'}], 21 + index)
        block = writer['native_state']['blocks'][0]['id']
        reader = sample(engine, monkeypatch, [read, 2], stack(trainer, 'by'),
                        [{'role': 'user', 'content': [{'type': 'text', 'text': 't6 '}, {'type': 'native_state', 'id': block}]}],
                        31 + index, native_state_modes={block: 'same-base-adapter-state'})
        records.append(rollout(engine, trainer, [('writer', writer['x_natlang_rl_capture']), ('reader', reader['x_natlang_rl_capture'])],
                               {'writer': writer_binding, 'reader': 'by'}, index, 1 - index))
    return records


def parent_fit(engine):
    return JointTrainer(engine, manifest_for(engine), rows_for(engine, 'bx', 2) + rows_for(engine, 'by', 1), Recipe(steps=0))


@pytest.mark.parametrize('family', ['lfm', 'mellum'])
def test_rl_base_producer_stays_in_whole_graph_with_tuned_consumer(tmp_path, monkeypatch, family):
    engine = engine_fixture(tmp_path, family, service_template=True)
    parent = parent_fit(engine)
    records = two_invocation_rollouts(engine, parent, monkeypatch, writer_binding=None)
    trainer = JointTrainer(engine, manifest_for(engine), records,
        Recipe(objectives=['rl'], batch_rows=2, optimizer={'name':'sgd', 'learning_rate':.5}, **NEUTRAL),
        workdir=tmp_path/'rl', parent=copy.deepcopy(parent.state))
    assert all(len(graph['nodes']) == 2 for graph in trainer.rl.graphs)
    before = dict(trainer.state['node_artifacts'])
    update = trainer.step()
    assert update['objective'] == 'rl' and len(update['row_losses']) == 2
    assert update['updated_nodes'] == ['folder:b', 'lambda:y', 'tree:']
    assert trainer.state['node_artifacts']['folder:a'] == before['folder:a']
    assert trainer.state['node_artifacts']['lambda:x'] == before['lambda:x']
    assert any(trainer.state['node_artifacts'][node] != before[node] for node in update['updated_nodes'])


def ratios(engine, records, state, workdir):
    """Mean current/behavior likelihood ratio of each episode under ``state``: a clipped-ratio probe fit continued
    from it (its unclipped policy loss is -advantage * mean ratio)."""
    probe = JointTrainer(engine, manifest_for(engine), records,
                         Recipe(objectives=['rl'], batch_rows=2, rl_policy_loss={'schema': 'clipped-ratio/1'}, **NEUTRAL),
                         workdir=workdir, parent=copy.deepcopy(state))
    losses = probe.rl_losses()
    return {ident: -loss / probe.rl.advantages[ident] for ident, loss in losses.items()}


def test_coefficient_update_with_captured_guidance_and_decision(tmp_path):
    from natlang_neuralese.model.native_adapters import run_native, execution_identity, execution_binding, SCHEMA
    from test_rl_jolt import episode_with_top_k, rollouts as graph_rollouts, source_for, DECLARATION
    from test_jolt_privileged_context import train_case
    engine = engine_fixture(tmp_path, 'lfm', service_template=True)
    parent = parent_fit(engine)
    entries = stack(parent, 'bx')
    def capture():
        identity = execution_identity(engine)
        binding = {'schema': SCHEMA, 'base_native_identity': dict(engine.native_identity),
                   'adapters': execution_binding(engine)}
        episode = episode_with_top_k(engine.shared_model, engine.role_tokens, identity, 'write-only',
                                     adapter_binding=binding, compile_fixture=False)
        for turn in episode.turns:
            receipt = turn['sampling']
            receipt['policy'] = identity
            receipt['adapter_binding'] = binding
            receipt['schema'] = 'natlang.native-rl-sampling/3'
            prompt_length = sum(len(segment['token_ids']) for segment in receipt['prompt'] if segment['kind'] == 'tokens')
            receipt['generated_inputs'] = []
            for sample in receipt['samples']:
                sample['query_own_position'] = prompt_length + sample['index'] - 1
                if not sample['terminal']:
                    receipt['generated_inputs'].append({'token_id':sample['token_id'],'input_role':sample['input_role'],
                                                        'sample_index':sample['index']})
            for write in receipt['writes']:
                for name, old in (('own_begin','begin_sample'),('own_end','end_sample'),
                                  ('payload_own_begin','payload_begin'),('payload_own_end','payload_end')):
                    write[name] = prompt_length + write[old]
        root = next(turn for turn in episode.turns if turn['graph_turn'] == 'root#turn1')
        old = root['sampling']
        root['sampling'] = {'schema':'natlang.native-decision-capture/1','policy':identity,'adapter_binding':binding,
            'prompt':old['prompt'],'privileged':old['privileged'],'continuations':[[11,2],[13,2]],
            'decision_sample':{'schema':'natlang.decision-sample/1','log_probs':[-1.,-2.],
                               'chosen':0,'temperature':1.,'seed':3}}
        return episode
    episode = run_native(engine, capture, entries)
    for block in episode.states.values():
        engine.store.put_native_state(block)
    episodes = graph_rollouts(episode.turns)
    records = []
    for index, item in enumerate(episodes):
        item['policy']['checkpoint_sha256'] = engine.native_identity['checkpoint_sha256']
        records.append({'schema':ROLLOUT_SCHEMA, 'root':train_case()['id'], 'component':'c','partition':'train',
            'group':item['group'],'sample':index,'reward':{'quality':item['reward']['quality'],'source':'root-oracle'},
            'behavior':{'release':None,'node_artifacts':dict(parent.state['node_artifacts']),'gates':{},
                        'soft_states':{},'temperature':.8,'seed':21+index},
            'bindings':{turn['invocation_id']:'bx' for turn in item['turns']},'episode':item})
    trainer = JointTrainer(engine, manifest_for(engine), records,
        Recipe(objectives=['rl'],batch_rows=2,optimizer={'name':'sgd','learning_rate':.5},
               rl_policy_loss=DECLARATION,**NEUTRAL), workdir=tmp_path/'joint', parent=copy.deepcopy(parent.state),
        privileged=source_for(episodes,episode))
    objective = json.loads(Path(trainer.rl.objective['path']).read_text())
    assert all(row['jolt']['privileged'] is not None and row['decision_actions'][0]['twin_options']
               for row in objective['rows'].values())
    before = dict(trainer.state['node_artifacts'])
    update = trainer.step()
    assert update['updated_nodes'] and any(trainer.state['node_artifacts'][key] != value for key,value in before.items())


def test_rl_sign_and_whole_graph_coverage(tmp_path, monkeypatch):
    torch.manual_seed(1)
    engine = engine_fixture(tmp_path, 'lfm', service_template=True)
    parent = parent_fit(engine)
    records = two_invocation_rollouts(engine, parent, monkeypatch)
    trainer = JointTrainer(engine, manifest_for(engine), records,
                           Recipe(objectives=['rl'], batch_rows=2, optimizer={'name': 'sgd', 'learning_rate': .5}, **NEUTRAL),
                           workdir=tmp_path / 'rl', parent=copy.deepcopy(parent.state))
    # The shared default: JOLT+ with score-centered/TIS outcome terms over per-invocation tuning actors.
    assert trainer.state['recipe_rl_policy_loss'] == default_policy_loss()
    spec = json.loads(Path(trainer.rl.objective['path']).read_text())
    assert spec['policy_loss'] == default_policy_loss() and 'recipe_transitions' not in trainer.state
    assert all(row['jolt']['privileged'] is None for row in spec['rows'].values())  # no privileged source here
    assert len(trainer.rl.graphs) == 2 and trainer.rl.skipped == 0
    positive = next(graph['id'] for graph in trainer.rl.graphs if trainer.rl.advantages[graph['id']] > 0)
    negative = next(graph['id'] for graph in trainer.rl.graphs if trainer.rl.advantages[graph['id']] < 0)
    before = ratios(engine, records, trainer.state, tmp_path / 'probe-before')
    assert all(value == pytest.approx(1., abs=1e-4) for value in before.values()), 'current equals behavior at first'
    start = dict(trainer.state['node_artifacts'])
    record = trainer.step()
    assert record['objective'] == 'rl' and len(record['row_losses']) == 2
    # Whole graph: the writer's and the reader's bindings both update their complete paths.
    assert record['updated_nodes'] == sorted(start)
    for node in start:
        old, new = tensors(engine, start[node]), tensors(engine, trainer.state['node_artifacts'][node])
        assert any(not torch.equal(old[key], new[key]) for key in new), node
        assert trainer.state['node_history'][node][-2:] == [start[node], trainer.state['node_artifacts'][node]]
    assert all(math.isfinite(value) for value in record['row_losses'])
    after = ratios(engine, records, trainer.state, tmp_path / 'probe-after')
    assert after[positive] > before[positive] and after[negative] < before[negative]
    assert trainer.receipt()['exposure']['rl_episodes_seen'] == 2
    # Rollouts sampled from the updated release are recorded-lineage behavior of the same continued fit.
    newer = two_invocation_rollouts(engine, trainer, monkeypatch)
    continued = JointTrainer(engine, manifest_for(engine), newer + records,
                             Recipe(objectives=['rl'], batch_rows=4, **NEUTRAL), workdir=tmp_path / 'rl2',
                             parent=copy.deepcopy(trainer.state))
    assert continued.state['recipe_prompt_weight'] == 0 and 'recipe_transitions' not in continued.state
    # A parent bound before the own-outputs default (implicit 0.1) is continued at 0 with a recorded transition.
    legacy = copy.deepcopy(trainer.state)
    legacy.pop('recipe_prompt_weight')
    adopted = JointTrainer(engine, manifest_for(engine), newer + records,
                           Recipe(objectives=['rl'], batch_rows=4, **NEUTRAL), workdir=tmp_path / 'rl3', parent=legacy)
    assert adopted.state['recipe_transitions'][-1]['prompt_weight'] == {'from': .1, 'to': 0.}
    # A parent that trained RL before the JOLT+ default (implicit clipped ratio) adopts it with a recorded transition;
    # resuming that parent's own in-flight state keeps its bound loss.
    legacy.pop('recipe_rl_policy_loss')
    assert any(key.startswith('rl:') for key in legacy['exposure'])
    transitioned = JointTrainer(engine, manifest_for(engine), newer + records,
                                Recipe(objectives=['rl'], batch_rows=4, **NEUTRAL), workdir=tmp_path / 'rl4', parent=legacy)
    assert transitioned.state['recipe_transitions'][-1]['rl_policy_loss'] == {
        'from': {'schema': 'clipped-ratio/1'}, 'to': default_policy_loss()}
    assert transitioned.state['recipe_rl_policy_loss'] == default_policy_loss()
    from natlang_neuralese.tune.trainer import resolve_rl_policy_loss
    assert resolve_rl_policy_loss(Recipe(**NEUTRAL), legacy) == {'schema': 'clipped-ratio/1'}
    with pytest.raises(TuneMathError, match='bound RL policy loss'):
        resolve_rl_policy_loss(Recipe(rl_policy_loss=default_policy_loss(), **NEUTRAL), legacy)
    record = continued.step()
    assert record['updated_nodes'] == sorted(start)
    assert sorted(record['optimizer_restarted']) == sorted(start), 'SGD history is not reinterpreted as Adam moments'


def test_rl_updates_consumed_soft_states(tmp_path, monkeypatch):
    torch.manual_seed(2)
    engine = engine_fixture(tmp_path, 'lfm', service_template=True)
    parent = parent_fit(engine)
    roles = engine.role_tokens
    # A stored soft instruction state of binding by, written outside the episode.
    writer = sample(engine, monkeypatch, [roles.produce_open, 5, 6, roles.produce_close, 2], None,
                    [{'role': 'user', 'content': 't20 t21'}], 11)
    state = writer['native_state']['blocks'][0]['id']
    key = soft_key('by', 'instructions', None)
    records = []
    for index, read in enumerate((7, 11)):
        reader = sample(engine, monkeypatch, [read, 2], stack(parent, 'by'),
                        [{'role': 'user', 'content': [{'type': 'text', 'text': 't6 '}, {'type': 'native_state', 'id': state}]}],
                        41 + index, native_state_modes={state: 'same-base-adapter-state'})
        records.append(rollout(engine, parent, [('reader', reader['x_natlang_rl_capture'])], {'reader': 'by'}, index,
                               1 - index, soft_states={key: state}))
    trainer = JointTrainer(engine, manifest_for(engine), records,
                           Recipe(objectives=['rl'], batch_rows=2, optimizer={'name': 'sgd', 'learning_rate': .5},
                                  state_optimizer={'name': 'adam', 'learning_rate': 1e-2}, **NEUTRAL),
                           workdir=tmp_path / 'rl', parent=copy.deepcopy(parent.state))
    assert trainer.state['soft_states'] == {key: state}
    record = trainer.step()
    assert record['updated_soft_states'] == [key] and record['updated_nodes'] == ['folder:b', 'lambda:y', 'tree:']
    from natlang_neuralese.model.native_state_derivation import state_values
    updated = trainer.state['soft_states'][key]
    assert updated != state and trainer.state['soft_history'][key] == [state, updated]
    old, new = (state_values(engine.store.get_native_state(ident).block) for ident in (state, updated))
    assert any(not torch.equal(old[name], new[name]) for name in new)


def test_rl_rejects_episodes_of_other_actors(tmp_path, monkeypatch):
    engine = engine_fixture(tmp_path, 'lfm', service_template=True)
    parent = parent_fit(engine)
    records = two_invocation_rollouts(engine, parent, monkeypatch)
    recipe = Recipe(objectives=['rl'], batch_rows=2, **NEUTRAL)
    # Behavior versions that are not part of this fit's recorded lineage.
    other = JointTrainer(engine, manifest_for(engine), rows_for(engine, 'bx', 2), Recipe(steps=0, parameterization='free-lora'))
    assert other.state['node_artifacts'] != parent.state['node_artifacts']
    with pytest.raises(TuneMathError, match='lineage'):
        JointTrainer(engine, manifest_for(engine), records, recipe, workdir=tmp_path / 'a', parent=copy.deepcopy(other.state))
    # An invocation sampled by another binding's stack than its declared binding.
    swapped = copy.deepcopy(records)
    for record in swapped:
        record['bindings'] = {'writer': 'by', 'reader': 'bx'}
    with pytest.raises(TuneMathError, match='another actor'):
        JointTrainer(engine, manifest_for(engine), swapped, recipe, workdir=tmp_path / 'b', parent=copy.deepcopy(parent.state))
    # Objectives must match the admitted row kinds.
    with pytest.raises(TuneMathError, match='objectives'):
        JointTrainer(engine, manifest_for(engine), records, Recipe(**NEUTRAL), workdir=tmp_path / 'c', parent=parent.state)


def test_rft_supervises_whole_successful_episodes_through_their_neuralese_edges(tmp_path, monkeypatch):
    """Global SFT: teacher-forced loss on every action of the successful episode's whole graph, so the writer that
    exported the Neuralese block and the reader that imported it both learn; the failed episode is not supervised."""
    torch.manual_seed(2)
    engine = engine_fixture(tmp_path, 'lfm', service_template=True)
    parent = parent_fit(engine)
    records = two_invocation_rollouts(engine, parent, monkeypatch)
    trainer = JointTrainer(engine, manifest_for(engine), records,
                           Recipe(objectives=['rft'], batch_rows=2, optimizer={'name': 'sgd', 'learning_rate': .5}, **NEUTRAL),
                           workdir=tmp_path / 'rl', parent=copy.deepcopy(parent.state))
    episodes = trainer.episodes['rft']
    assert len(episodes.graphs) == 1 and episodes.skipped == 1 and trainer.rl is None
    assert len(episodes.graphs[0]['nodes']) == 2
    before = trainer.rl_losses('rft')
    start = dict(trainer.state['node_artifacts'])
    record = trainer.step()
    assert record['objective'] == 'rft' and record['updated_nodes'] == sorted(start)
    after = trainer.rl_losses('rft')
    assert all(after[key] < before[key] for key in before), 'the supervised episode becomes more likely'
    receipt = trainer.receipt()
    assert receipt['exposure']['rft_episodes_seen'] == 1 and receipt['rft']['admitted_episodes'] == 1
    assert receipt['objectiveCycle'] == ['rft']


def test_graded_rewards_and_objective_cycle(tmp_path, monkeypatch):
    engine = engine_fixture(tmp_path, 'lfm', service_template=True)
    parent = parent_fit(engine)
    records = two_invocation_rollouts(engine, parent, monkeypatch)
    for record, quality in zip(records, (.75, .25)):
        record['reward'] = {'quality': quality, 'source': 'root-verifier'}
        record['episode']['reward'] = {'quality': quality}
    trainer = JointTrainer(engine, manifest_for(engine), records,
                           Recipe(objectives=['rl', 'rft'], batch_rows=2, rft_min_quality=.5, **NEUTRAL),
                           workdir=tmp_path / 'rl', parent=copy.deepcopy(parent.state))
    assert len(trainer.rl.graphs) == 2 and len(trainer.episodes['rft'].graphs) == 1
    assert [trainer.step()['objective'] for _ in range(2)] == ['rl', 'rft']
    # Without a successful episode at the threshold, RFT has no work and leaves the cycle.
    strict = JointTrainer(engine, manifest_for(engine), records, Recipe(objectives=['rl', 'rft'], batch_rows=2, **NEUTRAL),
                          workdir=tmp_path / 'rl2', parent=copy.deepcopy(parent.state))
    assert strict.cycle == ['rl'] and strict.receipt()['rft']['admitted_episodes'] == 0
