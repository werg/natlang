"""Real native graph replay matches the actor's incremental sampling inputs."""
import copy
from types import SimpleNamespace

import pytest
import torch

from natlang_neuralese.model.shared_state import SharedStateModel, native_step, splice_block, concatenate_captures
from natlang_neuralese.model.state_block import NativeStateBlock
from natlang_neuralese.rl.native_replay import compile_replay, replay_logps
from natlang_neuralese.serve.store import native_state_id
from test_shared_state import tiny


def actor_rollout(family,*,loaded=None,text='payload'):
    if loaded is None:
        torch.manual_seed(901)
        model = SharedStateModel(tiny(family), marker_ids=(58, 59, 60, 61), rank=4)
        roles = SimpleNamespace(produce_open=58, produce_close=59, consume_open=60, consume_close=61)
        identity = {'checkpoint_sha256': '1'*64, 'tokenizer_fingerprint': '2'*64,
                    'embedding_weight_sha256': '3'*64, 'architecture': model.port.config.model_type}
    else:model,roles,identity=loaded
    device=next(model.parameters()).device
    def capture(prompt):
        return {'schema': 'natlang.native-rl-sampling/1', 'policy': dict(identity), 'prompt': prompt,
                'samples': [], 'writes': [], 'sample_range_convention': 'half-open', 'finish_reason': 'stop',
                'sampling_law': {'temperature': .8, 'seed': 3, 'guidance': False,
                                 'probabilities': 'post-processing-normalized'}}
    def sample(out, record, tokens):
        chunks = []
        for index, (token, producing) in enumerate(tokens):
            logp = model.logits(out['h_final'][:, -1]).float().div(.8).log_softmax(-1)[0, token]
            record['samples'].append({'index': index, 'token_id': token, 'logp': float(logp),
                'input_role': 'produce' if producing else 'ordinary', 'terminal': token == 2})
            if token != 2:
                out = native_step(model, torch.tensor([[token]],device=device), out['cache'], producing=producing)
                if producing:
                    chunks.append(out)
        return out, chunks
    with torch.no_grad():
        producer = capture([{'kind': 'tokens', 'token_ids': [1, 4], 'producing': [False, False]}])
        _, chunks = sample(native_step(model, torch.tensor([[1, 4]],device=device)), producer,
                           [(roles.produce_open, False), (5, True), (6, True), (roles.produce_close, False), (2, False)])
        block = NativeStateBlock(text=text, **identity, **concatenate_captures(chunks)).validate()
        block_id = native_state_id(block)
        producer['writes'] = [{'state_id': block_id, 'begin_sample': 0, 'end_sample': 4,
                              'payload_begin': 1, 'payload_end': 3, 'native_identity': identity}]
        consumer = capture([{'kind': 'tokens', 'token_ids': [3, roles.consume_open], 'producing': [False, False]},
            {'kind': 'native_state', 'id': block_id, 'native_identity': identity},
            {'kind': 'tokens', 'token_ids': [roles.consume_close], 'producing': [False]}])
        out = native_step(model, torch.tensor([[3, roles.consume_open]],device=device))
        cache = splice_block(model, out['cache'], block,
                             **{key: value for key, value in identity.items() if key != 'architecture'})
        out = native_step(model, torch.tensor([[roles.consume_close]],device=device), cache)
        sample(out, consumer, [(11, False), (2, False)])
    turns = [{'graph_turn': 'consumer', 'sampling': consumer, 'read_bindings': {block_id: 'write-0'}},
             {'graph_turn': 'producer', 'sampling': producer, 'written_nodes': {block_id: 'write-0'}}]
    return model, roles, identity, block_id, block, turns


@pytest.mark.parametrize('family', ['lfm', 'mellum'])
def test_graph_replay_matches_incremental_actor_including_write_and_terminal_actions(family):
    model, roles, identity, block_id, block, turns = actor_rollout(family)
    replay = compile_replay(turns, identity=identity, roles=roles, fetch_block=lambda key: block)
    assert [node.name for node in replay.plan.invocations] == ['producer', 'consumer']
    assert len(replay.actions) == 7  # both delimiters, payload, ordinary reply and BOTH sampled STOPs
    logps, behavior = replay_logps(model, replay, chunk_tokens=2)
    torch.testing.assert_close(logps.detach(), behavior, rtol=2e-5, atol=2e-6)
    consumer = torch.tensor([i for i, action in enumerate(replay.actions) if action['turn'] == 'consumer'])
    (-logps[consumer].mean()).backward()
    assert model.adapter.up.weight.grad is not None
    assert model.adapter.up.weight.grad.abs().sum() > 0
    assert torch.isfinite(model.adapter.up.weight.grad).all()
    assert block.token_ids.tolist() == [5, 6]  # no learner mutation of archived observations


@pytest.mark.parametrize('stop', [False, True])
def test_optional_import_stop_keyword_preserves_legacy_default_and_explicit_stop(monkeypatch, stop):
    from natlang_neuralese.rl import native_replay
    _, roles, identity, _, block, turns = actor_rollout('lfm')
    for segment in turns[0]['sampling']['prompt']:
        if segment['kind'] == 'native_state' and stop:
            segment['stop_gradient'] = True
    constructor = native_replay.ImportSpan
    seen = []
    def import_span(at, source, start, end, **options):
        seen.append(options)
        if not stop:
            assert not options  # Old production ports implement the default four-field import.
        return constructor(at, source, start, end, **options)
    monkeypatch.setattr(native_replay, 'ImportSpan', import_span)
    native_replay.compile_replay(turns, identity=identity, roles=roles, fetch_block=lambda key: block)
    assert seen and all(options == ({'stop_gradient': True} if stop else {}) for options in seen)


def test_native_replay_rejects_missing_law_or_mismatched_writer_payload():
    _, roles, identity, _, block, turns = actor_rollout('lfm')
    guided = copy.deepcopy(turns)
    guided[0]['sampling']['sampling_law']['guidance'] = True
    with pytest.raises(ValueError, match='normalized-mask'):
        compile_replay(guided, identity=identity, roles=roles, fetch_block=lambda key: block)
    wrong = copy.deepcopy(turns)
    wrong[1]['sampling']['samples'][1]['token_id'] = 7
    with pytest.raises(ValueError, match='archived write tokens'):
        compile_replay(wrong, identity=identity, roles=roles, fetch_block=lambda key: block)


def test_native_replay_uses_captured_normalized_guidance_mask():
    model, roles, identity, _, block, turns = actor_rollout('lfm')
    for turn in turns:
        turn['sampling']['sampling_law']['guidance'] = True
        for sample in turn['sampling']['samples']:
            sample['allowed_token_ids'] = [sample['token_id']]
            sample['logp'] = 0.
    replay = compile_replay(turns, identity=identity, roles=roles, fetch_block=lambda key: block)
    logps, behavior = replay_logps(model, replay)
    torch.testing.assert_close(logps, behavior, atol=0, rtol=0)


def test_native_policy_objective_prepares_whole_episode_and_uses_supplied_forward(tmp_path, monkeypatch):
    import json
    from natlang_neuralese.rl.native_replay import prepare_objective, NativePolicyObjective
    from natlang_neuralese.data.native_graph import parse_graph
    model, roles, identity, _, block, turns = actor_rollout('lfm')
    # Keep both outcomes in the group to obtain a genuine relative-policy signal.
    raw_turns = [{'messages': [], 'target': {}, 'invocation_id': turn['graph_turn'],
                  'sampling': turn['sampling'], 'replay': {'raw_calls_retained': True}} for turn in turns]
    episodes = [{'task_id': 'train-task', 'sample': sample, 'split': 'train', 'group': 'g',
                 'reward': {'quality': reward}, 'policy': {'checkpoint_sha256': identity['checkpoint_sha256']},
                 'turns': copy.deepcopy(raw_turns)} for sample, reward in enumerate((1, 0))]
    from test_rl_episode_graph import attach_execution_graph
    attach_execution_graph(episodes)
    graphs, spec = prepare_objective(episodes, identity=identity, roles=roles, fetch_block=lambda key: block)
    assert len(graphs) == 2 and all(len(graph['nodes']) == 2 for graph in graphs)
    spec_path = tmp_path / 'objective.json'
    spec_path.write_text(json.dumps(spec))
    objective = NativePolicyObjective(spec_path)
    example = parse_graph(graphs[0], tokenizer_fingerprint=identity['tokenizer_fingerprint'])
    context = {'examples': [example], 'row_offsets': {example.id: {'physical_start': 0, 'external_names': {}}}}
    assert objective.forward_inputs(context) == {}
    result = model(example.plan, retain_states=False)
    # Objective is a readout on this result, never another backbone invocation.
    monkeypatch.setattr(model, 'forward', lambda *args, **kwargs: pytest.fail('duplicate graph forward'))
    loss, detail = objective(model, result, context)
    loss.backward()
    assert detail['sampled_tokens'] == 7
    assert torch.isfinite(loss) and model.adapter.up.weight.grad.abs().sum() > 0
    changed = copy.deepcopy(graphs[0]); changed['nodes'][0]['token_ids'][0] = 12
    context['examples'] = [parse_graph(changed, tokenizer_fingerprint=identity['tokenizer_fingerprint'])]
    with pytest.raises(ValueError, match='graph/masks'):
        objective(model, result, context)


def receipt_v2(capture, *, forced_tokens=(), force_all=False):
    value = copy.deepcopy(capture)
    value['schema'] = 'natlang.native-rl-sampling/2'
    prompt_length = sum(len(segment['token_ids']) for segment in value['prompt'] if segment['kind'] == 'tokens')
    inputs, samples = [], []
    for sample in value['samples']:
        if force_all and sample['terminal']:
            continue  # forced-only producer returned by its actual template, no inferred EOS action
        forced = force_all or sample['token_id'] in forced_tokens
        sample_index = None
        if not forced:
            sample_index = len(samples)
            samples.append({**sample, 'index': sample_index,
                            'query_own_position': prompt_length+len(inputs)-1})
        if not sample['terminal']:
            inputs.append({'token_id': sample['token_id'], 'input_role': sample['input_role'],
                           'sample_index': sample_index})
    value['samples'], value['generated_inputs'] = samples, inputs
    for write in value['writes']:
        for source, target in [('begin_sample', 'own_begin'), ('end_sample', 'own_end'),
                               ('payload_begin', 'payload_own_begin'), ('payload_end', 'payload_own_end')]:
            write[target] = prompt_length + write.pop(source)
    return value


@pytest.mark.parametrize('forced_only_writer', [False, True])
def test_v2_forced_template_inputs_are_replayed_without_invented_policy_actions(forced_only_writer):
    model, roles, identity, _, block, turns = actor_rollout('lfm')
    turns[0]['sampling'] = receipt_v2(turns[0]['sampling'])
    turns[1]['sampling'] = receipt_v2(turns[1]['sampling'], forced_tokens=(58, 59), force_all=forced_only_writer)
    replay = compile_replay(turns, identity=identity, roles=roles, fetch_block=lambda key: block)
    logps, behavior = replay_logps(model, replay)
    torch.testing.assert_close(logps.detach(), behavior, rtol=2e-5, atol=2e-6)
    assert len(logps) == (2 if forced_only_writer else 5)
    consumer = torch.tensor([i for i, action in enumerate(replay.actions) if action['turn'] == 'consumer'])
    (-logps[consumer].mean()).backward()
    assert model.adapter.up.weight.grad.abs().sum() > 0
