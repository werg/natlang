from types import SimpleNamespace

import pytest

from natlang_neuralese.data.native_graph import parse_graph, SCHEMA
from natlang_neuralese.train.native_state_sft import (append_only_rows, scheduled_indices, _row_bindings,
    NativeGraphSource, packed_graph_loss, _evaluate)


def graph(identifier, weights=None):
    return {'schema': SCHEMA, 'id': identifier, 'split': 'train', 'source_groups': ['source-' + identifier],
            'tokenizer_fingerprint': 'tokenizer-sha', 'nodes': [
                {'name': 'answer', 'token_ids': [1, 2, 3],
                 'target_weights': weights or [0, 1, 1]}]}


def test_native_data_continuation_only_adds_exact_rows():
    old = [parse_graph(graph('a'), tokenizer_fingerprint='tokenizer-sha'),
           parse_graph(graph('b'), tokenizer_fingerprint='tokenizer-sha')]
    expanded = old + [parse_graph(graph('c'), tokenizer_fingerprint='tokenizer-sha')]
    prior, current = _row_bindings(old), _row_bindings(expanded)
    assert append_only_rows(prior, current, label='training-source') == {'c'}
    with pytest.raises(ValueError, match='modified/removed'):
        append_only_rows(prior, _row_bindings([old[0]]), label='training-source')
    changed = parse_graph(graph('a', [0, .5, 1]), tokenizer_fingerprint='tokenizer-sha')
    with pytest.raises(ValueError, match='modified/removed'):
        append_only_rows(prior, _row_bindings([changed, old[1]]), label='training-source')


def test_append_schedule_preserves_resumed_epoch_prefix_and_deterministic_next_epoch():
    rows = [SimpleNamespace(id=key) for key in ('old-a', 'old-b', 'new-a', 'new-b')]
    old_ids = {'old-a', 'old-b'}
    previous = scheduled_indices(rows[:2], seed=17, epoch=3)
    continued = scheduled_indices(rows, seed=17, epoch=3, prior_ids=old_ids)
    assert [rows[index].id for index in continued[:2]] == [rows[index].id for index in previous]
    assert {rows[index].id for index in continued[2:]} == {'new-a', 'new-b'}
    assert scheduled_indices(rows, seed=17, epoch=4) == scheduled_indices(rows, seed=17, epoch=4)


def test_row_binding_includes_producer_and_target_mask_semantics():
    row = graph('producer')
    row['nodes'][0]['producing'] = [[1, 3]]
    producer = parse_graph(row, tokenizer_fingerprint='tokenizer-sha')
    altered = graph('producer')
    altered['nodes'][0]['producing'] = [[1, 2]]
    assert _row_bindings([producer]) != _row_bindings([
        parse_graph(altered, tokenizer_fingerprint='tokenizer-sha')])


def test_row_binding_includes_external_state_content_identity():
    first = graph('consumer')
    first['nodes'][0]['target_weights'] = [0, 0, 1]
    first['external_states'] = {'source': {'id': 'nzs1_' + 'a' * 52, 'token_count': 3}}
    first['nodes'][0]['imports'] = [{'source': 'source', 'start': 0, 'stop': 3, 'at': 1, 'logical': False}]
    changed = {**first, 'external_states': {'source': {'id': 'nzs1_' + 'b' * 52, 'token_count': 3}}}
    left = parse_graph(first, tokenizer_fingerprint='tokenizer-sha')
    right = parse_graph(changed, tokenizer_fingerprint='tokenizer-sha')
    assert _row_bindings([left]) != _row_bindings([right])


def test_native_graph_source_validates_then_streams_without_retaining_plans(tmp_path):
    import json
    from natlang_neuralese.train.native_state_sft import NativeGraphSource
    path = tmp_path / 'graphs.jsonl'
    path.write_text(''.join(json.dumps(graph(name)) + '\n' for name in ('a', 'b')))
    source = NativeGraphSource(path, 'tokenizer-sha', required_split='train')
    assert len(source) == 2
    assert source.ordered_ids() == ['a', 'b']
    assert list(source.iter_from(1))[0].id == 'b'
    assert source.receipt()['rows'] == 2
    assert all('plan' not in entry for entry in source.entries)


def test_packed_graph_batch_matches_mean_independent_graph_losses_and_gradients():
    import torch
    from torch import nn
    from natlang_neuralese.data.native_graph import graph_loss
    from natlang_neuralese.train.native_state_sft import packed_graph_loss

    class TinyGraphModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(8, 5)
            self.readout = nn.Linear(5, 8, bias=False)
            self.port = SimpleNamespace(checkpoint_layers=True)

        def forward(self, plan, retain_states=False):
            ids = [token for node in plan.invocations for token in node.token_ids]
            return {'h_final': self.embedding(torch.tensor(ids))[None], 'plan': plan}

        def logits(self, states):
            return self.readout(states)

    a = parse_graph(graph('a', [0, 1, .5]), tokenizer_fingerprint='tokenizer-sha')
    b = parse_graph(graph('b', [0, .25, 1]), tokenizer_fingerprint='tokenizer-sha')
    combined_model, separate_model = TinyGraphModel(), TinyGraphModel()
    separate_model.load_state_dict(combined_model.state_dict())
    expected = torch.stack([graph_loss(separate_model, item)[0] for item in (a, b)]).mean()
    expected.backward()
    actual, _ = packed_graph_loss(combined_model, [a, b], chunk_tokens=2)
    actual.backward()
    torch.testing.assert_close(actual, expected)
    for left, right in zip(combined_model.parameters(), separate_model.parameters()):
        torch.testing.assert_close(left.grad, right.grad)


def test_native_policy_objective_receives_one_forward_and_external_bindings():
    import torch
    from torch import nn
    from natlang_neuralese.train.native_state_sft import packed_graph_loss

    class TinyGraphModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(8, 5)
            self.port = SimpleNamespace(checkpoint_layers=True)
            self.calls = 0

        def forward(self, plan, retain_states=False, **kwargs):
            self.calls += 1
            ids = [token for node in plan.invocations for token in node.token_ids]
            return {'h_final': self.embedding(torch.tensor(ids))[None]}

    class Objective:
        def forward_inputs(self, context):
            assert 'consumer' in context['row_offsets']
            assert context['external_names']['consumer'] == {}
            return {}

        def __call__(self, model, result, context):
            assert model.calls == 1 and context['examples'][0].id == 'consumer'
            return result['h_final'].float().square().mean(), {'kind': 'test-objective'}

    model = TinyGraphModel()
    loss, detail = packed_graph_loss(model,
        [parse_graph(graph('consumer'), tokenizer_fingerprint='tokenizer-sha')], objective=Objective())
    loss.backward()
    assert model.calls == 1 and detail == {'kind': 'test-objective'}
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_tiny_native_sft_update_checkpoint_and_exact_resume(tmp_path):
    import json
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import Lfm2Config, Lfm2ForCausalLM, PreTrainedTokenizerFast
    from natlang_neuralese.data.text_corpus import tokenizer_fingerprint
    from natlang_neuralese.model.shared_state import ROLE_MARKERS
    from natlang_neuralese.train.native_state_sft import main

    base = tmp_path / 'base'
    base.mkdir()
    config = Lfm2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        layer_types=['full_attention'], max_position_embeddings=64)
    torch.manual_seed(19)
    Lfm2ForCausalLM(config).save_pretrained(base)
    tokenizer_object = Tokenizer(WordLevel({f't{i}': i for i in range(32)}, unk_token='t0'))
    tokenizer_object.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer_object, unk_token='t0', bos_token='t1', eos_token='t2')
    tokenizer.add_special_tokens({'additional_special_tokens': list(ROLE_MARKERS)})
    tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}assistant:"
    tokenizer.save_pretrained(base)
    fingerprint = tokenizer_fingerprint(tokenizer)
    row_train = graph('tiny-train', [0, 0, 1])
    row_held = {**graph('tiny-held', [0, 0, 1]), 'split': 'test', 'source_groups': ['held-group']}
    ids = tokenizer.encode('t4 t5 t6', add_special_tokens=False)
    assert len(ids) == 3
    for row in (row_train, row_held):
        row['tokenizer_fingerprint'] = fingerprint
        row['nodes'][0]['token_ids'] = ids
    train_path, held_path = tmp_path / 'train.jsonl', tmp_path / 'held.jsonl'
    train_path.write_text(json.dumps(row_train) + '\n')
    held_path.write_text(json.dumps(row_held) + '\n')

    def run(out, epochs, continue_from=None):
        argv = ['--native-graphs', str(train_path), '--held-native-graphs', str(held_path),
            '--out', str(out), '--base', str(base), '--device', 'cpu', '--rank', '2',
            '--epochs', str(epochs), '--eval-every', '1', '--batch-graphs', '1',
            '--learning-rate', '1e-3', '--weight-decay', '0', '--forward-atol', '1e-4',
            '--gradient-atol', '1e-5', '--seed', '23']
        if continue_from is not None:
            argv += ['--continue-from', str(continue_from)]
        assert main(argv) == 0

    split_out, continuous_out = tmp_path / 'split', tmp_path / 'continuous'
    run(split_out, 1)
    slot = split_out / 'checkpoint.pt'
    interrupted = torch.load(slot, map_location='cpu', weights_only=False)
    assert interrupted['training']['epoch'] == 1
    assert interrupted['training']['global_step'] == 1
    run(split_out, 2, slot)
    run(continuous_out, 2)
    resumed = torch.load(slot, map_location='cpu', weights_only=False)
    continuous = torch.load(continuous_out / 'checkpoint.pt', map_location='cpu', weights_only=False)
    assert resumed['training']['global_step'] == continuous['training']['global_step'] == 2
    assert resumed['training']['exposure_counts'] == continuous['training']['exposure_counts']
    def equal_state(left, right):
        if torch.is_tensor(left):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, dict):
            assert left.keys() == right.keys()
            for key in left:
                equal_state(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            assert len(left) == len(right)
            for a, b in zip(left, right): equal_state(a, b)
        else:
            assert left == right
    for section in ('backbone_state', 'interface_state'):
        equal_state(resumed[section], continuous[section])
    equal_state(resumed['training']['optimizer'], continuous['training']['optimizer'])
    equal_state(resumed['training']['rng'], continuous['training']['rng'])
