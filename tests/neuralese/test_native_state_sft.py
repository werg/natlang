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


def test_native_graph_index_reuses_validation_and_rebuilds_on_corruption_or_new_source(tmp_path, monkeypatch):
    import json
    import pickle
    from natlang_neuralese.data import native_graph
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE', str(tmp_path / 'cache'))
    path = tmp_path / 'graphs.jsonl'
    path.write_text(json.dumps(graph('a')) + '\n')
    expected = NativeGraphSource(path, 'tokenizer-sha', required_split='train')
    parser = native_graph.parse_graph
    def no_revalidation(*args, **kwargs):
        raise AssertionError('unchanged validated graph index should be reused')
    with monkeypatch.context() as patch:
        patch.setattr(native_graph, 'parse_graph', no_revalidation)
        cached = NativeGraphSource(path, 'tokenizer-sha', required_split='train')
        assert cached.entries == expected.entries and cached.receipt() == expected.receipt()
    cache_file, = (tmp_path / 'cache').glob('native-graphs-*.pkl')
    with cache_file.open('rb') as stream:
        broken = pickle.load(stream)
    broken['entries'][0]['supervised_tokens'] = 1000000
    cache_file.write_bytes(pickle.dumps(broken))
    rebuilt = NativeGraphSource(path, 'tokenizer-sha', required_split='train')
    assert rebuilt.entries == expected.entries
    path.write_text(path.read_text() + json.dumps(graph('b')) + '\n')
    expanded = NativeGraphSource(path, 'tokenizer-sha', required_split='train')
    assert len(expanded) == 2 and list(expanded.iter_from(1))[0].id == 'b'
    assert native_graph.parse_graph is parser


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

        def forward(self, plan, retain_states=False, **kwargs):
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


def test_real_stop_checkpoints_before_held_eval_and_resume_logs_actual_updates(tmp_path, monkeypatch):
    import json
    import math
    import torch
    from test_native_model import tiny_source
    from natlang_neuralese.model.native_model import load_native_model
    from natlang_neuralese.train import native_state_sft as trainer

    base = tmp_path / 'base'
    tiny_source(base)
    _, _, _, config = load_native_model(base, dtype=torch.float32, rank=4)
    paths = {}
    for split in ('train', 'test'):
        row = graph(split)
        row.update(split=split, source_groups=[split], tokenizer_fingerprint=config['tokenizer_fingerprint'])
        paths[split] = tmp_path / (split + '.jsonl')
        paths[split].write_text(json.dumps(row) + '\n')
    argv = ['--base', str(base), '--dtype', 'float32', '--rank', '4', '--device', 'cpu',
            '--native-graphs', str(paths['train']), '--held-native-graphs', str(paths['test']),
            '--epochs', '3', '--eval-every', '100', '--out', str(tmp_path / 'run')]
    def stop_after_update(policy, step, **kwargs):
        assert step == 1
        policy.stop.requested = True
        return False
    with monkeypatch.context() as patch:
        patch.setattr(trainer.CheckpointPolicy, 'due', stop_after_update)
        def no_held_before_save(*args, **kwargs):
            raise AssertionError('operational stop must save without starting held evaluation')
        patch.setattr(trainer, '_evaluate', no_held_before_save)
        assert trainer.main(argv) == 0
    checkpoint = torch.load(tmp_path / 'run/checkpoint.pt', weights_only=False)
    assert checkpoint['training']['step'] == 1
    assert checkpoint['training']['exposure_counts'] == {'train': 1}
    assert trainer.main(argv) == 0
    rows = [json.loads(line) for line in (tmp_path / 'run/train.jsonl').read_text().splitlines()]
    assert [row['step'] for row in rows] == [1, 2, 3]
    assert all(math.isfinite(row['loss']) and row['seconds'] > 0 for row in rows)
    assert all(row['forward_tokens'] == 3 and row['supervised_tokens'] == 2 for row in rows)
    assert json.loads((tmp_path / 'run/progress.json').read_text()) == rows[-1]
    # A completed three-epoch run must give an appended row all three visits,
    # without replaying the already-complete prefix or resetting its moments.
    appended = graph('new-train')
    appended.update(source_groups=['new-train'], tokenizer_fingerprint=config['tokenizer_fingerprint'])
    union = tmp_path / 'train-union.jsonl'
    union.write_text(paths['train'].read_text() + json.dumps(appended) + '\n')
    argv[argv.index('--native-graphs') + 1] = str(union)
    with pytest.raises(ValueError, match='continuation predecessor'):
        trainer.main([*argv, '--continued-native-graphs', str(paths['test'])])
    argv += ['--continued-native-graphs', str(paths['train'])]
    assert trainer.main(argv) == 0
    checkpoint = torch.load(tmp_path / 'run/checkpoint.pt', weights_only=False)
    assert checkpoint['training']['step'] == 6
    assert checkpoint['training']['exposure_counts'] == {'train': 3, 'new-train': 3}
    assert checkpoint['training']['optimizer']
    assert json.loads((tmp_path / 'run/report.json').read_text())['completed'] is True


def test_remaining_exposure_covers_partial_and_fractional_append_without_replay():
    from natlang_neuralese.train.native_state_sft import pending_exposure_batches
    entries = [{'id': str(index), 'forward_tokens': 3} for index in range(4)]
    quotas, batches = pending_exposure_batches(entries, {'0': 2, '1': 1}, epochs=1.5, batch_tokens=6)
    assert quotas == {'0': 2, '1': 2, '2': 1, '3': 1}
    assert batches == [(0, [2, 3], True), (1, [1], True)]


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

    def run(out, epochs, continue_from=None, batch_tokens=0, weight_decay=0, latent_ns='quintic'):
        argv = ['--native-graphs', str(train_path), '--held-native-graphs', str(held_path),
            '--out', str(out), '--base', str(base), '--device', 'cpu', '--rank', '2',
            '--epochs', str(epochs), '--eval-every', '1', '--batch-graphs', '1',
            '--dtype', 'float32', '--learning-rate', '1e-3', '--weight-decay', str(weight_decay), '--batch-tokens', str(batch_tokens), '--forward-atol', '1e-4',
            '--gradient-atol', '1e-5', '--seed', '23', '--latent-ns', latent_ns]
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
    run(split_out, 3, slot, batch_tokens=8, weight_decay=.001, latent_ns='cubic5')
    adapted = torch.load(slot, map_location='cpu', weights_only=False)
    assert adapted['training']['global_step'] == 3
    assert adapted['training']['identity']['batch_tokens'] == 8
    assert adapted['training']['identity']['optimizer_options']['latent_ns'] == 'cubic5'
    assert any(group.get('weight_decay') == .001 for section in adapted['training']['optimizer'].values()
               if isinstance(section, dict) for group in section.get('param_groups', []))


def test_token_volume_batches_preserve_every_graph_and_long_graph():
    from natlang_neuralese.train.native_state_sft import batch_boundaries
    entries = [{'forward_tokens': count} for count in (3, 4, 20, 2, 5)]
    assert batch_boundaries(entries, batch_tokens=8) == [(0, 2), (2, 3), (3, 5)]
    assert batch_boundaries(entries, batch_graphs=2) == [(0, 2), (2, 4), (4, 5)]
    assert batch_boundaries([], batch_tokens=8) == []


def authenticated_pair(identifier='pair', *, roles=(58, 59, 60, 61), fingerprint='1'*64):
    import copy
    from natlang_neuralese.common.hashing import canonical_json_sha256_hex, sha256_hex
    from natlang_neuralese.train.native_semantic import PAIR_SCHEMA
    po, pc, co, cc = roles
    models = {}
    for name, ids, offsets, fp in [('lfm', [5, 6], [[0, 2], [2, 4]], fingerprint),
                                   ('mellum', [7, 8, 9], [[0, 1], [1, 3], [3, 4]], '2'*64)]:
        models[name] = {'graph': {'schema': SCHEMA, 'id': name+':'+identifier, 'split': 'train',
            'source_groups': ['paired:'+identifier], 'tokenizer_fingerprint': fp, 'nodes': [
                {'name': 'producer', 'token_ids': [1, po, *ids, pc],
                 'target_weights': [0.]*(len(ids)+3), 'producing': [[2, 2+len(ids)]]},
                {'name': 'consumer', 'token_ids': [3, co, cc, 12, 13],
                 'target_weights': [0., 0., 0., 1., 1.],
                 'imports': [{'at': 2, 'source': 'producer', 'start': 2, 'stop': 2+len(ids)}]}]},
            'export': {'node': 'producer', 'start': 2, 'stop': 2+len(ids), 'text': 'abcd', 'offsets': offsets},
            'consumer_nodes': ['consumer'], 'witness': {'source_id': identifier,
                'payload_sha256': sha256_hex(b'abcd')}}
    for view in models.values():
        view['graph']['source_pairing'] = {'consumer_source_row_id':'reader:'+identifier,
            'consumer_original_text_sha256':'b'*64,'consumed_occurrences':[{
                'source_row_id':'reader:'+identifier,'payload_sha256':sha256_hex(b'abcd'),
                'wrapped_token_start':2,'wrapped_token_stop':view['export']['stop'],
                'consumer_import_at':2}]}
    communication = {'source_id': identifier, 'writer_record_id': 'writer:'+identifier,
        'reader_record_id': 'reader:'+identifier, 'write_name': 'value', 'occurrence': 'read:0',
        'source_binding': {'manifest_sha256': 'a'*64}}
    row = {'schema': PAIR_SCHEMA, 'id': identifier, 'split': 'train',
        'source_groups': ['paired:'+identifier], 'models': models, 'communication': communication,
        'communication_sha256': canonical_json_sha256_hex(communication), 'payload_sha256': sha256_hex(b'abcd')}
    for view in models.values():
        view['witness']['communication_sha256'] = row['communication_sha256']
    return copy.deepcopy(row)


def source_manifest(tmp_path, ordinary, pairs, *, model='lfm', name='source.json'):
    import json
    from natlang_neuralese.common.hashing import sha256_file_hex
    ordinary_path, pairs_path = tmp_path/'ordinary.jsonl', tmp_path/'pairs.jsonl'
    ordinary_path.write_text(''.join(json.dumps(row)+'\n' for row in ordinary))
    pairs_path.write_text(''.join(json.dumps(row)+'\n' for row in pairs))
    manifest = {'schema': 'natlang.native-graph-sources/1', 'sources': [
        {'kind': 'native-graphs', 'path': ordinary_path.name, 'sha256': sha256_file_hex(ordinary_path)},
        {'kind': 'semantic-pairs', 'path': pairs_path.name, 'sha256': sha256_file_hex(pairs_path), 'model': model}]}
    path = tmp_path/name
    path.write_text(json.dumps(manifest))
    return path


def test_pair_source_prioritizes_actual_graphs_and_preserves_requested_stream_order(tmp_path):
    from natlang_neuralese.train.native_state_sft import pending_exposure_batches, communication_exposure
    ordinary = [dict(graph(name), tokenizer_fingerprint='1'*64) for name in ('a', 'b', 'c')]
    pair = authenticated_pair()
    manifest = source_manifest(tmp_path, ordinary, [pair])
    source = NativeGraphSource(manifest, '1'*64, required_split='train', role_ids=(58,59,60,61))
    assert source.ordered_ids() == ['a', 'b', 'c', 'lfm:pair']
    assert source.bindings['a'] == _row_bindings([parse_graph(ordinary[0], tokenizer_fingerprint='1'*64)])['a']
    quotas, batches = pending_exposure_batches(source.entries, {'a': 1}, epochs=1, batch_graphs=1)
    assert [indices for _, indices, _ in batches] == [[3], [1], [2]]
    assert [item.id for item in source.iter_indices([3,1,2])] == ['lfm:pair','b','c']
    assert [item.id for item in source.iter_indices([2,0,3,1])] == ['c','a','lfm:pair','b']
    assert communication_exposure(source.entries, {'a':1}, quotas) == {
        'graphs':1,'required_graph_visits':1,'graph_visits':0,'complete':False}
    assert communication_exposure(source.entries, {'lfm:pair':1}, quotas)['complete']
    # Fractional ordinary exposure never omits an admitted communication graph.
    quotas, _ = pending_exposure_batches(source.entries, {}, epochs=.25)
    assert quotas == {'a':1,'b':0,'c':0,'lfm:pair':1}
    assert source.receipt()['communication_graphs'] == 1


def test_pair_source_rejects_plaintext_bypass_split_or_witness_tampering(tmp_path):
    import copy
    ordinary = [dict(graph('ordinary'), tokenizer_fingerprint='1'*64)]
    original = authenticated_pair()
    broken = []
    row = copy.deepcopy(original)
    row['models']['lfm']['graph']['nodes'][1]['token_ids'] = [3,60,5,6,61,12,13]
    row['models']['lfm']['graph']['nodes'][1]['target_weights'] = [0,0,0,0,0,1,1]
    broken.append((row, 'consuming boundary'))
    row = copy.deepcopy(original);row['models']['lfm']['witness']['payload_sha256']='0'*64
    broken.append((row, 'payload witness'))
    row = copy.deepcopy(original);row['models']['lfm']['graph']['split']='test'
    broken.append((row, 'split/source groups'))
    row = copy.deepcopy(original);row['models']['lfm']['graph']['nodes'][1]['target_weights']=[0]*5
    broken.append((row, 'supervised consumer'))
    row = copy.deepcopy(original);row['models']['lfm']['graph'].pop('source_pairing')
    broken.append((row, 'occurrence provenance'))
    for row, error in broken:
        manifest = source_manifest(tmp_path, ordinary, [row])
        with pytest.raises(ValueError, match=error):
            NativeGraphSource(manifest, '1'*64, required_split='train', role_ids=(58,59,60,61))


@pytest.mark.parametrize('family', ['lfm','mellum'])
def test_each_paired_model_trains_consumer_to_producer_through_one_flat_forward(tmp_path, family):
    import torch
    from test_shared_state import tiny
    from natlang_neuralese.model.shared_state import SharedStateModel, native_step, capture_invocation_span, splice_block
    from natlang_neuralese.train.native_state_sft import packed_graph_loss
    pair = authenticated_pair()
    fingerprint = pair['models'][family]['graph']['tokenizer_fingerprint']
    ordinary = [dict(graph('ordinary'), tokenizer_fingerprint=fingerprint)]
    source = NativeGraphSource(source_manifest(tmp_path, ordinary, [pair], model=family), fingerprint,
                               required_split='train', role_ids=(58,59,60,61))
    example = next(source.iter_indices([1]))
    model = SharedStateModel(tiny(family), (58,59,60,61), rank=4)
    model.port.checkpoint_layers = False
    count = []
    hook = model.register_forward_hook(lambda *args: count.append(1))
    loss, _ = packed_graph_loss(model, [example], checkpoint_layers=False)
    gradient = torch.autograd.grad(loss, model.adapter.up.weight)[0]
    assert count == [1] and gradient.norm() > 0
    hook.remove()
    actual = model(example.plan)
    expected = model(example.plan, reference=True)
    torch.testing.assert_close(actual['h_final'], expected['h_final'], atol=3e-6, rtol=2e-5)
    # The consumer has only five own queries; payload aliases add cache slots,
    # never plaintext payload query rows. Check source captured IDs as well.
    assert len(example.plan.invocations[1].token_ids) == 5
    exported = pair['models'][family]['export']
    assert capture_invocation_span(actual, 'producer', exported['start'], exported['stop'])['token_ids'].tolist() == list(
        example.plan.invocations[0].token_ids[exported['start']:exported['stop']])
    from natlang_neuralese.model.state_block import NativeStateBlock
    source_node, consumer_node = example.plan.invocations
    produce_mask = torch.zeros(1,len(source_node.token_ids),dtype=torch.bool)
    produce_mask[:,exported['start']:exported['stop']] = True
    producer = native_step(model,torch.tensor([source_node.token_ids]),producing=produce_mask)
    start,stop = exported['start'],exported['stop']
    identity = {'checkpoint_sha256':'4'*64,'tokenizer_fingerprint':fingerprint,'embedding_weight_sha256':'3'*64}
    block = NativeStateBlock(text='abcd',architecture=model.port.config.model_type,**identity,
        token_ids=producer['token_ids'][start:stop],embeddings=producer['embeddings'][start:stop],
        positions=producer['positions'][start:stop],layers=tuple({key:(value[start:stop] if key!='kind' else value)
            for key,value in layer.items()} for layer in producer['layers'])).validate()
    edge = consumer_node.imports[0]
    prefix = native_step(model,torch.tensor([consumer_node.token_ids[:edge.at]]))
    cache = splice_block(model,prefix['cache'],block,**identity)
    suffix = native_step(model,torch.tensor([consumer_node.token_ids[edge.at:]]),cache)
    offset = example.plan.offsets[1]+edge.at
    torch.testing.assert_close(suffix['h_final'],actual['h_final'][:,offset:],atol=3e-6,rtol=2e-5)
    cached_loss = torch.nn.functional.cross_entropy(model.logits(suffix['h_final'][:,:-1])[0].float(),
                                                   torch.tensor(consumer_node.token_ids[edge.at+1:]))
    torch.testing.assert_close(cached_loss,loss,atol=3e-6,rtol=2e-5)
    cached_gradient = torch.autograd.grad(cached_loss,model.adapter.up.weight)[0]
    torch.testing.assert_close(cached_gradient,gradient,atol=3e-6,rtol=2e-5)


def test_native_manifest_hash_and_recipe_dependencies_bind_every_input(tmp_path):
    from natlang_neuralese.train.recipe import native_source_dependencies, require_input_decoders, _inputs_without
    from natlang_neuralese.common.hashing import sha256_file_hex
    ordinary = [dict(graph('ordinary'), tokenizer_fingerprint='1'*64)]
    manifest = source_manifest(tmp_path, ordinary, [authenticated_pair()])
    dependencies = native_source_dependencies(manifest)
    assert {row['path'] for row in dependencies} == {str(tmp_path/'ordinary.jsonl'),str(tmp_path/'pairs.jsonl')}
    value = {'path':str(manifest),'sha256':sha256_file_hex(manifest),'dependencies':dependencies}
    identity = {item['path']:item['sha256'] for item in [value,*dependencies]}
    assert _inputs_without(identity, {'s':{'native_graphs':value}}, {'s':{'old':{'native_graphs':value}}}, 'old') == {}
    (tmp_path/'pairs.jsonl').write_text('{}\n')
    with pytest.raises(ValueError, match='content hash mismatch'):
        native_source_dependencies(manifest)
    with pytest.raises(ValueError, match='SHA-256 mismatch'):
        NativeGraphSource(manifest, '1'*64, required_split='train', role_ids=(58,59,60,61))


def test_priority_deficits_repack_resume_and_append_have_exact_row_credits():
    from natlang_neuralese.train.native_state_sft import pending_exposure_batches, communication_exposure
    entries = [{'id':name,'forward_tokens':size,'communication':comm}
               for name,size,comm in [('a',3,False),('b',4,False),('pair1',11,True),('pair2',9,True)]]
    credits = {'a':1,'pair1':1}
    quotas,batches = pending_exposure_batches(entries,credits,epochs=2,batch_tokens=12)
    order = [entries[index]['id'] for _,indices,_ in batches for index in indices]
    assert order[:3] == ['pair2','pair1','pair2']
    # Stop after one committed batch, repack, and append a new genuine pair.
    for index in batches[0][1]:credits[entries[index]['id']]=credits.get(entries[index]['id'],0)+1
    entries.append({'id':'pair3','forward_tokens':8,'communication':True})
    quotas,resumed = pending_exposure_batches(entries,credits,epochs=2,batch_graphs=3)
    expected = {entry['id']:quotas[entry['id']]-credits.get(entry['id'],0) for entry in entries}
    observed = dict.fromkeys(expected,0)
    for _,indices,_ in resumed:
        for index in indices:observed[entries[index]['id']]+=1
    assert observed == expected
    for name,count in observed.items():credits[name]=credits.get(name,0)+count
    assert communication_exposure(entries,credits,quotas) == {
        'graphs':3,'required_graph_visits':6,'graph_visits':6,'complete':True}
    assert pending_exposure_batches(entries,credits,epochs=2)[1] == []


def test_real_resume_appends_pair_ahead_of_backlog_without_losing_optimizer_rng_or_credits(tmp_path, monkeypatch):
    import json
    import torch
    from test_native_model import tiny_source
    from natlang_neuralese.model.native_model import load_native_model
    from natlang_neuralese.train import native_state_sft as trainer
    base = tmp_path/'base'
    tiny_source(base)
    _,_,roles,config = load_native_model(base,dtype=torch.float32,rank=4)
    fp = config['tokenizer_fingerprint']
    ordinary = [dict(graph(name),tokenizer_fingerprint=fp) for name in ('a','b')]
    old = tmp_path/'ordinary.jsonl'
    old.write_text(''.join(json.dumps(row)+'\n' for row in ordinary))
    held = tmp_path/'held.jsonl'
    held.write_text(json.dumps(dict(graph('held'),split='test',tokenizer_fingerprint=fp))+'\n')
    out = tmp_path/'run'
    argv = ['--base',str(base),'--dtype','float32','--rank','4','--device','cpu',
        '--native-graphs',str(old),'--held-native-graphs',str(held),'--epochs','2',
        '--eval-every','100','--out',str(out),'--forward-atol','1e-4','--gradient-atol','1e-5',
        '--held-eval-tokens','4']
    def stop(policy,step,**kwargs):
        policy.stop.requested=True
        return False
    with monkeypatch.context() as patch:
        patch.setattr(trainer.CheckpointPolicy,'due',stop)
        assert trainer.main(argv)==0
    initial = torch.load(out/'checkpoint.pt',weights_only=False)
    assert initial['training']['exposure_counts']=={'a':1,'b':0}
    manifest = source_manifest(tmp_path,ordinary,[authenticated_pair('new',roles=roles.ids,fingerprint=fp)])
    continued = argv.copy()
    continued[continued.index('--native-graphs')+1] = str(manifest)
    continued += ['--continued-native-graphs',str(old),'--batch-tokens','32']
    with monkeypatch.context() as patch:
        patch.setattr(trainer.CheckpointPolicy,'due',stop)
        assert trainer.main(continued)==0
    checkpoint = torch.load(out/'checkpoint.pt',weights_only=False)
    assert checkpoint['training']['exposure_counts']=={'a':1,'b':0,'lfm:new':1}
    assert checkpoint['training']['global_step']==2
    assert checkpoint['training']['identity']['schedule']=='row-visit-deficit/2'
    assert checkpoint['training']['rng'].keys()==initial['training']['rng'].keys()
    partial = json.loads((out/'report.json').read_text())['communication_exposure']
    assert partial=={'graphs':1,'required_graph_visits':2,'graph_visits':1,'complete':False}
    # Same immutable predecessor remains declared on subsequent restarts; the
    # trainer remembers its actual adoption, rather than rejecting its own new slot.
    continued[continued.index('--batch-tokens')+1]='0'
    continued += ['--batch-graphs','2']
    assert trainer.main(continued)==0
    final = torch.load(out/'checkpoint.pt',weights_only=False)
    assert final['training']['exposure_counts']=={'a':2,'b':2,'lfm:new':2}
    assert final['training']['metrics']['held']['scope']=='full'
    assert json.loads((out/'report.json').read_text())['communication_exposure']['complete']
    frozen_rng = final['training']['rng']['torch_rng'].clone()
    assert trainer.main(continued)==0
    unchanged = torch.load(out/'checkpoint.pt',weights_only=False)
    assert torch.equal(unchanged['training']['rng']['torch_rng'],frozen_rng)
    assert unchanged['training']['global_step']==final['training']['global_step']


def test_rotating_held_cadence_preserves_whole_long_graphs_and_final_coverage(tmp_path, monkeypatch):
    import torch
    from natlang_neuralese.train import native_state_sft as trainer
    entries=[{'forward_tokens':size} for size in (3,4,20,2)]
    first,cursor=trainer.held_cadence_indices(entries,token_budget=8)
    assert (first,cursor)==([0,1],2)
    assert trainer.held_cadence_indices(entries,cursor,token_budget=8)==([2],3)
    assert trainer.held_cadence_indices(entries,3,token_budget=8)==([3,0],1)
    assert trainer.held_cadence_indices(entries,2,graph_budget=2)==([2,3],0)
    class Source:
        file_sha256='1'*64
        entries=[{'forward_tokens':size} for size in (3,4,20,2)]
        def __len__(self):return len(self.entries)
        def iter_indices(self,indices):return iter(indices)
    seen=[]
    def evaluate(model,rows,**kwargs):
        selected=list(rows);seen.append(selected)
        return {'loss':1.,'examples':len(selected),'forward_tokens':sum(entries[i]['forward_tokens'] for i in selected)}
    monkeypatch.setattr(trainer,'_evaluate',evaluate)
    args=SimpleNamespace(held_eval_tokens=8,held_eval_graphs=0,checkpoint_layers=False,chunk_tokens=32)
    _,coverage=trainer._evaluate_cadence(None,Source(),args=args)
    result,coverage=trainer._evaluate_cadence(None,Source(),args=args,coverage=coverage)
    assert result['scope']=='cadence-subset' and coverage['cursor']==3
    result,coverage=trainer._evaluate_cadence(None,Source(),args=args,coverage=coverage,final=True)
    assert result['scope']=='full' and seen[-1]==[0,1,2,3] and coverage['cursor']==0


def test_exact_previous_ordinary_index_migration_never_admits_changed_source_or_pair(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    from natlang_neuralese.common import row_index
    from natlang_neuralese.common.hashing import canonical_json_sha256_hex, sha256_file_hex_cached
    from natlang_neuralese.train import native_state_sft as trainer
    from natlang_neuralese.data import native_graph
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE',str(tmp_path/'cache'))
    path=tmp_path/'ordinary.jsonl';row=graph('a')
    path.write_text(json.dumps(row)+'\n')
    package=Path(trainer.__file__).resolve().parents[1]
    policy={name:sha256_file_hex_cached(package/name) for name in (
        'data/native_graph.py','model/shared_state.py','common/row_index.py','common/jsonio.py','common/hashing.py')}
    policy['train/native_state_sft.py']='d8f3d6413a40f26389c90b6e5c3cc50781e3fdae70942de241301fd678327c9d'
    key={'source_sha256':sha256_file_hex_cached(path),'tokenizer':'tokenizer-sha','split':'train',
         'policy_sha256':canonical_json_sha256_hex(policy)}
    entry={'id':'a','groups':['source-a'],'binding':_row_bindings([
        parse_graph(row,tokenizer_fingerprint='tokenizer-sha')])['a'],'supervised_tokens':2,'forward_tokens':3}
    row_index.save(row_index.entry_path('native-graphs',canonical_json_sha256_hex(key)),
                   {'key':key,'entries':[entry],'metadata_sha256':canonical_json_sha256_hex([entry])})
    def forbid(*args,**kwargs):raise AssertionError('exact known admission index should be reused')
    with monkeypatch.context() as patch:
        patch.setattr(native_graph,'parse_graph',forbid)
        source=NativeGraphSource(path,'tokenizer-sha',required_split='train')
    assert source.entries==[entry] and next(iter(source)).id=='a'
    row['nodes'][0]['target_weights']=[0,.5,1]
    path.write_text(json.dumps(row)+'\n')
    changed=NativeGraphSource(path,'tokenizer-sha',required_split='train')
    assert changed.bindings['a'] != entry['binding']
