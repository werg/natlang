"""Real hybrid/attention model adaptation, control loss and behavioral gradients."""
import copy
import hashlib

import pytest
import torch

from natlang_neuralese.data.native_graph import SCHEMA
from natlang_neuralese.model.shared_state import SharedStateModel
from natlang_neuralese.model.semantic_vae import StateLayout, SemanticStateVAE
from natlang_neuralese.train.native_semantic import (PAIR_SCHEMA, SemanticParticipant,
    parse_pair, producer, consumer_example, joint_loss)
from test_shared_state import tiny


class PayloadTokenizer:
    def __init__(self, ids, offsets):
        self.ids, self.offsets = ids, offsets

    def decode(self, ids, **kwargs):
        return 'abcd' if list(ids) == self.ids else ' '.join('t' + str(i) for i in ids)

    def __call__(self, text, **kwargs):
        class Encoding(dict):
            __getattr__ = dict.__getitem__
        if text == 'abcd':
            return Encoding(input_ids=self.ids, offset_mapping=self.offsets)
        return Encoding(input_ids=list(range(14, 26)), offset_mapping=[])


def setup_pair():
    participants, views = {}, {}
    digest = hashlib.sha256(b'abcd').hexdigest()
    for name, family, ids, offsets, key in [
        ('lfm', 'lfm', [5, 6], [(0, 2), (2, 4)], '1'),
        ('mellum', 'mellum', [8, 9, 10], [(0, 1), (1, 3), (3, 4)], '2')]:
        model = SharedStateModel(tiny(family), marker_ids=(58, 59, 60, 61), rank=4)
        identity = {'checkpoint_sha256': key * 64, 'tokenizer_fingerprint': key * 64,
                    'embedding_weight_sha256': key * 64, 'architecture': model.port.config.model_type}
        participants[name] = SemanticParticipant(model, PayloadTokenizer(ids, offsets),
            {'tokenizer_fingerprint': key * 64}, identity)
        graph = {'schema': SCHEMA, 'id': name + ':paired', 'split': 'train',
            'source_groups': ['source:one'], 'tokenizer_fingerprint': key * 64,
            'nodes': [{'name': 'writer', 'token_ids': [1, 58, *ids, 59],
                       'target_weights': [0., 1., *([1.] * len(ids)), 1.], 'producing': [[2, 2 + len(ids)]]},
                      {'name': 'consumer', 'token_ids': [3, 60, 61, 12, 13],
                       'target_weights': [0., 1., 0., 1., 1.],
                       'imports': [{'at': 2, 'source': 'writer', 'start': 2, 'stop': 2 + len(ids)}]}]}
        views[name] = {'graph': graph, 'export': {'node': 'writer', 'start': 2, 'stop': 2 + len(ids),
            'text': 'abcd', 'offsets': offsets}, 'consumer_nodes': ['consumer'],
            'witness': {'source_id': 'one', 'payload_sha256': digest}}
    row = {'schema': PAIR_SCHEMA, 'id': 'one:payload', 'split': 'train', 'source_groups': ['source:one'],
           'payload_sha256': digest, 'models': views}
    from natlang_neuralese.common.hashing import canonical_json_sha256_hex
    row['communication'] = {'source_id': 'one', 'writer_record_id': 'writer:one', 'reader_record_id': 'reader:one',
        'write_name': 'result', 'occurrence': 'edge:0', 'source_binding': {'manifest_sha256': 'a' * 64}}
    row['communication_sha256'] = canonical_json_sha256_hex(row['communication'])
    for view in views.values():
        view['witness']['communication_sha256'] = row['communication_sha256']
    layouts = {}
    for name, (example, view) in parse_pair(row, participants).items():
        with torch.no_grad():
            block = producer(participants[name], example, view)[0]
        layouts[name] = StateLayout.from_block(block)
    vae = SemanticStateVAE(layouts, latent_width=6, hidden_width=12)
    return participants, row, vae


def test_joint_control_and_consumer_update_both_models_without_producer_replay():
    participants, row, vae = setup_pair()
    counts = dict.fromkeys(participants, 0)
    hooks = []
    for name, participant in participants.items():
        def count(module, args, output, name=name):
            counts[name] += 1
        hooks.append(participant.model.register_forward_hook(count))
    loss, stats = joint_loss(vae, participants, row)
    loss.backward()
    assert all(count == 3 for count in counts.values())  # producer, baseline, adapted consumer
    assert stats['pairs'][0]['control_loss'] > 0 and stats['pairs'][0]['behavior_loss'] > 0
    for participant in participants.values():
        assert participant.model.adapter.up.weight.grad.norm() > 0
        assert participant.model.input_rows.grad.norm() > 0
    assert vae.posterior[-1].weight.grad.norm() > 0
    for hook in hooks:
        hook.remove()


def test_vae_phase_has_no_model_gradient_and_distillation_trains_own_state():
    participants, row, vae = setup_pair()
    loss, _ = joint_loss(vae, participants, row, phase='vae')
    loss.backward()
    assert all(p.grad is None for participant in participants.values() for p in participant.model.parameters())
    assert vae.posterior[-1].weight.grad.norm() > 0
    vae.zero_grad(set_to_none=True)
    loss, stats = joint_loss(vae, participants, row, phase='distill')
    loss.backward()
    assert stats['pairs'][0]['distillation_loss'] >= 0
    assert all(p.grad is None for p in vae.parameters())
    assert all(participant.model.adapter.up.weight.grad.norm() > 0 for participant in participants.values())


def test_consumer_rewrite_removes_unused_writer_but_preserves_other_dependencies():
    participants, row, _ = setup_pair()
    example, view = parse_pair(row, participants)['lfm']
    consumer, wire = consumer_example(example, view, 2)
    assert [node.name for node in consumer.plan.invocations] == ['consumer']
    assert consumer.plan.external_names == (wire,)
    assert consumer.plan.invocations[0].imports[0].source == wire
    broken = copy.deepcopy(view); broken['export']['start'] = 1
    with pytest.raises(ValueError, match='never import'):
        consumer_example(example, broken, 2)


def test_pair_rejects_false_provenance_and_offset_retokenization():
    participants, row, vae = setup_pair()
    bad = copy.deepcopy(row); bad['models']['lfm']['witness']['payload_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='authenticated'):
        parse_pair(bad, participants)
    bad = copy.deepcopy(row); bad['models']['lfm']['export']['offsets'] = [(0, 1), (1, 4)]
    with pytest.raises(ValueError, match='token IDs/offsets'):
        joint_loss(vae, participants, bad)
    example, view = parse_pair(row, participants)['lfm']
    unrelated = copy.deepcopy(view); unrelated['consumer_nodes'].append('writer')
    with pytest.raises(ValueError, match='each declared'):
        consumer_example(example, unrelated, 2)
