"""Raw profile: token-preserving encode/read/write and the real gradient replay."""
import pytest
import torch

from natlang_neuralese.model.heads import PortHeads
from natlang_neuralese.read import build_inputs, read_forward, splice
from natlang_neuralese.serve.engine import Engine, GenerationRequest
from natlang_neuralese.serve.grad import GradSession, embed_text, encode_text
from natlang_neuralese.serve.store import TensorStore
from natlang_neuralese.train.execution import consumer_forward, consumer_forward_batch
from natlang_neuralese.write import greedy_continue, open_block, write_block, read_back
from types import SimpleNamespace


@pytest.fixture(scope='module')
def raw(loaded):
    _, tokenizer, base = loaded
    heads = PortHeads(base, cutoff=base.num_layers, max_length=16,
                      profile='raw-token-v1', stop_source='shallow', stop_position=False).eval()
    with torch.no_grad():
        heads.stop.mlp_out.weight.zero_()
        heads.stop.mlp_out.bias.fill_(-100.)
    return Engine(base, heads, tokenizer, TensorStore(), 'nd:natlang-raw-token@1', max_block=16)


@torch.no_grad()
def test_encode_and_profile_aware_splice_are_identity(raw):
    text = 'The clinic opens on Monday and Wednesday.'
    encoded, embedded = encode_text(raw, text), embed_text(raw, text)
    assert torch.equal(encoded.payload, embedded.payload)
    ids = raw._tokens(text)
    payload = raw.backbone.embed(torch.tensor([ids[2:5]]))[0]
    inputs = build_inputs(raw.backbone, [[ids[:2], payload, ids[5:]]], heads=raw.heads)
    ordinary = raw.backbone.embed(torch.tensor([ids]))
    assert torch.equal(splice(raw.backbone, raw.heads, inputs), ordinary)
    assert torch.equal(read_forward(raw.backbone, raw.heads, inputs)['logits'], raw.backbone.forward_ids(torch.tensor([ids]))['logits'])
    wrong = build_inputs(raw.backbone, [[ids[:2], payload, ids[5:]]])
    with pytest.raises(ValueError, match='profile mismatch'):
        splice(raw.backbone, raw.heads, wrong)


@torch.no_grad()
def test_training_consumer_uses_the_same_raw_transport(raw):
    ids = raw._tokens('The answer is Paris. It is in France.')
    before, source, after, target = ids[:2], ids[2:4], ids[4:-3], ids[-3:]
    payload = raw.backbone.embed(torch.tensor([source]))
    reference = raw.backbone.forward_ids(torch.tensor([ids[:-1]]))['logits'][:, -len(target):]
    marker_before = before + [raw.backbone.controls.open_id]
    marker_after = [raw.backbone.controls.close_id] + after
    single = consumer_forward(raw.backbone, raw.heads, marker_before, payload, marker_after, target)
    assert torch.equal(single, reference)
    row = SimpleNamespace(consumer_before=marker_before, consumer_after=marker_after, target=target)
    batch = consumer_forward_batch(raw.backbone, raw.heads, [row], payload, torch.tensor([len(source)]))[0]
    torch.testing.assert_close(batch, reference[0], atol=2e-4, rtol=1e-4)


@torch.no_grad()
def test_reference_write_matches_ordinary_greedy_tokens_and_readback(raw):
    ids = raw._tokens('A concise answer to the question is')
    ordinary = raw.backbone.forward_ids(torch.tensor([ids]))
    expected, _ = greedy_continue(raw.backbone, ordinary['cache'], ordinary['logits'][:, -1], 5)
    opened = open_block(raw.backbone, raw.heads, torch.tensor([ids + [raw.backbone.controls.open_id]]))
    result = write_block(raw.backbone, raw.heads, opened, max_length=5)
    assert torch.equal(result.payload, raw.backbone.embed(torch.tensor([expected])))
    readback = read_back(raw.backbone, raw.heads, opened.cache, result.payload)
    reference = raw.backbone.forward_ids(torch.tensor([ids + expected]))
    torch.testing.assert_close(readback['logits'], reference['logits'][:, -1], atol=2e-4, rtol=1e-4)


@torch.no_grad()
def test_server_forced_write_is_the_same_greedy_reference(raw):
    messages = [{'role': 'user', 'content': 'Name the capital of France.'}]
    ordinary = raw.backbone.forward_embeds(raw.prompt_embeddings(messages, None))
    expected, _ = greedy_continue(raw.backbone, ordinary['cache'], ordinary['logits'][:, -1], 4)
    response = raw.generate(GenerationRequest(messages=messages, forced=[{'neuralese': 'write'}],
                                              neuralese_length=4, max_tokens=16))
    block = raw.lookup(response['neuralese']['blocks'][0]['id'])
    assert torch.equal(block.payload, raw.backbone.embed(torch.tensor([expected]))[0])


def test_gradient_replay_matches_serving_transport_and_has_finite_input_gradient(raw):
    block = embed_text(raw, 'a useful geography hint')
    messages = [{'role': 'user', 'content': [{'type': 'text', 'text': 'Hint: '},
                 {'type': 'neuralese', 'id': block.id}, {'type': 'text', 'text': '\nCapital of France?'}]}]
    session = GradSession(raw)
    # The real term renderer/replay is tested; no test-only read mode override.
    result = session.run({'arguments': [block.id], 'terms': [{'kind': 'crossEntropy', 'messages': messages,
                                                           'target': {'role': 'assistant', 'content': 'Paris'}}]})
    gradient = raw.store.get(result['gradients'][block.id]).payload
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    assert result['loss'] >= 0
