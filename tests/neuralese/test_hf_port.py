import pytest
import torch

from natlang_neuralese.model.heads import PortHeads
from natlang_neuralese.model.hf_port import QwenPortBackbone
from natlang_neuralese.model.lfm2_port import ControlTokens, PortCache


def _tiny(tied=True, sliding=None):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    kwargs = {}
    if sliding:
        kwargs = dict(use_sliding_window=True, sliding_window=sliding, max_window_layers=0,
                      layer_types=["sliding_attention", "full_attention", "sliding_attention", "full_attention"])
    config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=8, tie_word_embeddings=tied,
                         max_position_embeddings=128, attn_implementation="eager", **kwargs)
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(config).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@pytest.mark.parametrize("tied,sliding", [(True, None), (False, None), (True, 5)])
def test_matches_hf_and_is_incremental(tied, sliding):
    model = _tiny(tied, sliding)
    port = QwenPortBackbone(model, ControlTokens(open_id=62, close_id=63), fast=False)
    ids = torch.randint(0, 60, (2, 12))
    reference = model(input_ids=ids).logits
    full = port.forward_ids(ids)
    assert torch.allclose(full["logits"][..., :62], reference[..., :62], atol=1e-4)  # 62, 63: marker rows
    split = port.forward_ids(ids, cutoff=2)
    assert torch.allclose(split["h_final"], full["h_final"], atol=1e-5)
    # prefix then one token at a time, through a cache
    out = port.forward_ids(ids[:, :7])
    cache = out["cache"]
    steps = [out["h_final"]]
    for t in range(7, 12):
        out = port.forward_ids(ids[:, t:t + 1], cache=cache)
        cache = out["cache"]
        steps.append(out["h_final"])
    assert torch.allclose(torch.cat(steps, 1), full["h_final"], atol=1e-4)
    port.fast = True
    with torch.no_grad():
        assert torch.allclose(port.forward_ids(ids)["h_final"], full["h_final"], atol=1e-4)


def test_markers_use_their_own_rows():
    model = _tiny(tied=False)
    port = QwenPortBackbone(model, ControlTokens(open_id=62, close_id=63))
    assert not port.tied and port.control_head_rows.shape == (2, 32)
    e = port.embed(torch.tensor([[62, 5]]))
    assert torch.equal(e[0, 0], port.control_rows[0]) and torch.equal(e[0, 1], model.model.embed_tokens.weight[5])
    h = torch.randn(1, 1, 32)
    logits = port.logits(h)
    normed = port.final_norm(h)
    assert torch.allclose(logits[0, 0, 63], normed[0, 0] @ port.control_head_rows[1], atol=1e-5)
    heads = PortHeads(port, cutoff=2, max_length=8)
    assert torch.allclose(heads.feedback.readout.weight, model.lm_head.weight)
    assert torch.equal(heads.feedback.embedding, model.model.embed_tokens.weight)


def test_cache_lengths_per_range():
    model = _tiny()
    port = QwenPortBackbone(model, ControlTokens(open_id=62, close_id=63))
    ids = torch.randint(0, 60, (1, 6))
    h = port.embed(ids)
    h, cache = port.run_layers(h, range(0, 2), PortCache.empty(4))
    assert cache.lengths == (6, 6, 0, 0)


MAPLE_TOKENIZER = "/home/werg/data/models/maple-tokenizer"


@pytest.mark.skipif(not __import__("os").path.isdir(MAPLE_TOKENIZER), reason="Maple tokenizer not present")
def test_maple_renderer_matches_chat_template():
    from transformers import AutoTokenizer

    from natlang_neuralese.data.render import Renderer
    from natlang_neuralese.model.hf_port import qwen_controls

    tokenizer = AutoTokenizer.from_pretrained(MAPLE_TOKENIZER)
    renderer = Renderer(tokenizer, qwen_controls(tokenizer))
    assert renderer.special.bos is None
    messages = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Add 2 and 3."}]
    expected = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + "\n</think>\n\n"
    assert renderer.chat(messages, generation_prompt=True) == tokenizer(expected, add_special_tokens=False).input_ids
    history = messages + [{"role": "assistant", "content": "5"}]
    assert renderer.chat(history) == tokenizer(tokenizer.apply_chat_template(history, tokenize=False),
                                               add_special_tokens=False).input_ids


@pytest.mark.parametrize("sliding", [None, 5])
def test_checkpointed_and_chunked_layers_are_exact(sliding):
    """The LFM2 port's training memory controls on the Qwen/Maple runner: per-layer checkpointing and token-chunked
    feed-forward give the same outputs, caches and gradients as the plain path, from a fresh prefill and after a
    cached prefix."""
    model = _tiny(sliding=sliding)
    port = QwenPortBackbone(model, ControlTokens(open_id=62, close_id=63))
    ids = torch.randint(0, 60, (2, 12))
    with torch.no_grad():
        prefix = port.forward_ids(ids[:, :5])["cache"]

    def run(checkpoint, chunk):
        port.checkpoint_layers, port.ffn_chunk_tokens = checkpoint, chunk
        x = port.embed(ids[:, 5:]).detach().requires_grad_(True)
        h, cache = port.run_layers(x, range(0, port.num_layers), prefix)
        (h.square().sum() + sum(s.k.sum() for s in cache.states)).backward()
        return h.detach(), [s.k.detach() for s in cache.states], cache.lengths, x.grad, port.control_rows.grad

    plain = run(False, 0)
    port.control_rows.grad = None
    for setting in [(True, 0), (False, 3), (True, 3)]:
        other = run(*setting)
        port.control_rows.grad = None
        torch.testing.assert_close(other[0], plain[0])
        for a, b in zip(other[1], plain[1]):
            torch.testing.assert_close(a, b)
        assert other[2] == plain[2]
        torch.testing.assert_close(other[3], plain[3])
