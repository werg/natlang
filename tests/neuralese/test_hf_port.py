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
