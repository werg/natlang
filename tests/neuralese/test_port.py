"""The model's LFM backbone path remains numerically aligned with its HF checkpoint."""

import torch


ATOL = 2e-4


def text_ids(tokenizer, text):
    return tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids


def without_controls(backbone, logits):
    keep = torch.ones(logits.shape[-1], dtype=torch.bool)
    keep[[backbone.controls.open_id, backbone.controls.close_id]] = False
    return logits[..., keep]


@torch.no_grad()
def test_forward_matches_hf(loaded):
    model, tokenizer, backbone = loaded
    ids = text_ids(tokenizer, "Short convolutions and attention layers alternate in this model.")
    ours = backbone.forward_ids(ids)["logits"]
    theirs = model(input_ids=ids).logits
    torch.testing.assert_close(without_controls(backbone, ours), without_controls(backbone, theirs), atol=ATOL, rtol=1e-4)


@torch.no_grad()
def test_chunked_prefill_matches_full(loaded):
    _, tokenizer, backbone = loaded
    ids = text_ids(tokenizer, "One two three four five six seven eight nine ten eleven twelve.")
    full = backbone.forward_ids(ids)["logits"]
    cache, pieces = None, []
    for chunk in (ids[:, :5], ids[:, 5:6], ids[:, 6:]):
        out = backbone.forward_ids(chunk, cache=cache)
        cache = out["cache"]
        pieces.append(out["logits"])
    torch.testing.assert_close(torch.cat(pieces, 1), full, atol=ATOL, rtol=1e-4)
    assert set(cache.lengths) == {ids.shape[1]}


@torch.no_grad()
def test_layer_ranges_compose(loaded):
    _, tokenizer, backbone = loaded
    ids = text_ids(tokenizer, "Layer ranges compose into the full stack.")
    full = backbone.forward_ids(ids)
    split = backbone.forward_ids(ids, cutoff=6)
    torch.testing.assert_close(split["logits"], full["logits"], atol=ATOL, rtol=1e-4)
    assert split["h_cut"].shape == (1, ids.shape[1], backbone.config.hidden_size)


def test_cache_selection_preserves_the_selected_rows():
    from natlang_neuralese.model.lfm2_port import AttentionState, ConvState, DeltaState, PortCache

    cache = PortCache((AttentionState(torch.arange(12).reshape(3, 1, 4), torch.arange(12, 24).reshape(3, 1, 4)),
                       ConvState(torch.arange(6).reshape(3, 2)),
                       DeltaState(torch.arange(6).reshape(3, 2), torch.arange(9).reshape(3, 3))),
                      (3, 3, 3))
    selected = cache.select([2, 0])
    assert selected.lengths == (3, 3, 3)
    torch.testing.assert_close(selected.states[0].k, cache.states[0].k[[2, 0]])
    torch.testing.assert_close(selected.states[1].window, cache.states[1].window[[2, 0]])
    torch.testing.assert_close(selected.states[2].recurrent, cache.states[2].recurrent[[2, 0]])
