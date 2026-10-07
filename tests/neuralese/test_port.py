import copy

import pytest
import torch

from natlang_neuralese.model.dialect import DIALECT, StoredBlock, block_id
from natlang_neuralese.model.lfm2_port import CLOSE_TOKEN, OPEN_TOKEN, PortCache
from natlang_neuralese.read import build_inputs, read_forward, splice
from natlang_neuralese.write import greedy_continue, open_block, read_back, write_block

ATOL = 2e-4


def text_ids(tokenizer, text):
    return tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids


def without_controls(backbone, logits):
    """Logits for ordinary tokens; the two control columns are trainable rows that differ from the base."""
    keep = torch.ones(logits.shape[-1], dtype=torch.bool)
    keep[[backbone.controls.open_id, backbone.controls.close_id]] = False
    return logits[..., keep]


def prefix(tokenizer, backbone, text="The committee met on Tuesday and decided"):
    ids = text_ids(tokenizer, text)
    open_id = torch.tensor([[backbone.controls.open_id]])
    return torch.cat([ids, open_id], 1)


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


@torch.no_grad()
def test_right_padding_matches_unpadded(loaded, heads):
    _, tokenizer, backbone = loaded
    a = text_ids(tokenizer, "A short row.")[0].tolist()
    b = text_ids(tokenizer, "A noticeably longer row with more tokens in it.")[0].tolist()
    payload = torch.randn(3, backbone.config.hidden_size)
    batch = build_inputs(backbone, [[a, payload, a], [b]])
    out = read_forward(backbone, heads, batch)["logits"]
    for row, segments in enumerate([[a, payload, a], [b]]):
        single = build_inputs(backbone, [segments])
        alone = read_forward(backbone, heads, single)["logits"]
        n = single.ids.shape[1]
        torch.testing.assert_close(out[row, :n], alone[0], atol=ATOL, rtol=1e-4)


@torch.no_grad()
def test_splice_positions(loaded, heads):
    _, tokenizer, backbone = loaded
    before = text_ids(tokenizer, "Notes:")[0].tolist()
    after = text_ids(tokenizer, " Summary:")[0].tolist()
    payload = torch.randn(4, backbone.config.hidden_size)
    inputs = build_inputs(backbone, [[before, payload, after]])
    ids = inputs.ids[0]
    n = len(before)
    assert ids[n] == backbone.controls.open_id
    assert ids[n + 5] == backbone.controls.close_id
    assert inputs.payload_mask[0].nonzero().squeeze(-1).tolist() == list(range(n + 1, n + 5))
    embeds = splice(backbone, heads, inputs)
    torch.testing.assert_close(embeds[0, n + 1: n + 5], heads.interface(payload))
    torch.testing.assert_close(embeds[0, :n], backbone.embed(ids[None, :n])[0])
    torch.testing.assert_close(embeds[0, n], backbone.control_rows[0])


@torch.no_grad()
def test_hf_inputs_embeds_agrees(loaded, heads):
    model, tokenizer, backbone = loaded
    payload = torch.randn(3, backbone.config.hidden_size)
    inputs = build_inputs(backbone, [[text_ids(tokenizer, "Read this:")[0].tolist(), payload,
                                      text_ids(tokenizer, " Done.")[0].tolist()]])
    embeds = splice(backbone, heads, inputs)
    ours = backbone.forward_embeds(embeds)["logits"]
    theirs = model(inputs_embeds=embeds).logits
    torch.testing.assert_close(without_controls(backbone, ours), without_controls(backbone, theirs), atol=ATOL, rtol=1e-4)


def test_control_tokens(loaded):
    _, tokenizer, backbone = loaded
    assert tokenizer.convert_ids_to_tokens([backbone.controls.open_id, backbone.controls.close_id]) == [OPEN_TOKEN, CLOSE_TOKEN]
    rendered = backbone.controls.render("x <|neuralese|> y <|/neuralese|>")
    assert rendered == f"x {OPEN_TOKEN} y {CLOSE_TOKEN}"
    # Marker text in ordinary content is not a control token.
    plain = tokenizer("<|neuralese|>", add_special_tokens=False).input_ids
    assert backbone.controls.open_id not in plain


@torch.no_grad()
def test_write_shapes_and_determinism(loaded, heads, device):
    _, tokenizer, backbone = loaded
    opened = open_block(backbone, heads, prefix(tokenizer, backbone).repeat(2, 1))
    first = write_block(backbone, heads, opened, max_length=5)
    second = write_block(backbone, heads, opened, max_length=5)
    d = backbone.config.hidden_size
    assert first.payload.shape == (2, 5, d)
    assert first.sketches.shape == first.shallow.shape == first.final.shape == (2, 5, d)
    torch.testing.assert_close(first.payload, second.payload, rtol=0, atol=0)
    assert first.lengths.tolist() == second.lengths.tolist()
    g1, g2 = torch.Generator(device=device).manual_seed(7), torch.Generator(device=device).manual_seed(7)
    s1 = write_block(backbone, heads, opened, max_length=5, sample=True, generator=g1)
    s2 = write_block(backbone, heads, opened, max_length=5, sample=True, generator=g2)
    assert s1.lengths.tolist() == s2.lengths.tolist()
    torch.testing.assert_close(s1.payload, s2.payload, rtol=0, atol=0)
    # Content projection starts at zero: the first payload equals the sketch.
    torch.testing.assert_close(first.payload, first.sketches)


@torch.no_grad()
def test_block_start_cache_is_a_snapshot(loaded, heads):
    _, tokenizer, backbone = loaded
    ids = prefix(tokenizer, backbone)
    opened = open_block(backbone, heads, ids)
    before = [s.k.clone() if hasattr(s, "k") else s.window.clone() for s in opened.cache.states]
    result = write_block(backbone, heads, opened, max_length=4)
    assert result.block_start is opened.cache
    assert set(opened.cache.lengths) == {ids.shape[1]}
    after = [s.k if hasattr(s, "k") else s.window for s in opened.cache.states]
    for x, y in zip(before, after):
        torch.testing.assert_close(x, y, rtol=0, atol=0)


@torch.no_grad()
def test_stop_masking_and_hard_maximum(loaded, heads):
    _, tokenizer, backbone = loaded
    opened = open_block(backbone, heads, prefix(tokenizer, backbone))
    # Initial bias -3: greedy never stops, so the hard maximum truncates.
    capped = write_block(backbone, heads, opened, max_length=3)
    assert capped.lengths.tolist() == [3] and capped.truncated.tolist() == [True]
    eager = copy.deepcopy(heads)
    torch.nn.init.constant_(eager.stop.mlp_out.bias, 20.0)
    # Stop is masked before the first vector, so an eager stop head still writes one.
    one = write_block(backbone, eager, opened, max_length=3)
    assert one.lengths.tolist() == [1] and one.truncated.tolist() == [False]
    empty = write_block(backbone, eager, opened, max_length=3, allow_empty=True)
    assert empty.lengths.tolist() == [0]


@torch.no_grad()
def test_prefix_payload_property(loaded, heads):
    _, tokenizer, backbone = loaded
    opened = open_block(backbone, heads, prefix(tokenizer, backbone))
    result = write_block(backbone, heads, opened, max_length=6)
    final, _ = backbone.run_layers(result.shallow[:, :3], range(heads.cutoff, backbone.num_layers), opened.cache)
    torch.testing.assert_close(heads.content(result.sketches[:, :3], final), result.payload[:, :3], atol=ATOL, rtol=1e-4)


@torch.no_grad()
def test_cached_matches_recomputed(loaded, heads):
    _, tokenizer, backbone = loaded
    ids = prefix(tokenizer, backbone)
    opened = open_block(backbone, heads, ids)
    result = write_block(backbone, heads, opened, max_length=4)
    payload = result.row(0)[None]
    back = read_back(backbone, heads, result.block_start, payload)
    tokens, cached_logits = greedy_continue(backbone, back["cache"], back["logits"], steps=6)

    # Recompute the whole committed sequence in one pass, without any cache.
    close = torch.tensor([[backbone.controls.close_id]])
    sequence = torch.cat([
        backbone.embed(ids), heads.interface(payload), backbone.embed(close),
        backbone.embed(torch.tensor([tokens])),
    ], 1)
    full = backbone.forward_embeds(sequence)["logits"][0]
    start = ids.shape[1] + payload.shape[1]  # position of the close marker
    recomputed = full[start: start + len(tokens) + 1]
    torch.testing.assert_close(torch.cat(cached_logits, 0), recomputed, atol=1e-3, rtol=1e-4)
    assert recomputed[:-1].argmax(-1).tolist() == tokens
    # The readback cache is aligned with the committed sequence: no position is counted twice.
    assert set(back["cache"].lengths) == {ids.shape[1] + payload.shape[1] + 1}


def test_lora_merge(loaded, tmp_path):
    peft = pytest.importorskip("peft")
    from natlang_neuralese.model.lfm2_port import load_backbone

    model, tokenizer, _ = loaded
    base = copy.deepcopy(model)
    config = peft.LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj", "w1"])
    adapted = peft.get_peft_model(base, config)
    torch.manual_seed(3)
    for name, parameter in adapted.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(parameter, std=0.02)
    adapted.save_pretrained(tmp_path)
    ids = text_ids(tokenizer, "Merged adapters change the base.")
    with torch.no_grad():
        expected = adapted(input_ids=ids).logits
        merged, _ = load_backbone(lora=str(tmp_path), dtype=torch.float32)
        got = merged(input_ids=ids).logits
        plain = model(input_ids=ids).logits
    torch.testing.assert_close(got, expected, atol=ATOL, rtol=1e-4)
    assert (got - plain).abs().max() > 1e-3
    assert not any(p.requires_grad for p in merged.parameters())


def test_block_ids_and_dialect():
    payload = torch.arange(12, dtype=torch.float32).view(3, 4)
    a = block_id(DIALECT, payload)
    assert a == block_id(DIALECT, payload.clone()) and a.startswith("nz1_")
    assert a != block_id("nd:natlang@2", payload)
    assert block_id(DIALECT, payload.to(torch.bfloat16)) != a
    block = StoredBlock.make(payload, "Neuralese<Plan>", truncated=False)
    assert block.length == 3
    block.check_dialect(DIALECT)
    with pytest.raises(ValueError):
        block.check_dialect("nd:natlang@2")


def test_cache_select():
    states = (None, None)
    cache = PortCache(states, (0, 0))
    assert cache.select([0]).lengths == (0, 0)
