import json

import pytest
import torch

from natlang_neuralese.data.fixtures import synthetic_records, write_jsonl
from natlang_neuralese.data.records import RecordError, parse_record, read_records
from natlang_neuralese.data.render import Renderer, SpanExample, render_record, span_examples
from natlang_neuralese.eval.harness import representation_monitors, run_harness
from natlang_neuralese.laws import combine_identity, map_identity, read_map_commutation, split_zip
from natlang_neuralese.train.execution import one_step_write, parallel_write, prefill, supplied_inputs, unroll_write
from natlang_neuralese.train.losses import consumer_loss, distill_loss, span_loss
from natlang_neuralese.write import open_block, write_block

def trainable_parameters(backbone, heads):
    from natlang_neuralese.train.optim import port_named_parameters
    return [p for _, p in port_named_parameters(backbone, heads)]


ATOL = 2e-4


@pytest.fixture(scope="module")
def renderer(loaded):
    _, tokenizer, backbone = loaded
    return Renderer(tokenizer, backbone.controls)


@pytest.fixture(scope="module")
def spans(renderer):
    text = ("The river rises in the northern hills and flows south through three valleys before it reaches "
            "the coast, where a small harbour town grew up around the old ferry crossing. ") * 4
    return list(span_examples(renderer, [text], prefix_len=12, span_len=5, cont_len=8, limit=4))


@pytest.fixture()
def fresh_heads(loaded):
    from natlang_neuralese.model.heads import PortHeads

    torch.manual_seed(1)
    return PortHeads(loaded[2], cutoff=6, max_length=8)


def test_chat_rendering_matches_template(loaded, renderer):
    _, tokenizer, _ = loaded
    messages = [{"role": "user", "content": "Hello there"}, {"role": "assistant", "content": "Hi"}]
    expected = tokenizer.apply_chat_template(messages, tokenize=False)
    assert renderer.chat(messages) == tokenizer(expected, add_special_tokens=False).input_ids
    prompt = tokenizer.apply_chat_template(messages[:1], tokenize=False, add_generation_prompt=True)
    assert renderer.chat(messages[:1], generation_prompt=True) == tokenizer(prompt, add_special_tokens=False).input_ids


def test_record_validation():
    raw = synthetic_records(4)[0]
    record = parse_record(raw)
    assert record.task == "consume" and record.withheld == ("sources",)
    for broken in ({**raw, "version": "x"}, {**raw, "task": "guess"},
                   {**raw, "writer": {**raw["writer"], "result_type": "string"}}, {**raw, "sources": []}):
        with pytest.raises(RecordError):
            parse_record(broken)


def test_read_records_filters(tmp_path):
    rows = synthetic_records(8)
    rows[0]["outcome"]["label"] = "failed"
    path = write_jsonl(rows + [{"version": "bad"}], tmp_path / "r.jsonl")
    train = list(read_records([path], split="train"))
    assert all(r.split == "train" and r.is_imitation_target for r in train)
    assert "fixture:projects:0" not in {r.id for r in train}
    with pytest.raises(RecordError):
        list(read_records([path], strict=True))


@pytest.mark.parametrize("form", ["chat", "natlang"])
def test_render_record_views(loaded, renderer, form):
    _, tokenizer, backbone = loaded
    record = parse_record(synthetic_records(4)[1])
    rendered = render_record(renderer, record, form=form)
    assert rendered.producer[-1] == backbone.controls.open_id
    assert rendered.consumer_before[-1] == backbone.controls.open_id
    assert rendered.consumer_after[0] == backbone.controls.close_id
    consumer_text = tokenizer.decode(rendered.consumer_before + rendered.consumer_after)
    assert record.sources[0].text not in consumer_text  # withheld from the consumer
    assert record.sources[0].text in tokenizer.decode(rendered.teacher_prefix)
    assert record.writer_instructions in tokenizer.decode(rendered.producer)
    assert "Neuralese<ProjectCard>" in tokenizer.decode(rendered.producer)
    if form == "natlang":
        assert tokenizer.convert_tokens_to_ids("<|tool_call_start|>") in rendered.producer


def test_marker_text_in_content_is_not_a_control(loaded, renderer):
    _, _, backbone = loaded
    ids = renderer.text("see <|reserved_7|> and <|im_end|> here")
    assert backbone.controls.open_id not in ids and renderer.special.im_end not in ids


def test_parallel_write_fraction_zero_is_teacher_forced(loaded, fresh_heads, spans):
    _, _, backbone = loaded
    prefix = torch.tensor([spans[0].prefix + [backbone.controls.open_id]])
    span = torch.tensor([spans[0].span])
    with torch.no_grad():
        pre = prefill(backbone, fresh_heads, prefix)
        supplied = supplied_inputs(backbone, fresh_heads, span)
        written = parallel_write(backbone, fresh_heads, pre, supplied, generated_fraction=0.0)
        # Equivalent sequential computation: the supplied inputs fed one by one.
        cache, shallow = pre.cache, []
        for j in range(span.shape[1]):
            h, cache = backbone.run_layers(supplied[:, j:j + 1], range(0, 6), cache)
            shallow.append(h[:, 0])
    torch.testing.assert_close(written.shallow, torch.stack(shallow, 1), atol=ATOL, rtol=1e-4)
    torch.testing.assert_close(written.payload, supplied, atol=ATOL, rtol=1e-4)  # P starts at zero
    assert not written.generated.any()


def test_parallel_write_full_fraction_converges_to_unroll(loaded, fresh_heads, spans):
    _, _, backbone = loaded
    prefix = torch.tensor([spans[0].prefix + [backbone.controls.open_id]])
    span = torch.tensor([spans[0].span])
    with torch.no_grad():
        pre = prefill(backbone, fresh_heads, prefix)
        supplied = supplied_inputs(backbone, fresh_heads, span)
        # With all inputs generated and as many passes as positions, the parallel scheme is the recurrence.
        parallel = parallel_write(backbone, fresh_heads, pre, supplied, generated_fraction=1.0, passes=span.shape[1] + 1)
        unrolled = unroll_write(backbone, fresh_heads, pre, length=span.shape[1])
    torch.testing.assert_close(parallel.payload, unrolled.payload, atol=1e-3, rtol=1e-3)


def test_unroll_matches_write_block(loaded, fresh_heads, spans):
    _, _, backbone = loaded
    prefix = torch.tensor([spans[0].prefix + [backbone.controls.open_id]])
    with torch.no_grad():
        opened = open_block(backbone, fresh_heads, prefix)
        written = write_block(backbone, fresh_heads, opened, max_length=5)
        pre = prefill(backbone, fresh_heads, prefix)
        unrolled = unroll_write(backbone, fresh_heads, pre, max_length=5)
    assert unrolled.lengths.tolist() == written.lengths.tolist()
    torch.testing.assert_close(unrolled.payload, written.payload, atol=ATOL, rtol=1e-4)
    torch.testing.assert_close(unrolled.stop_logits, written.stop_logits, atol=ATOL, rtol=1e-4)


def test_one_step_write_reproduces_the_rollout_and_targets_the_previous_payload(loaded, fresh_heads, spans):
    _, _, backbone = loaded
    prefix = torch.tensor([spans[0].prefix + [backbone.controls.open_id]])
    pre = prefill(backbone, fresh_heads, prefix)
    with torch.no_grad():
        unrolled = unroll_write(backbone, fresh_heads, pre, length=5)
    written = one_step_write(backbone, fresh_heads, pre, length=5)
    # Forward: the parallel re-run of the greedy rollout is the same write.
    torch.testing.assert_close(written.inputs.detach(), unrolled.inputs, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(written.payload.detach(), unrolled.payload, atol=1e-3, rtol=1e-3)
    assert written.lengths.tolist() == unrolled.lengths.tolist()
    # Self-target: the sketch written from position i (input i + 1) predicts the payload completed at i.
    target = written.sample.mean[:, :-1].detach().float()
    expected = ((written.inputs[:, 1:].float() - target).pow(2).mean(-1) / target.pow(2).mean(-1)).mean()
    torch.testing.assert_close(written.sketch_target_loss, expected, atol=1e-5, rtol=1e-4)
    # One step of gradient reaches the feedback projection.
    fresh_heads.zero_grad()
    written.sketch_target_loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in fresh_heads.feedback.parameters())


def _grads(params):
    return [p.grad for p in params if p.grad is not None and p.grad.abs().sum() > 0]


def test_span_loss_gradients_reach_modules(loaded, fresh_heads, spans):
    _, _, backbone = loaded
    params = trainable_parameters(backbone, fresh_heads)
    for p in params:
        p.grad = None
    loss, metrics = span_loss(backbone, fresh_heads, spans[:2], generated_fraction=0.5,
                              generator=torch.Generator().manual_seed(0), kl_weight=0.5)
    loss.backward()
    assert torch.isfinite(loss) and {"continuation_ce", "entry_ce", "stop_bce", "continuation_kl"} <= metrics.keys()
    assert backbone.control_rows.grad is not None and backbone.control_rows.grad.abs().sum() > 0
    assert fresh_heads.stop.mlp_out.weight.grad.abs().sum() > 0
    assert fresh_heads.content.proj.weight.grad.abs().sum() > 0
    assert fresh_heads.feedback.mlp_out.weight.grad.abs().sum() > 0  # generated inputs train F
    assert not any(p.requires_grad for p in backbone.hf.parameters())


def test_unrolled_span_loss_backpropagates_through_recurrence(loaded, fresh_heads, spans):
    _, _, backbone = loaded
    for p in trainable_parameters(backbone, fresh_heads):
        p.grad = None
    loss, _ = span_loss(backbone, fresh_heads, spans[:1], unroll=True)
    loss.backward()
    assert fresh_heads.feedback.readout.weight.grad.abs().sum() > 0


def test_distill_loss(loaded, fresh_heads, spans):
    _, _, backbone = loaded
    loss, metrics = distill_loss(backbone, fresh_heads, spans[:2])
    loss.backward()
    assert torch.isfinite(loss) and 0 <= metrics["readout_agreement"] <= 1
    assert fresh_heads.feedback.readout.weight.grad.abs().sum() > 0


def test_consumer_loss(loaded, fresh_heads, renderer):
    _, _, backbone = loaded
    rendered = render_record(renderer, parse_record(synthetic_records(4)[2]))
    loss, metrics = consumer_loss(backbone, fresh_heads, rendered, max_length=4)
    loss.backward()
    assert torch.isfinite(loss) and metrics["block_length"] == 4 and metrics["truncated"] == 1.0
    assert fresh_heads.feedback.mlp_out.weight.grad.abs().sum() > 0  # through the sketch recurrence




def test_harness_report(loaded, fresh_heads, spans, renderer, tmp_path):
    _, _, backbone = loaded
    records = [render_record(renderer, parse_record(r)) for r in synthetic_records(4)]
    report = run_harness(backbone, fresh_heads, span_examples=spans[:3], rendered_records=records[:3],
                         cache_prefixes=[spans[0].prefix], latency_prefix=spans[0].prefix, max_length=4,
                         out_path=tmp_path / "report.json", label="test")
    saved = json.loads((tmp_path / "report.json").read_text())
    assert saved["label"] == "test"
    assert {"correct", "shuffled", "zeroed", "no_block", "full_text"} <= report["spans"]["nll"].keys()
    family = report["records"]["families"]["fixture_projects"]
    assert {"correct", "shuffled", "zeroed", "no_block", "full_text"} <= family["nll"].keys()
    assert report["cache_agreement"]["identical_greedy_rate"] == 1.0
    assert report["cache_agreement"]["max_logit_diff"] < 1e-2
    assert report["spans"]["stopping"]["truncation_rate"] == 1.0  # untrained stop head never stops
    assert {"prefix", "shallow_generation", "completion", "projection", "readback"} <= report["latency_seconds"].keys()


def test_representation_monitors_detect_collapse():
    torch.manual_seed(0)
    varied = [torch.randn(4, 16) for _ in range(5)]
    collapsed = [torch.ones(4, 16) + 1e-3 * torch.randn(4, 16) for _ in range(5)]
    a, b = representation_monitors(varied), representation_monitors(collapsed)
    assert b["cosine_to_mean"] > 0.99 > a["cosine_to_mean"]
    assert b["cross_source_similarity"] > a["cross_source_similarity"]


def test_law_hooks_with_stand_ins():
    values = [1, 2, 3]
    read = lambda v: v
    exact_map = lambda v, f: f(v)
    assert map_identity(values, read, exact_map).agreement == 1.0
    assert read_map_commutation(values, read, exact_map, lambda x: x * 2).agreement == 1.0
    lossy_map = lambda v, f: f(v) + (1 if v == 2 else 0)
    assert read_map_commutation(values, read, lossy_map, lambda x: x).agreement == pytest.approx(2 / 3)
    assert combine_identity(values, read, lambda a, b: a + b, lambda: 0).agreement == 1.0
    assert split_zip([(1, 2)], read, lambda a, b: (a, b), lambda z: z).agreement == 1.0


def _write_batch(backbone, heads, spans, rows, **options):
    prefix = torch.tensor([spans[i % len(spans)].prefix + [backbone.controls.open_id] for i in range(rows)])
    with torch.no_grad():
        pre = prefill(backbone, heads, prefix)
        return unroll_write(backbone, heads, pre, max_length=8, sample=True, **options)


def test_sampled_stopping_records_the_behaviour_log_prob_of_the_stop_head(loaded, fresh_heads, spans):
    from natlang_neuralese.train.execution import stop_log_prob

    _, _, backbone = loaded
    with torch.no_grad():
        fresh_heads.stop.mlp_out.bias.fill_(-0.5)  # stop sometimes, so rows differ in length
    written = _write_batch(backbone, fresh_heads, spans, 6, generator=torch.Generator().manual_seed(3))
    # Without exploration the behaviour is the stop head: the importance ratio is exactly one.
    torch.testing.assert_close(written.behavior_log_prob, stop_log_prob(fresh_heads, written).detach(), atol=1e-4, rtol=1e-4)


def test_exploration_spreads_lengths_of_a_never_stopping_head(loaded, fresh_heads, spans):
    _, _, backbone = loaded
    with torch.no_grad():
        fresh_heads.stop.mlp_out.bias.fill_(-20.0)  # the phase-E failure: always continue to the limit
    plain = _write_batch(backbone, fresh_heads, spans, 16, generator=torch.Generator().manual_seed(0))
    assert plain.lengths.tolist() == [8] * 16 and plain.truncated.all()
    explored = _write_batch(backbone, fresh_heads, spans, 16, generator=torch.Generator().manual_seed(0), stop_exploration=1.0)
    assert len(set(explored.lengths.tolist())) >= 4
    # Pure exploration makes every length 1..8 equally likely, whatever the head says.
    torch.testing.assert_close(explored.behavior_log_prob, torch.full((16,), torch.log(torch.tensor(1 / 8.0)).item()))


def test_exploring_phase_e_weights_the_policy_gradient_by_importance(loaded, fresh_heads, renderer):
    from natlang_neuralese.train.losses import consumer_batch_loss

    _, _, backbone = loaded
    with torch.no_grad():
        fresh_heads.stop.mlp_out.bias.fill_(-20.0)
    rendered = [render_record(renderer, parse_record(r)) for r in synthetic_records(4)[:2]]
    for p in trainable_parameters(backbone, fresh_heads):
        p.grad = None
    loss, metrics = consumer_batch_loss(backbone, fresh_heads, rendered, max_length=6, stop_policy_weight=1.0,
                                        policy_samples=3, generator=torch.Generator().manual_seed(1),
                                        stop_exploration=0.5, stop_ratio_clip=5.0)
    loss.backward()
    assert {"stop_ratio_mean", "stop_ratio_clipped", "stop_ratio_ess", "block_length_std"} <= metrics.keys()
    assert metrics["block_length_std"] > 0 and metrics["advantage_abs"] > 0
    assert fresh_heads.stop.mlp_out.weight.grad.abs().sum() > 0






def test_muon_policy_routes_lora_groups_to_adamw(loaded, fresh_heads):
    from natlang_neuralese.train.optim import make_port_optimizer

    _, _, backbone = loaded
    optimizer = make_port_optimizer("muon", backbone, fresh_heads, lr=1e-3)
    groups = len(optimizer.param_groups)
    extra = torch.nn.Parameter(torch.zeros(4, 3))
    optimizer.add_param_group({"params": [extra], "lr": 5e-4, "lr_scale": 0.5, "weight_decay": 0.0})
    assert len(optimizer.param_groups) == groups + 1 and optimizer.param_groups[-1]["lr_scale"] == 0.5
    assert any(extra is p for g in optimizer.auxiliary.param_groups for p in g["params"])
    assert optimizer.schema[-1]["optimizer"] == "adamw"


def _final_heads(loaded, bias):
    from natlang_neuralese.model.heads import PortHeads

    torch.manual_seed(2)
    heads = PortHeads(loaded[2], cutoff=6, max_length=8, stop_source="final")
    with torch.no_grad():
        heads.stop.mlp_out.bias.fill_(bias)
        heads.stop.mlp_out.weight.mul_(50)  # make decisions depend visibly on content
    return heads.eval()


def test_final_stop_lookahead_equals_completing_at_the_chosen_length(loaded, spans):
    _, _, backbone = loaded
    heads = _final_heads(loaded, 0.0)
    assert not heads.stop.use_position
    prefix = torch.tensor([spans[i].prefix + [backbone.controls.open_id] for i in range(3)])
    with torch.no_grad():
        opened = open_block(backbone, heads, prefix)
        result = write_block(backbone, heads, opened, max_length=8, lookahead=3)
        pre = prefill(backbone, heads, prefix)
        for b in range(3):
            n = int(result.lengths[b])
            assert 1 <= n <= 8
            direct = unroll_write(backbone, heads, pre, length=n)
            torch.testing.assert_close(result.payload[b, :n], direct.payload[b, :n], atol=2e-3, rtol=1e-3)
        # Deciding in training along the completed states agrees with the lookahead writer.
        decided = unroll_write(backbone, heads, pre, max_length=8)
        assert decided.lengths.tolist() == result.lengths.tolist()


def test_server_step_writer_lookahead_matches_write_block(loaded, spans):
    from natlang_neuralese.serve.engine import StepWriter

    _, _, backbone = loaded
    heads = _final_heads(loaded, 0.0)
    prefix = torch.tensor([spans[1].prefix + [backbone.controls.open_id]])
    with torch.no_grad():
        opened = open_block(backbone, heads, prefix)
        expected = write_block(backbone, heads, opened, max_length=8, lookahead=4)
        writer = StepWriter(backbone, heads, open_block(backbone, heads, prefix), max_length=8, lookahead=2)
        while not writer.step():
            pass
    assert writer.count == int(expected.lengths[0]) and writer.truncated == bool(expected.truncated[0])


def test_supervised_lengths_teacher_force_and_train_the_boundary(loaded, renderer):
    from natlang_neuralese.train.losses import consumer_batch_loss

    _, _, backbone = loaded
    heads = _final_heads(loaded, -3.0).train()
    rendered = [render_record(renderer, parse_record(r)) for r in synthetic_records(4)[:2]]
    for p in trainable_parameters(backbone, heads):
        p.grad = None
    loss, metrics = consumer_batch_loss(backbone, heads, rendered, target_lengths=[3, 5], stop_weight=1.0)
    loss.backward()
    assert metrics["block_length"] == 4.0 and "stop_bce" in metrics and "stop_length_error" in metrics
    assert heads.stop.mlp_out.weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="exclusive"):
        consumer_batch_loss(backbone, heads, rendered, target_lengths=[3, 5], stop_policy_weight=1.0)




def test_shuffled_payloads_fill_a_width_longer_than_every_row():
    """Pilot v4 failed in phase E: rows of 10 and 20 vectors in a payload padded to 32 (the old repeat count came
    from the longest row, not the width)."""
    from natlang_neuralese.train.losses import shuffled_payloads

    payload = torch.arange(2 * 32 * 3, dtype=torch.float32).reshape(2, 32, 3)
    out = shuffled_payloads(payload, torch.tensor([10, 20]), [1, 0])
    assert out.shape == (2, 32, 3)
    torch.testing.assert_close(out[0, :20], payload[1, :20])
    torch.testing.assert_close(out[0, 20:], payload[1, :12])
    torch.testing.assert_close(out[1, 10:20], payload[0, :10])



def test_text_replay_takes_spans_of_different_lengths(loaded):
    """Phase F's text replay draws spans of several lengths (variable span lengths); pilot v4 failed stacking them.
    The loss is the count-weighted mean of the per-length groups."""
    from types import SimpleNamespace

    from natlang_neuralese.train.losses import replay_loss

    _, _, backbone = loaded
    span = lambda n, k: SimpleNamespace(prefix=list(range(100, 104)), span=list(range(200 + k, 200 + k + n)), continuation=[300, 301])
    batch = [span(8, 0), span(12, 1), span(8, 2)]
    with torch.no_grad():
        loss, metrics = replay_loss(backbone, batch, _nullctx)
        eight, _ = replay_loss(backbone, [batch[0], batch[2]], _nullctx)
        twelve, _ = replay_loss(backbone, [batch[1]], _nullctx)
    assert torch.isfinite(loss)
    torch.testing.assert_close(loss, (2 * eight + twelve) / 3)


class _nullctx:
    def __init__(self, *args):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
