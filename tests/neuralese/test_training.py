import json

import pytest
import torch

from natlang_neuralese.data.fixtures import synthetic_records, write_jsonl
from natlang_neuralese.data.records import RecordError, parse_record, read_records
from natlang_neuralese.data.render import Renderer, SpanExample, render_record, span_examples
from natlang_neuralese.eval.harness import representation_monitors, run_harness
from natlang_neuralese.laws import combine_identity, map_identity, read_map_commutation, split_zip
from natlang_neuralese.train.execution import parallel_write, prefill, supplied_inputs, unroll_write
from natlang_neuralese.train.losses import consumer_loss, distill_loss, span_loss
from natlang_neuralese.train.phases import Phase
from natlang_neuralese.train.trainer import Trainer, trainable_parameters
from natlang_neuralese.write import open_block, write_block

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


def test_trainer_runs_and_resumes(loaded, fresh_heads, spans, renderer, tmp_path):
    _, _, backbone = loaded
    records = [render_record(renderer, parse_record(r)) for r in synthetic_records(4)]
    phases = [Phase("A", 2, batch_size=2), Phase("B", 1, batch_size=2),
              Phase("C", 2, batch_size=1, fraction_end=1.0, ramp_steps=1, unroll_after=1), Phase("D", 1, batch_size=1)]
    rows_before = backbone.control_rows.detach().clone()
    trainer = Trainer(backbone, fresh_heads, phases[:2], tmp_path, span_train=spans, records_train=records, log=lambda *_: None)
    trainer.run()
    assert trainer.global_step == 3 and (tmp_path / "checkpoint.pt").exists()
    lines = [json.loads(l) for l in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert [l["phase"] for l in lines] == ["A", "A", "B"]
    resumed = Trainer(backbone, fresh_heads, phases, tmp_path, span_train=spans, records_train=records, log=lambda *_: None)
    assert resumed.global_step == 3 and resumed.phase_index == 2
    resumed.run()
    assert resumed.global_step == 6
    with torch.no_grad():
        backbone.control_rows.copy_(rows_before)  # leave the shared fixture as it was


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
