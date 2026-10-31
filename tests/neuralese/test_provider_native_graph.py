import hashlib
import json
import copy

import torch
import pytest

from test_native_serving_adapters import engine_fixture, run_native
from natlang_neuralese.serve.engine import GenerationRequest
from natlang_neuralese.data.provider_native_graph import (native_graph_from_luna_capture,
                                                          native_graph_from_luna_episode)
from natlang_neuralese.data.native_graph import graph_loss
from natlang_neuralese.serve.teacher_source_receipt import validate_provider_teacher_source
from natlang_neuralese.data.provider_teacher_source import _rewrite_target_text_spans


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def test_luna_provider_return_result_wrapper_rewrites_to_bound_native_span():
    body = '{"kind":"evidence","lineIds":[292]}'
    wrapped = "<|neuralese|>" + body + "<|/neuralese|>"
    source_turn = {"invocation_id": "case/12", "raw_response_sha256": "a" * 64,
        "assistant": {"calls": [{"tool": "return_result", "arguments": {
            "status": "success", "value": wrapped}}]}}
    target = {"role": "assistant", "content": "", "tool_calls": [{"id": "teacher_12_0",
        "type": "function", "function": {"name": "return_result",
        "arguments": json.dumps({"status": "success", "value": wrapped}, separators=(",", ":"))}}]}
    rewritten, spans, binding = _rewrite_target_text_spans(target, source_turn)
    assert len(spans) == 1
    span = spans[0]
    assert span["path"] == ["tool_calls", 0, "arguments", "value"]
    assert span["text"] == body
    args = json.loads(rewritten["tool_calls"][0]["function"]["arguments"])
    assert args["value"] == span["marker"] and "<|neuralese|>" not in args["value"]
    assert binding["raw_target_sha256"] == _sha(_canonical(target))
    assert binding["rewritten_target_sha256"] == _sha(_canonical(rewritten))
    broken = copy.deepcopy(target)
    broken["tool_calls"][0]["function"]["arguments"] = json.dumps({"status": "success", "value": "different"})
    with pytest.raises(ValueError, match="does not map"):
        _rewrite_target_text_spans(broken, source_turn)


def test_luna_episode_uses_inline_return_span_as_the_authenticated_reader_writer(tmp_path):
    engine = engine_fixture(tmp_path, "lfm")
    def template(messages, tools=None, tokenize=False, add_generation_prompt=False, **kwargs):
        rendered = "".join(f"<{message['role']}>{message.get('content') or ''}" +
            "".join(f"<toolcall>{call.get('function', {}).get('name')}:{call.get('function', {}).get('arguments')}</toolcall>"
                for call in message.get("tool_calls", [])) for message in messages)
        if add_generation_prompt: rendered += "<assistant>"
        return engine.tokenizer(rendered, add_special_tokens=False)["input_ids"] if tokenize else rendered
    engine.tokenizer.apply_chat_template = template
    tools = [{"type": "function", "function": {"name": "return_result", "parameters": {
        "type": "object", "properties": {"status": {"type": "string"}, "value": {}},
        "required": ["status", "value"]}}}]
    body = 't20 t21'
    marker = "NATLANG_INLINE_SPAN_0123456789"
    target = {"role": "assistant", "content": marker}
    spans = [{"path": ["content"], "marker": marker, "text": body}]
    binding = {"body_sha256": _sha(body.encode()), "source_turn_id": "episode/0",
        "source_turn_sha256": "3"*64, "source_response_sha256": "a"*64,
        "raw_target_sha256": _sha(_canonical(target))}
    opening = [{"role": "user", "content": "Write a typed result."}]
    producer_receipt = {"schema": "natlang.provider-teacher-turn-source/1",
        "source_path": "raw/provider.jsonl", "source_file_sha256": "1"*64,
        "source_row_id": "teacher-program:inline-edge", "source_row_sha256": "2"*64,
        "source_turn_index": 0, "source_turn_sha256": "3"*64, "task_sha256": "4"*64,
        "program_sha256": "5"*64, "prompt_sha256": _sha(_canonical({"messages": opening, "tools": tools})),
        "source_prompt_sha256": _sha(_canonical({"messages": opening, "tools": tools})),
        "target_sha256": _sha(_canonical(target)), "raw_target_sha256": "6"*64,
        "target_binding_sha256": "7"*64, "plan_path": "plan.json", "plan_sha256": "8"*64,
        "runtime_path": "runtime", "runtime_receipt_sha256": "9"*64, "split": "train",
        "source_groups": ["loghub-system:HDFS"], "trace_sha256": "a"*64,
        "request_sha256": "b"*64, "response_sha256": "c"*64,
        "transport_provenance_sha256": "d"*64,
        "actor": {"model": "gpt-6-luna", "provider": "openai-codex", "role": "teacher",
                  "transport": "pi-provider"}, "target_supervision": True,
        "typed_text_inputs": [], "typed_text_writes": [],
        "target_text_spans": [{"path": spans[0]["path"], "marker": spans[0]["marker"],
            "body_sha256": binding["body_sha256"], "rewrite_sha256": _sha(_canonical(binding)),
            "source_turn_id": binding["source_turn_id"], "source_turn_sha256": binding["source_turn_sha256"],
            "source_response_sha256": binding["source_response_sha256"],
            "raw_target_sha256": binding["raw_target_sha256"]}]}
    forced = {"schema": "natlang.native-teacher-forced-turn/1", "assistant": target,
        "source_receipt": producer_receipt, "native_text_spans": spans,
        "source_opening": opening, "native_text_bindings": []}
    producer_response = engine.generate(GenerationRequest(messages=opening, tools=tools,
        max_tokens=128, temperature=0.0, seed=21, forced_message=forced,
        rl_capture=True, native_recording=True))
    producer_capture = producer_response["x_natlang_rl_capture"]
    assert len(producer_capture["writes"]) == 1, producer_capture
    native_id = producer_capture["writes"][0]["state_id"]
    native_block = engine.store.get_native_state(native_id)
    assert native_block.block.text == body

    writer_node = "episode/0#24"
    teacher_id = "nzt1_" + "f" * 52
    consumer_body = body
    encoded = engine.native_write_text(consumer_body, value_type="Neuralese<TemplateHypothesis>")
    encoded_record = engine.store.get_native_state(encoded["id"])
    occurrence = "teacher-text-encoder:inline-edge"
    source_opening = [{"role": "user", "content": [
        {"type": "text", "text": "Read: "}, {"type": "teacher_text", "id": teacher_id,
            "source_occurrence": writer_node}]}]
    reader_opening = [{"role": "user", "content": [
        {"type": "text", "text": "Read: "}, {"type": "native_state", "id": encoded["id"],
            "source_occurrence": occurrence}]}]
    reader_target = {"role": "assistant", "content": "t5"}
    writer_receipt = {"block_id": teacher_id, "body_sha256": _sha(body.encode()),
        "source": "return_result", "writer_call_id": "episode/0", "writer_node": writer_node,
        "host_result_receipt_sha256": "e"*64, "raw_response_sha256": "f"*64,
        "context_receipt_sha256": "0"*64, "target_span_path": None, "context_occurrences": 1}
    reader_receipt = {**producer_receipt, "source_turn_index": 1, "source_turn_sha256": "c"*64,
        "prompt_sha256": _sha(_canonical({"messages": reader_opening, "tools": tools})),
        "source_prompt_sha256": _sha(_canonical({"messages": source_opening, "tools": tools})),
        "target_sha256": _sha(_canonical(reader_target)), "typed_text_writes": [writer_receipt],
        "typed_text_inputs": [], "target_text_spans": []}
    reader_binding = {"teacher_block_id": teacher_id, "body_sha256": _sha(body.encode()),
        "native_block_id": encoded["id"], "source_occurrence": occurrence,
        "source_capture": encoded["source_capture"]}
    reader_forced = {"schema": "natlang.native-teacher-forced-turn/1", "assistant": reader_target,
        "source_receipt": reader_receipt, "native_text_spans": [], "source_opening": source_opening,
        "native_text_bindings": [reader_binding]}
    reader_response = engine.generate(GenerationRequest(messages=reader_opening, tools=tools,
        max_tokens=128, temperature=0.0, seed=22, forced_message=reader_forced,
        rl_capture=True, native_recording=True))
    reader_capture = reader_response["x_natlang_rl_capture"]
    reader_capture_bundle = {"source_receipt": reader_receipt, "capture": reader_capture,
        "loaded": {"source_receipt": reader_receipt, "typed_text_bodies": [{"block_id": teacher_id,
            "body": body, "body_sha256": _sha(body.encode())}]}, "response": reader_response,
        "native_identity": engine.native_identity, "encoder_binding": reader_binding,
        "encoder_bindings": [reader_binding], "encoder_state": encoded_record.block,
        "encoder_states": [encoded_record.block], "encoder_occurrence_counts": {teacher_id: 1}}
    producer_bundle = {"source_receipt": producer_receipt, "capture": producer_capture,
        "loaded": {"source_receipt": producer_receipt, "native_text_spans": spans},
        "response": producer_response, "native_identity": engine.native_identity,
        "encoder_binding": None, "encoder_bindings": [], "encoder_state": None, "encoder_states": []}
    graph, example = native_graph_from_luna_episode([producer_bundle, reader_capture_bundle])
    assert graph["teacher_text_inline_producers"][0]["writer_node"] == writer_node
    assert example.plan.invocations[0].name == writer_node
    assert example.plan.invocations[0].producing
    assert example.plan.invocations[1].imports[0].source == writer_node
    assert example.plan.invocations[1].imports[0].stop - example.plan.invocations[1].imports[0].start == len(
        native_block.block.token_ids)


def test_luna_text_source_replays_as_student_encoder_and_full_gradient_graph(tmp_path):
    engine = engine_fixture(tmp_path, "lfm")
    def exact_fixture_template(messages, tools=None, tokenize=False, add_generation_prompt=False, **kwargs):
        rendered = "".join(f"<{message['role']}>{''.join(part.get('text', '') for part in message.get('content', [])) if isinstance(message.get('content'), list) else message.get('content', '')}"
            + "".join(f"<toolcall>{call.get('function', {}).get('name', '')}:{call.get('function', {}).get('arguments', '')}</toolcall>"
                for call in message.get('tool_calls', [])) for message in messages)
        if add_generation_prompt: rendered += "<assistant>"
        if tokenize: return engine.tokenizer(rendered, add_special_tokens=False)["input_ids"]
        return rendered
    engine.tokenizer.apply_chat_template = exact_fixture_template
    bodies = ["t5 t6", "t7 t8"]
    teacher_ids = ["nzt1_" + "a" * 52, "nzt1_" + "b" * 52]
    source_calls = ["teacher-call/1", "teacher-call/2"]
    source_nodes = [call + "#31" for call in source_calls]
    source_opening = [{"role": "user", "content": [
        {"type": "text", "text": "Read the verified brief: "},
        {"type": "teacher_text", "id": teacher_ids[0], "source_occurrence": source_nodes[0]},
        {"type": "text", "text": " Cross-check with "},
        {"type": "teacher_text", "id": teacher_ids[1], "source_occurrence": source_nodes[1]},
        {"type": "text", "text": "; then cite the brief again: "},
        {"type": "teacher_text", "id": teacher_ids[0], "source_occurrence": source_nodes[0]},
        {"type": "text", "text": " and return the next step."}]}]
    target = {"role": "assistant", "content": "t4 t5"}
    tools = []
    bindings, blocks = [], []
    for body, teacher_id in zip(bodies, teacher_ids):
        written = engine.native_write_text(body, value_type="Neuralese<string>")
        block = engine.store.get_native_state(written["id"])
        occurrence = "teacher-text-encoder:fixture-5925:" + teacher_id[-1]
        bindings.append({"teacher_block_id": teacher_id, "body_sha256": _sha(body.encode()),
            "native_block_id": written["id"], "source_occurrence": occurrence,
            "source_capture": written["source_capture"]})
        blocks.append(block)
    opening = [{"role": "user", "content": [
        {"type": "text", "text": "Read the verified brief: "},
        {"type": "native_state", "id": blocks[0].id, "source_occurrence": bindings[0]["source_occurrence"]},
        {"type": "text", "text": " Cross-check with "},
        {"type": "native_state", "id": blocks[1].id, "source_occurrence": bindings[1]["source_occurrence"]},
        {"type": "text", "text": "; then cite the brief again: "},
        {"type": "native_state", "id": blocks[0].id, "source_occurrence": bindings[0]["source_occurrence"]},
        {"type": "text", "text": " and return the next step."}]}]
    receipt = {"schema": "natlang.provider-teacher-turn-source/1",
        "source_path": "bound/raw.jsonl", "source_file_sha256": "1" * 64,
        "source_row_id": "teacher-program:case-5925", "source_row_sha256": "2" * 64,
        "source_turn_index": 5, "source_turn_sha256": "3" * 64,
        "task_sha256": "4" * 64, "program_sha256": "5" * 64,
        "prompt_sha256": _sha(_canonical({"messages": opening, "tools": tools})),
        "source_prompt_sha256": _sha(_canonical({"messages": source_opening, "tools": tools})),
        "target_sha256": _sha(_canonical(target)), "raw_target_sha256": "6" * 64,
        "target_binding_sha256": "7" * 64, "plan_path": "bound/plan.json",
        "plan_sha256": "8" * 64, "runtime_path": "bound/runtime.json",
        "runtime_receipt_sha256": "9" * 64, "split": "train",
        "source_groups": ["sql-db:bird:food_inspection"], "trace_sha256": "a" * 64,
        "request_sha256": "b" * 64, "response_sha256": "c" * 64,
        "transport_provenance_sha256": "d" * 64,
        "actor": {"model": "gpt-6-luna", "provider": "openai-codex",
                  "role": "teacher", "transport": "pi-provider"},
        "target_supervision": True,
        "typed_text_inputs": [],
        "target_text_spans": [],
        "typed_text_writes": [{"block_id": teacher_id, "body_sha256": _sha(body.encode()),
            "source": "eval-finish", "writer_call_id": source_call, "writer_node": source_node,
            "host_result_receipt_sha256": chr(101 + i) * 64,
            "raw_response_sha256": ("f" if i == 0 else "a") * 64,
            "context_receipt_sha256": chr(48 + i) * 64, "target_span_path": None,
            "context_occurrences": 2 if i == 0 else 1}
            for i, (teacher_id, body, source_call, source_node) in enumerate(
                zip(teacher_ids, bodies, source_calls, source_nodes))]}
    forced = {"schema": "natlang.native-teacher-forced-turn/1", "assistant": target,
        "source_receipt": receipt, "native_text_spans": [], "source_opening": source_opening,
        "native_text_bindings": bindings}
    tampered_receipt = copy.deepcopy(receipt)
    tampered_receipt["typed_text_writes"][0]["context_occurrences"] = 3
    with pytest.raises(ValueError, match="occurrence count"):
        validate_provider_teacher_source(tampered_receipt, target, opening, tools, [],
            source_opening=source_opening, native_text_bindings=bindings)
    response = engine.generate(GenerationRequest(messages=opening, tools=tools, max_tokens=128,
        temperature=0.0, seed=31, forced_message=forced, rl_capture=True, native_recording=True))
    capture = response["x_natlang_rl_capture"]
    assert capture["schema"] == "natlang.native-teacher-forced-capture/1"
    bundle = {"source_receipt": receipt, "encoder_bindings": bindings,
        "encoder_states": [block.block for block in blocks],
        "encoder_occurrence_counts": {teacher_ids[0]: 2, teacher_ids[1]: 1}, "capture": capture,
        "loaded": {"source_receipt": receipt}, "native_identity": engine.native_identity}
    row, example = native_graph_from_luna_capture(bundle)
    bad_bundle = copy.deepcopy(bundle)
    removed = False
    for index, segment in enumerate(bad_bundle["capture"]["prompt"]):
        if segment.get("kind") == "native_state" and segment.get("id") == bindings[0]["native_block_id"]:
            del bad_bundle["capture"]["prompt"][index]
            removed = True
            break
    assert removed
    with pytest.raises(ValueError, match="imports do not cover"):
        native_graph_from_luna_capture(bad_bundle)
    assert row["teacher_source_receipt"]["actor"]["model"] == "gpt-6-luna"
    assert len(example.plan.invocations) == 3
    assert [edge.source for edge in example.plan.invocations[2].imports] == [
        bindings[0]["source_occurrence"], bindings[1]["source_occurrence"], bindings[0]["source_occurrence"]]
    assert (example.plan.invocations[2].imports[0].start,
            example.plan.invocations[2].imports[0].stop) == (1, 1 + len(blocks[0].block.token_ids))
    assert sum(example.target_weights) == len(capture["generated_inputs"])

    prior_prompt = [{"role": "user", "content": "Earlier episode context."}]
    prior_receipt = {**receipt, "source_turn_index": 4, "source_turn_sha256": "9" * 64,
        "prompt_sha256": _sha(_canonical({"messages": prior_prompt, "tools": tools})),
        "source_prompt_sha256": _sha(_canonical({"messages": prior_prompt, "tools": tools})),
        "target_sha256": _sha(_canonical(target)), "typed_text_writes": []}
    # This is a unit fixture for episode assembly: the previous capture's target
    # tokens are reused with a plain prompt, never admitted as sampled data.
    prior_capture = copy.deepcopy(capture)
    prior_capture["teacher_source_receipt"] = prior_receipt
    prior_capture["prompt"] = [{"kind": "tokens", "token_ids": [token for segment in capture["prompt"]
        if segment.get("kind") == "tokens" for token in segment["token_ids"]]}]
    prior_bundle = {"source_receipt": prior_receipt,
        "capture": prior_capture, "native_identity": engine.native_identity,
        "encoder_bindings": [], "encoder_states": []}
    episode_row, episode = native_graph_from_luna_episode([bundle, prior_bundle])
    assert [action["source_turn_index"] for action in episode_row["teacher_episode_actions"]] == [4, 5]
    assert len(episode.plan.invocations) == 4
    assert sum(weight > 0 for weight in episode.target_weights) == (
        len(prior_capture["generated_inputs"]) + len(capture["generated_inputs"]))

    model = engine.shared_model
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    with torch.no_grad():
        model.adapter.down.weight.normal_(std=0.02)
    def backward():
        ids = torch.tensor([[token for node in episode.plan.invocations for token in node.token_ids]],
                           device=next(model.parameters()).device)
        embeddings = model.embed(ids).detach().requires_grad_(True)
        result = model(episode.plan, embeddings=embeddings, retain_states=False)
        from natlang_neuralese.data.native_graph import graph_loss_from_result
        loss, detail = graph_loss_from_result(model, episode, result)
        embedding_gradient = torch.autograd.grad(loss, embeddings, retain_graph=True)[0]
        reader_index = len(episode.plan.invocations) - 1
        for index, writer in enumerate(episode.plan.invocations[:-2]):
            writer_offset = episode.plan.offsets[index]
            writer_stop = writer_offset + len(writer.token_ids)
            assert embedding_gradient[:, writer_offset + 1:writer_stop - 1].norm() > 0
        reader_offset = episode.plan.offsets[reader_index]
        assert embedding_gradient[:, reader_offset:].norm() > 0
        loss.backward()
        grads = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        assert detail["supervised_tokens"] == (len(capture["generated_inputs"])
                                                 + len(prior_capture["generated_inputs"]))
        assert any(torch.isfinite(grad).all() and grad.norm() > 0 for grad in grads)
        assert model.adapter.up.weight.grad is not None and model.adapter.up.weight.grad.norm() > 0
        return loss.detach()
    loss = run_native(engine, backward)
    assert torch.isfinite(loss)
