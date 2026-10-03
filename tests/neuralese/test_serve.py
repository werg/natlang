"""Reference server: content IDs, the step writer, chat rendering/parsing, generation with blocks, HTTP."""

import json
import threading
import urllib.error
import urllib.request

import pytest
import torch

from natlang_neuralese.serve.chat import RequestError, build_message, parse_pythonic_calls, placeholder, render_messages
from natlang_neuralese.serve.store import TensorStore, content_id, decode_block, encode_block, make_block

DIALECT = "nd:natlang@1"


def test_content_ids_match_the_runtime():
    # Computed by ts-host's neuraleseContentId for the same block.
    data = torch.tensor([[1.0, 2.5, -3.0], [0.25, 0.0, 7.0]])
    block = make_block(data, DIALECT)
    assert block.id == "nz1_ci7j22ryfvfvhqfdnrjebcodnnkg7xl47ohdz5yawcti2e6cp7la"
    assert content_id(DIALECT, 2, 3, "f32", data.numpy().tobytes()) == block.id


def test_block_wire_format_round_trips_and_checks_ids():
    block = make_block(torch.randn(4, 8), DIALECT, type="Neuralese<string>", producer={"kind": "test"}, truncated=True)
    back = decode_block(encode_block(block))
    assert back.id == block.id and back.type == block.type and back.truncated and back.producer == {"kind": "test"}
    assert torch.equal(back.payload, block.payload)
    tampered = bytearray(encode_block(block))
    tampered[-1] ^= 1
    with pytest.raises(ValueError, match="does not match"):
        decode_block(bytes(tampered))


def test_store_pins_and_collects():
    store = TensorStore()
    a, b = make_block(torch.ones(1, 2), DIALECT), make_block(torch.zeros(1, 2), DIALECT)
    store.put(a), store.put(b)
    store.pin(b.id)
    assert store.collect(set()) == [a.id]
    store.unpin(b.id)
    assert store.collect(set()) == [b.id]


def _template(messages, tools):
    out = []
    for m in messages:
        calls = "".join(f"<|tool_call_start|>[{c['function']['name']}("
                        + ", ".join(f"{k}='{v}'" for k, v in c['function']['arguments'].items()) + ")]<|tool_call_end|>"
                        for c in m.get("tool_calls") or [])
        out.append(f"<|im_start|>{m['role']}\n{m.get('content') or ''}{calls}<|im_end|>\n")
    return "".join(out) + "<|im_start|>assistant\n"


def test_rendering_cuts_blocks_out_of_content_and_tool_arguments():
    a, b = "nz1_" + "a" * 52, "nz1_" + "b" * 52
    arguments = [{"type": "text", "text": '{"code": "const p: Neuralese<Plan> = '}, {"type": "neuralese", "id": b},
                 {"type": "text", "text": '; return p;"}'}]
    rendered = render_messages([
        {"role": "user", "content": [{"type": "text", "text": "memo: "}, {"type": "neuralese", "id": a}]},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "1", "type": "function",
                                                              "function": {"name": "eval", "arguments": arguments}}]},
    ], None, _template)
    assert rendered.blocks == [a, b]
    assert [s for s in rendered.segments if isinstance(s, int)] == [0, 1]
    text = "".join(s if isinstance(s, str) else "#" for s in rendered.segments)
    assert "memo: #" in text and "eval(code='const p: Neuralese<Plan> = #; return p;')" in text


def test_parsing_pythonic_calls_restores_blocks_as_parts():
    block = "nz1_" + "c" * 52
    text = ("Writing it.<|tool_call_start|>[eval(code='const note: Neuralese<string> = "
            + placeholder(0) + ";\\nreturn note;')]<|tool_call_end|><|im_end|>")
    message = build_message(text, [block])
    assert message["content"] == "Writing it."
    (call,) = message["tool_calls"]
    assert call["function"]["name"] == "eval"
    code = json.loads(call["function"]["arguments"])["code"]
    assert code == [{"type": "text", "text": "const note: Neuralese<string> = "}, {"type": "neuralese", "id": block},
                    {"type": "text", "text": ";\nreturn note;"}]
    assert parse_pythonic_calls("[return_result(status='success', value=3)]") == [
        ("return_result", {"status": "success", "value": 3})]


@pytest.fixture(scope="module")
def engine(loaded):
    from natlang_neuralese.model.heads import PortHeads
    from natlang_neuralese.serve.engine import Engine

    _, tokenizer, backbone = loaded
    torch.manual_seed(1)
    heads = PortHeads(backbone, cutoff=6, max_length=8).eval()
    return Engine(backbone, heads, tokenizer, TensorStore(), DIALECT, max_block=4)


def test_step_writer_matches_write_block(loaded, heads):
    from natlang_neuralese.serve.engine import StepWriter
    from natlang_neuralese.write import open_block, write_block

    _, tokenizer, backbone = loaded
    ids = tokenizer("Summarise the ticket:", add_special_tokens=False)["input_ids"] + [backbone.controls.open_id]
    with torch.no_grad():
        opened = open_block(backbone, heads, torch.tensor([ids]))
        reference = write_block(backbone, heads, opened, max_length=5)
        writer = StepWriter(backbone, heads, opened, 5)
        while not writer.step():
            pass
        payload, _, _ = writer.complete(0.0, None)
    assert writer.count == int(reference.lengths[0]) and writer.truncated == bool(reference.truncated[0])
    assert torch.allclose(payload[0], reference.row(0))
    for temperature in (0.5,):
        with torch.no_grad():
            sampled = write_block(backbone, heads, opened, max_length=5, temperature=temperature,
                                  generator=torch.Generator().manual_seed(7))
            ours, _, _ = writer.complete(temperature, torch.Generator().manual_seed(7))
        assert torch.allclose(ours[0], sampled.row(0))


FORCED = ["<|tool_call_start|>[eval(code='const note: Neuralese<string> = ", {"neuralese": "write"},
          ";\\nreturn note;')]<|tool_call_end|>"]


def test_generation_writes_a_block_inside_eval_code_and_reads_it_back(engine):
    from natlang_neuralese.serve.engine import GenerationRequest

    tools = [{"type": "function", "function": {"name": "eval", "parameters": {"type": "object", "properties": {
        "code": {"type": "string"}}}}}]
    messages = [{"role": "user", "content": "Write a note."}]
    response = engine.generate(GenerationRequest(messages=messages, tools=tools, forced=FORCED,
                                                 neuralese_temperature=0.5, seed=3))
    message = response["choices"][0]["message"]
    assert response["choices"][0]["finish_reason"] == "tool_calls"
    code = json.loads(message["tool_calls"][0]["function"]["arguments"])["code"]
    assert [p["type"] for p in code] == ["text", "neuralese", "text"]
    (meta,) = response["neuralese"]["blocks"]
    block = engine.store.get(code[1]["id"])
    assert meta["id"] == block.id and block.length == meta["length"] and meta["length"] <= 4
    record = block.producer
    assert record["kind"] == "write" and record["temperature"] == 0.5 and isinstance(record["seed"], int)
    assert engine.store.get(record["mean"]).length == block.length
    assert engine.store.get(record["log_sigma"]).length == block.length
    # The same seed reproduces the same block.
    again = engine.generate(GenerationRequest(messages=messages, tools=tools, forced=FORCED, neuralese_temperature=0.5,
                                              seed=3))
    assert again["neuralese"]["blocks"][0]["id"] == block.id

    # The block travels back in the conversation and is read through the read port.
    follow = messages + [message, {"role": "tool", "content": [{"type": "text", "text": "stored "},
                                                                {"type": "neuralese", "id": block.id}]}]
    reply = engine.generate(GenerationRequest(messages=follow, tools=tools, max_tokens=3))
    assert reply["usage"]["prompt_tokens"] > response["usage"]["prompt_tokens"]
    unknown = [{"role": "user", "content": [{"type": "neuralese", "id": "nz1_" + "z" * 52}]}]
    with pytest.raises(RequestError, match="neuralese-unknown-block"):
        engine.generate(GenerationRequest(messages=unknown, max_tokens=1))
    foreign = engine.store.put(make_block(torch.ones(2, engine.width), "nd:other@1"))
    with pytest.raises(RequestError, match="neuralese-dialect-mismatch"):
        engine.generate(GenerationRequest(messages=[{"role": "user", "content": [{"type": "neuralese", "id": foreign.id}]}],
                                          max_tokens=1))


def test_http_endpoints_and_interleaved_requests(engine):
    from natlang_neuralese.serve.http import serve

    engine.start()
    server = serve(engine)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def call(method, path, body=None, raw=False):
        data = body if isinstance(body, bytes) or body is None else json.dumps(body).encode()
        request = urllib.request.Request(base + path, data=data, method=method)
        with urllib.request.urlopen(request) as response:
            payload = response.read()
            return payload if raw else json.loads(payload)

    try:
        info = call("GET", "/v1/neuralese/info")
        assert info["dialects"] == [DIALECT] and info["grad"] is False and info["width"] == engine.width
        block = make_block(torch.randn(3, engine.width), DIALECT, type="Neuralese<string>")
        assert call("PUT", f"/v1/neuralese/blocks/{block.id}", encode_block(block))["id"] == block.id
        assert decode_block(call("GET", f"/v1/neuralese/blocks/{block.id}", raw=True)).id == block.id
        assert call("GET", f"/v1/neuralese/blocks/{block.id}/meta")["length"] == 3
        with pytest.raises(urllib.error.HTTPError) as error:
            call("PUT", f"/v1/neuralese/blocks/nz1_{'a' * 52}", encode_block(block))
        assert error.value.code == 400
        with pytest.raises(urllib.error.HTTPError) as error:
            call("POST", "/v1/neuralese/grad", {})
        assert error.value.code == 501
        # Two requests in flight at once: one writes a block while the other decodes text.
        results = {}

        def run(name, body):
            results[name] = call("POST", "/v1/chat/completions", body)

        bodies = {
            "writer": {"messages": [{"role": "user", "content": "go"}], "x_natlang_forced": FORCED, "seed": 1},
            "reader": {"messages": [{"role": "user", "content": [{"type": "text", "text": "read "},
                                                                {"type": "neuralese", "id": block.id}]}],
                       "max_tokens": 6},
        }
        threads = [threading.Thread(target=run, args=item) for item in bodies.items()]
        [t.start() for t in threads]
        [t.join(timeout=300) for t in threads]
        assert results["writer"]["neuralese"]["blocks"][0]["length"] >= 1
        assert results["reader"]["usage"]["completion_tokens"] >= 1
    finally:
        server.shutdown()
        engine.stop()
