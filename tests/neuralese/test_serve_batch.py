"""Reference server: escaping of control-token text in content, batched steps, streaming."""

import json
import threading
import urllib.request

import pytest
import torch

from natlang_neuralese.serve.chat import render_messages
from natlang_neuralese.serve.store import TensorStore

DIALECT = "nd:natlang@1"


@pytest.fixture(scope="module")
def engine(loaded):
    from natlang_neuralese.model.heads import PortHeads
    from natlang_neuralese.serve.engine import Engine

    _, tokenizer, backbone = loaded
    torch.manual_seed(1)
    heads = PortHeads(backbone, cutoff=6, max_length=8).eval()
    return Engine(backbone, heads, tokenizer, TensorStore(), DIALECT, max_block=4)


def test_control_token_text_in_content_is_plain_text(engine):
    open_id = engine.backbone.controls.open_id
    for marker in ("<|reserved_7|>", "<|neuralese|>", "<|im_end|>"):
        messages = [{"role": "user", "content": f"quote {marker} here"}]
        rendered = render_messages(messages, None, engine._template, engine.specials)
        embeds_ids = [i for seg in rendered.segments for i in engine._template_tokens(seg)]
        special = engine.tokenizer.convert_tokens_to_ids(marker) if marker != "<|neuralese|>" else open_id
        im_end = engine.tokenizer.convert_tokens_to_ids("<|im_end|>")
        # The template's own <|im_end|> stays special; the quoted marker does not.
        if marker == "<|im_end|>":
            assert embeds_ids.count(im_end) == 1
        else:
            assert special not in embeds_ids
    # Inside tool-call arguments too.
    calls = [{"role": "assistant", "content": None, "tool_calls": [{"type": "function", "function": {
        "name": "eval", "arguments": json.dumps({"code": "s = '<|reserved_7|>'"})}}]}]
    rendered = render_messages(calls, None, engine._template, engine.specials)
    ids = [i for seg in rendered.segments if isinstance(seg, str) for i in engine._template_tokens(seg)]
    assert open_id not in ids
    assert engine.tokenizer.convert_tokens_to_ids("<|tool_call_start|>") in ids


def test_batched_steps_match_single_rows(loaded):
    from natlang_neuralese.serve.batch import step_rows

    _, tokenizer, backbone = loaded
    prompts = ["The quick brown fox", "A", "Natural language functions are typed and"]
    caches, nexts = [], []
    with torch.no_grad():
        for text in prompts:
            ids = torch.tensor([tokenizer(text, add_special_tokens=False)["input_ids"]])
            out = backbone.forward_ids(ids)
            caches.append(out["cache"])
            nexts.append(int(out["logits"][0, -1].argmax()))
        h = backbone.embed(torch.tensor([[t] for t in nexts]))
        batched, new_caches = step_rows(backbone, h, caches, range(0, backbone.num_layers))
        for row, cache in enumerate(caches):
            single = backbone.forward_ids(torch.tensor([[nexts[row]]]), cache=cache)
            assert torch.allclose(backbone.logits(batched[row:row + 1])[0, -1], single["logits"][0, -1], atol=1e-3)
            assert new_caches[row].lengths == single["cache"].lengths
            # The batched cache continues exactly like the single one.
            follow = torch.tensor([[1000]])
            a = backbone.forward_ids(follow, cache=new_caches[row])["logits"][0, -1]
            b = backbone.forward_ids(follow, cache=single["cache"])["logits"][0, -1]
            assert torch.allclose(a, b, atol=1e-3)


FORCED = ["<|tool_call_start|>[eval(code='const note: Neuralese<string> = ", {"neuralese": "write"},
          ";\\nreturn note;')]<|tool_call_end|>"]


def test_scheduler_batches_writers_and_readers_and_streams(engine):
    from natlang_neuralese.serve.engine import GenerationRequest
    from natlang_neuralese.serve.http import serve

    # Same result whether a request runs alone or in a batch with others.
    alone = engine.generate(GenerationRequest(messages=[{"role": "user", "content": "go"}], forced=FORCED, seed=1))
    engine.start()
    server = serve(engine)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        futures = [engine.submit(GenerationRequest(messages=[{"role": "user", "content": "go"}], forced=FORCED, seed=1)),
                   engine.submit(GenerationRequest(messages=[{"role": "user", "content": "say hi"}], max_tokens=5)),
                   engine.submit(GenerationRequest(messages=[{"role": "user", "content": "go"}], forced=FORCED, seed=1))]
        results = [f.result(timeout=300) for f in futures]
        assert results[0]["neuralese"]["blocks"][0]["id"] == alone["neuralese"]["blocks"][0]["id"]
        assert results[2]["neuralese"]["blocks"][0]["id"] == alone["neuralese"]["blocks"][0]["id"]
        assert results[1]["usage"]["completion_tokens"] >= 1
        # Streaming.
        body = json.dumps({"messages": [{"role": "user", "content": "go"}], "x_natlang_forced": FORCED, "seed": 1,
                           "stream": True}).encode()
        request = urllib.request.Request(base + "/v1/chat/completions", data=body, method="POST")
        with urllib.request.urlopen(request) as response:
            assert response.headers["content-type"] == "text/event-stream"
            events = [line[6:] for line in response.read().decode().splitlines() if line.startswith("data: ")]
        assert events[-1] == "[DONE]"
        chunks = [json.loads(e) for e in events[:-1]]
        text = "".join(c["choices"][0]["delta"]["content"] for c in chunks
                       if isinstance(c["choices"][0]["delta"].get("content"), str))
        blocks = [c["choices"][0]["delta"]["content"][0]["id"] for c in chunks
                  if isinstance(c["choices"][0]["delta"].get("content"), list)]
        assert "eval(code=" in text and blocks == [alone["neuralese"]["blocks"][0]["id"]]
        final = chunks[-1]
        assert final["choices"][0]["finish_reason"] == "tool_calls"
        strip = lambda m: [c["function"] for c in m["tool_calls"]]  # noqa: E731 - call IDs carry request IDs
        assert strip(final["x_natlang_message"]) == strip(alone["choices"][0]["message"])
    finally:
        server.shutdown()
        engine.stop()
