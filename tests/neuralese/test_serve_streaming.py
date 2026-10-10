"""The reference server declares streaming in its info and streams a reply in the documented chunk format
(spec/NEURALESE_PORT.md, "Streaming"; plans/STREAMING.md §1.2). A fake engine, no model."""

from __future__ import annotations

import json
import threading
import urllib.request
from concurrent.futures import Future
from types import SimpleNamespace

import pytest
import torch

from natlang_neuralese.serve.store import TensorStore, make_block

DIALECT = "nd:test@1"


class _FakeEngine:
    """Enough of `Engine` for info and a chat that streams text, one written block, and more text."""

    def __init__(self):
        self.store, self.dialect, self.width, self.max_block, self.model_name = TensorStore(), DIALECT, 4, 8, "fake"
        self.heads = SimpleNamespace(cutoff=6)
        self.projections, self.context = {}, 8192

    def submit(self, request):
        block = self.store.put(make_block(torch.ones((1, self.width)), DIALECT))
        if request.on_delta is not None:
            request.on_delta({"text": "Note: "})
            request.on_delta({"neuralese": block.meta()})
            request.on_delta({"text": " done."})
        content = [{"type": "text", "text": "Note: "}, {"type": "neuralese", "id": block.id},
                   {"type": "text", "text": " done."}]
        future = Future()
        future.set_result({"id": "chatcmpl-1", "object": "chat.completion", "choices": [
            {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4}, "neuralese": {"blocks": [block.meta()]},
            "x_natlang_guidance": {"rejections": []}})
        return future


@pytest.fixture
def base():
    from natlang_neuralese.serve.http import serve

    httpd = serve(_FakeEngine())
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def test_info_declares_streaming(base):
    with urllib.request.urlopen(base + "/v1/neuralese/info") as response:
        assert json.loads(response.read())["stream"] is True


def test_stream_carries_text_block_and_final_message(base):
    request = urllib.request.Request(base + "/v1/chat/completions", method="POST", data=json.dumps(
        {"messages": [{"role": "user", "content": "hi"}], "stream": True}).encode())
    with urllib.request.urlopen(request) as response:
        assert response.headers["content-type"] == "text/event-stream"
        lines = [line[len("data: "):] for line in response.read().decode().split("\n") if line.startswith("data: ")]
    assert lines[-1] == "[DONE]"
    chunks = [json.loads(line) for line in lines[:-1]]
    deltas = [chunk["choices"][0]["delta"] for chunk in chunks]
    block_id = chunks[-1]["neuralese"]["blocks"][0]["id"]
    assert [d.get("content") for d in deltas if "content" in d] == [
        "Note: ", [{"type": "neuralese", "id": block_id}], " done."]
    final = chunks[-1]
    assert final["choices"][0]["finish_reason"] == "stop"
    assert final["x_natlang_message"]["content"][1] == {"type": "neuralese", "id": block_id}
    assert final["x_natlang_guidance"] == {"rejections": []}
    assert final["usage"] == {"prompt_tokens": 3, "completion_tokens": 4}
