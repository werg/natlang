"""Reference server block store: owner-scoped holds, pins and collection; persistent spill with a resident LRU; the
HTTP surface (owner header, holds from responses) over a fake engine that loads no model."""

import json
import threading
import urllib.error
import urllib.request
from concurrent.futures import Future
from types import SimpleNamespace

import pytest
import torch

from natlang_neuralese.serve.store import TensorStore, encode_block, make_block

DIALECT = "nd:natlang@1"


def _blocks(n, width=4):
    return [make_block(torch.full((2, width), float(i)), DIALECT) for i in range(n)]


def test_owner_collect_never_drops_another_owners_blocks():
    store = TensorStore()
    a, b, shared = _blocks(3)
    store.put(a, "s1"), store.put(b, "s2"), store.put(shared, "s1")
    store.hold("s2", [shared.id])
    assert store.collect(set(), "s1") == [a.id]
    assert store.get(b.id) is not None and store.get(shared.id) is not None
    assert store.collect(set(), "s2") == sorted([b.id, shared.id])
    assert len(store) == 0


def test_owner_collect_keeps_referenced_and_takes_them_over():
    store = TensorStore()
    a, b = _blocks(2)
    store.put(a, "s1"), store.put(b)  # b is anonymous: no owner's collection may drop it
    assert store.collect({a.id, b.id}, "s2") == []
    assert store.holders(b.id) == {"s2"}
    assert store.collect(set(), "s1") == []  # s2 still holds a
    assert store.collect(set(), "s2") == sorted([a.id, b.id])


def test_pins_are_per_owner_and_protect_against_everyone():
    store = TensorStore()
    (a,) = _blocks(1)
    store.put(a, "s1")
    store.pin(a.id, "s1")
    assert store.collect(set(), "s1") == [] and store.collect(set(), "s2") == [] and store.collect(set()) == []
    store.unpin(a.id, "s2")  # not s2's pin
    assert store.collect(set(), "s1") == []
    store.unpin(a.id, "s1")
    assert store.collect(set(), "s1") == [a.id]
    with pytest.raises(KeyError):
        store.pin(a.id, "s1")


def test_anonymous_collect_drops_only_unheld_blocks():
    store = TensorStore()
    a, b, c = _blocks(3)
    store.put(a), store.put(b, "s1"), store.put(c)
    store.pin(c.id)
    assert store.collect(set()) == [a.id]
    store.unpin(c.id)
    assert store.collect(set()) == [c.id]
    assert store.get(b.id) is not None


def test_persistent_store_survives_restart_and_bounds_resident_memory(tmp_path):
    blocks = _blocks(4, width=64)  # 2 x 64 f32 = 512 bytes of payload each
    store = TensorStore(tmp_path, resident_bytes=1100)
    for block in blocks:
        store.put(block, "s1")
    store.pin(blocks[0].id, "s2")
    assert len(store) == 4 and store.resident == 2
    reloaded = store.get(blocks[0].id)  # evicted, read back from disk
    assert reloaded.id == blocks[0].id and torch.equal(reloaded.payload, blocks[0].payload)
    with pytest.raises(RuntimeError, match="another process"):
        TensorStore(tmp_path)
    store.close()

    store = TensorStore(tmp_path, resident_bytes=1100)
    assert len(store) == 4 and store.resident == 0
    assert all(torch.equal(store.get(b.id).payload, b.payload) for b in blocks)
    assert store.holders(blocks[1].id) == {"s1"}
    assert store.collect(set(), "s1") == sorted(b.id for b in blocks[1:])  # the pin outlived the restart
    assert not any((tmp_path / "blocks").glob(f"*/{blocks[1].id}.safetensors"))
    assert store.get(blocks[1].id) is None and store.get(blocks[0].id) is not None
    store.close()


def test_persistent_store_learns_a_type_on_disk(tmp_path):
    (block,) = _blocks(1)
    store = TensorStore(tmp_path)
    store.put(block)
    store.put(make_block(block.payload, DIALECT, type="Neuralese<string>"))
    store.close()
    store = TensorStore(tmp_path)
    assert store.get(block.id).type == "Neuralese<string>"
    store.close()


class _FakeEngine:
    """Enough of `Engine` for the block endpoints and a chat that writes one block."""

    def __init__(self, store):
        self.store, self.dialect, self.width, self.max_block, self.model_name = store, DIALECT, 4, 8, "fake"
        self.heads = SimpleNamespace(cutoff=6)
        self.projections = {}
        self.written = 0

    def submit(self, request):
        self.written += 1
        block = self.store.put(make_block(torch.full((1, self.width), 100.0 + self.written), DIALECT))
        future = Future()
        future.set_result({"id": "chatcmpl-1", "object": "chat.completion", "choices": [
            {"index": 0, "finish_reason": "stop",
             "message": {"role": "assistant", "content": [{"type": "neuralese", "id": block.id}]}}],
            "usage": {}, "neuralese": {"blocks": [block.meta()]}})
        return future


@pytest.fixture
def server():
    from natlang_neuralese.serve.http import serve

    engine = _FakeEngine(TensorStore())
    httpd = serve(engine)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"

    def call(method, path, body=None, owner=None):
        data = body if isinstance(body, bytes) or body is None else json.dumps(body).encode()
        request = urllib.request.Request(base + path, data=data, method=method,
                                         headers={"x-natlang-owner": owner} if owner else {})
        try:
            with urllib.request.urlopen(request) as response:
                return response.status, json.loads(response.read() or b"null")
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    yield engine, call
    httpd.shutdown()
    httpd.server_close()


def test_http_owners_hold_uploads_and_written_blocks(server):
    engine, call = server
    mine, theirs = _blocks(2)
    assert call("PUT", f"/v1/neuralese/blocks/{mine.id}", encode_block(mine), "s1")[0] == 201
    assert call("PUT", f"/v1/neuralese/blocks/{theirs.id}", encode_block(theirs), "s2")[0] == 201
    status, reply = call("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]}, "s1")
    written = reply["neuralese"]["blocks"][0]["id"]
    assert status == 200 and engine.store.holders(written) == {"s1"}
    assert call("GET", "/v1/neuralese/info")[1]["store"] == {"owners": True, "persistent": False}

    # s1 collecting everything leaves s2's block; s2 pins its block, which then outlives s2's own collection.
    assert call("POST", "/v1/neuralese/collect", {"referenced": []}, "s1")[1] == {"removed": sorted([mine.id, written])}
    assert call("POST", f"/v1/neuralese/blocks/{theirs.id}/pin", None, "s2")[0] == 200
    assert call("POST", "/v1/neuralese/collect", {"referenced": []}, "s2")[1] == {"removed": []}
    status, error = call("GET", f"/v1/neuralese/blocks/{mine.id}/meta")
    assert status == 404 and error["error"]["code"] == "neuralese-unknown-block"
    assert call("POST", f"/v1/neuralese/blocks/{theirs.id}/unpin", None, "s2")[0] == 200
    assert call("POST", "/v1/neuralese/collect", {"referenced": []}, "s2")[1] == {"removed": [theirs.id]}


def test_http_collect_without_owner_keeps_owned_blocks(server):
    engine, call = server
    owned, anonymous = _blocks(2)
    call("PUT", f"/v1/neuralese/blocks/{owned.id}", encode_block(owned), "s1")
    call("PUT", f"/v1/neuralese/blocks/{anonymous.id}", encode_block(anonymous))
    assert call("POST", "/v1/neuralese/collect", {"referenced": []})[1] == {"removed": [anonymous.id]}
    assert engine.store.get(owned.id) is not None
