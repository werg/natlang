"""Native state and named tensor storage, custody, persistence and HTTP transport."""

import json
import threading
import urllib.error
import urllib.request
from types import SimpleNamespace

import pytest
import torch

from natlang_neuralese.model.native_artifact import NativeTensorArtifact, encode_native_tensor_artifact
from natlang_neuralese.model.state_block import NativeStateBlock
from natlang_neuralese.serve.store import TensorStore, encode_native_state, native_state_id

IDENTITY = {
    "checkpoint_sha256": "2" * 64,
    "tokenizer_fingerprint": "1" * 64,
    "embedding_weight_sha256": "3" * 64,
    "architecture": "fixture",
}


def _state(value: float, text="shared"):
    return NativeStateBlock(
        text=text,
        token_ids=torch.tensor([1], dtype=torch.int64),
        tokenizer_fingerprint=IDENTITY["tokenizer_fingerprint"],
        checkpoint_sha256=IDENTITY["checkpoint_sha256"],
        embedding_weight_sha256=IDENTITY["embedding_weight_sha256"],
        embeddings=torch.full((1, 4), value, dtype=torch.float32),
        positions=torch.tensor([0]),
        layers=({"kind": "attention", "raw_k": torch.full((1, 1, 2), value, dtype=torch.float32),
                 "v": torch.full((1, 1, 2), -value, dtype=torch.float32)},),
        architecture="fixture",
    )


def _artifact(value: float):
    return NativeTensorArtifact(
        "parameter-set", dict(IDENTITY),
        {"adapter.coefficients": torch.tensor([[value, value + 1]], dtype=torch.float32)},
        provenance={"fixture": "native-store-test"},
    ).validate()


def test_owner_collection_and_pins_preserve_native_artifact_custody():
    store = TensorStore()
    first = store.put_native_state(_state(1), "actor-a")
    second = store.put_native_tensor_artifact(_artifact(1), "actor-b")
    shared = store.put_native_state(_state(2))
    store.hold("actor-b", [first.id])

    assert store.collect(set(), "actor-a") == []
    assert store.get_native_state(first.id) is not None
    assert store.collect(set(), "actor-b") == sorted([first.id, second.id])
    assert store.get_native_state(first.id) is None

    store.pin(shared.id, "actor-a")
    assert store.collect(set(), "actor-a") == []
    assert store.collect(set()) == []
    store.unpin(shared.id, "actor-a")
    assert store.collect(set(), "actor-a") == [shared.id]
    assert store.get_native_state(shared.id) is None
    store.close()


def test_persistent_store_spills_and_reloads_native_state_and_tensor_artifact(tmp_path):
    states = [_state(float(i)) for i in range(1, 5)]
    artifacts = [_artifact(float(i)) for i in range(1, 3)]
    store = TensorStore(tmp_path, resident_bytes=40)
    state_records = [store.put_native_state(state, "producer") for state in states]
    tensor_records = [store.put_native_tensor_artifact(artifact, "consumer") for artifact in artifacts]
    assert len(store) == 6 and store.resident < 6
    store.pin(state_records[0].id, "archive")
    assert store.get_native_state(state_records[0].id).id == state_records[0].id
    store.close()

    store = TensorStore(tmp_path, resident_bytes=40)
    assert len(store) == 6 and store.resident == 0
    restored_state = store.get_native_state(state_records[0].id)
    restored_artifact = store.get_native_tensor_artifact(tensor_records[0].id)
    assert restored_state.block.metadata() == states[0].metadata()
    assert torch.equal(restored_state.block.layers[0]["raw_k"], states[0].layers[0]["raw_k"])
    assert restored_artifact.artifact.metadata() == artifacts[0].metadata()
    assert torch.equal(restored_artifact.artifact.tensors["adapter.coefficients"],
                       artifacts[0].tensors["adapter.coefficients"])
    assert store.holders(state_records[1].id) == {"producer"}
    assert store.collect(set(), "producer") == sorted(record.id for record in state_records[1:])
    assert store.get_native_state(state_records[1].id) is None
    assert store.get_native_tensor_artifact(tensor_records[0].id) is not None
    store.close()


def test_native_store_refuses_legacy_flat_rows_from_a_persisted_store(tmp_path):
    legacy_id = "nz1_" + "a" * 52
    store = TensorStore(tmp_path)
    path = store._path(legacy_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"legacy flat payload")
    store._db.execute("INSERT INTO blocks (id, bytes) VALUES (?, ?)", (legacy_id, 19))
    store.close()

    store = TensorStore(tmp_path)
    with pytest.raises(ValueError, match="unsupported non-native artifact ID"):
        store.get_native_state(legacy_id)
    store.close()


def test_value_harness_fetches_native_state_bytes_from_native_store():
    from natlang_neuralese.rl.value_harness import native_artifact_bytes
    from natlang_neuralese.serve.store import decode_native_state

    block = _state(7)
    store = TensorStore()
    record = store.put_native_state(block)
    raw = native_artifact_bytes(SimpleNamespace(store=store), record.id)
    restored, actual_id = decode_native_state(raw)
    assert actual_id == record.id
    assert torch.equal(restored.embeddings, block.embeddings)
    with pytest.raises(ValueError, match="omitted native state"):
        native_artifact_bytes(SimpleNamespace(store=store), "nzs1_" + "a" * 52)
    store.close()


def test_value_harness_emits_native_role_markers_for_writer_spans():
    from natlang_neuralese.rl.value_harness import _forced_native_reply_tokens

    fixture = SimpleNamespace(role_tokens=SimpleNamespace(produce_open=91, produce_close=92),
                              stop_ids=[99],
                              _template_tokens=lambda text: [ord(character) for character in text],
                              _tokens=lambda text: [ord(character) for character in text])
    tokens = _forced_native_reply_tokens(fixture, [('text', 'before'), ('native-state', 'note'), ('text', 'after')])
    assert tokens == [*map(ord, 'before'), 91, *map(ord, 'note'), 92, *map(ord, 'after'), 99]


def test_rl_archive_id_scanner_rejects_flat_projected_references():
    from natlang_neuralese.rl.graph_replay import block_ids

    assert set(block_ids({"input": "nzs1_" + "a" * 52,
                          "gradient": "nnt1_" + "b" * 52})) == {
                              "nzs1_" + "a" * 52, "nnt1_" + "b" * 52}
    with pytest.raises(ValueError, match="projected artifact"):
        list(block_ids({"input": "nz1_" + "c" * 52}))


class _FakeEngine:
    def __init__(self, store):
        self.store = store


def _call(base, method, path, body=None, owner=None):
    data = body if isinstance(body, bytes) or body is None else json.dumps(body).encode()
    headers = {"x-natlang-owner": owner} if owner else {}
    request = urllib.request.Request(base + path, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def test_http_native_artifact_routes_share_owner_custody_and_collection():
    from natlang_neuralese.model.native_artifact import native_tensor_artifact_id
    from natlang_neuralese.serve.http import serve

    store = TensorStore()
    httpd = serve(_FakeEngine(store))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        state = _state(1)
        state_id = native_state_id(state)
        state_body = encode_native_state(state)
        status, metadata = _call(base, "PUT", f"/v1/neuralese/native-states/{state_id}", state_body, "producer")
        assert status == 201
        assert json.loads(metadata)["schema"] == "neuralese.native-state/1"
        status, body = _call(base, "GET", f"/v1/neuralese/native-states/{state_id}")
        assert status == 200 and body == state_body

        artifact = _artifact(4)
        artifact_id = native_tensor_artifact_id(artifact)
        artifact_body = encode_native_tensor_artifact(artifact)
        status, metadata = _call(base, "PUT", f"/v1/neuralese/native-tensor-artifacts/{artifact_id}",
                                 artifact_body, "consumer")
        assert status == 201
        assert json.loads(metadata)["kind"] == "parameter-set"
        assert _call(base, "GET", f"/v1/neuralese/native-tensor-artifacts/{artifact_id}")[1] == artifact_body

        assert store.holders(state_id) == {"producer"}
        assert store.holders(artifact_id) == {"consumer"}
        assert _call(base, "POST", f"/v1/neuralese/native-states/{state_id}/pin", owner="producer")[0] == 200
        assert json.loads(_call(base, "POST", "/v1/neuralese/collect", {"referenced": []}, "producer")[1]) == {"removed": []}
        assert json.loads(_call(base, "POST", "/v1/neuralese/collect", {"referenced": []}, "consumer")[1]) == {"removed": [artifact_id]}
        assert _call(base, "GET", f"/v1/neuralese/native-tensor-artifacts/{artifact_id}/meta")[0] == 404
        assert _call(base, "POST", f"/v1/neuralese/native-states/{state_id}/unpin", owner="producer")[0] == 200
        assert json.loads(_call(base, "POST", "/v1/neuralese/collect", {"referenced": []}, "producer")[1]) == {"removed": [state_id]}
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join()
        store.close()
