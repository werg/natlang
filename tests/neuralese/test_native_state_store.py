import torch
import pytest

from natlang_neuralese.model.state_block import NativeStateBlock
from natlang_neuralese.serve.store import TensorStore, decode_native_state, encode_native_state, native_state_id
from natlang_neuralese.serve.chat import RequestError, render_messages


def _block():
    return NativeStateBlock(
        text="hello", token_ids=torch.tensor([11, 12], dtype=torch.int64),
        tokenizer_fingerprint="1" * 64, checkpoint_sha256="2" * 64,
        embedding_weight_sha256="3" * 64, embeddings=torch.tensor([[.1, .2], [.3, .4]], dtype=torch.bfloat16),
        positions=torch.tensor([7, 8], dtype=torch.int64),
        layers=({"kind": "attention", "raw_k": torch.ones((2, 1, 2), dtype=torch.float16),
                 "v": torch.zeros((2, 1, 2), dtype=torch.float16)},
                {"kind": "finite_convolution", "gated_inputs": torch.ones((2, 3), dtype=torch.float32)}),
        architecture="fixture",
    )


def test_native_state_wire_round_trip_is_deterministic_and_verified():
    block = _block()
    body = encode_native_state(block)
    restored, block_id = decode_native_state(body)
    assert block_id == native_state_id(block)
    assert block_id.startswith("nzs1_")
    assert block_id == "nzs1_zqbj3ixi4kqxyhv5ofcpuhabvqd76bwigo4z6wq42ujoq6lbpewq"
    assert encode_native_state(restored) == body
    assert restored.text == block.text
    assert restored.tokenizer_fingerprint == block.tokenizer_fingerprint
    assert torch.equal(restored.layers[0]["raw_k"], block.layers[0]["raw_k"])
    assert torch.equal(restored.layers[1]["gated_inputs"], block.layers[1]["gated_inputs"])


def test_native_state_wire_rejects_payload_corruption():
    body = bytearray(encode_native_state(_block()))
    body[-1] ^= 1
    with pytest.raises(ValueError, match="ID does not match"):
        decode_native_state(bytes(body))


def test_native_state_uses_shared_store_holds_and_survives_restart(tmp_path):
    block = _block()
    store = TensorStore(tmp_path, resident_bytes=1)
    record = store.put_native_state(block, "owner-a")
    store.pin(record.id, "owner-b")
    assert store.get_native_state(record.id).meta() == record.meta()
    assert store.collect(set(), "owner-a") == []
    store.close()
    store = TensorStore(tmp_path, resident_bytes=1)
    restored = store.get_native_state(record.id)
    assert restored is not None and restored.id == record.id
    assert store.holders(record.id) == {"owner-b"}
    store.unpin(record.id, "owner-b")
    assert store.collect(set(), "owner-a") == []
    assert store.collect(set(), "owner-b") == [record.id]
    assert not list((tmp_path / "blocks").glob(f"*/{record.id}.safetensors"))
    store.close()


def test_native_state_parts_render_as_structured_imports():
    messages = [{"role": "user", "content": [{"type": "text", "text": "read "},
                                                  {"type": "native_state", "id": "nzs1_" + "a" * 52}]}]
    apply = lambda rows, tools: rows[0]["content"]
    rendered = render_messages(messages, None, apply)
    assert rendered.blocks == ["nzs1_" + "a" * 52]
    assert len(rendered.segments) == 2 and rendered.segments[1] == 0


def test_projected_parts_are_rejected_instead_of_being_forwarded_to_the_template():
    messages = [{"role": "user", "content": [{"type": "neuralese", "id": "nz1_" + "a" * 52}]}]
    with pytest.raises(RequestError, match="projected neuralese parts are retired"):
        render_messages(messages, None, lambda rows, tools: rows[0]["content"])



def _stored_file(tmp_path, block, name="state.safetensors"):
    from natlang_neuralese.serve.store import write_native_state_file
    path = tmp_path / name
    return path, write_native_state_file(path, block.metadata(), block.tensors())


def test_native_state_file_is_the_wire_encoding_and_reads_without_the_payload(tmp_path, monkeypatch):
    """One format: the file writer (the SGLang fork's capture output) writes exactly `encode_native_state`'s bytes,
    and reading it decodes only the two index tensors while still checking identity, layout and content id."""
    from natlang_neuralese.serve import store as store_module
    block = _block()
    path, block_id = _stored_file(tmp_path, block)
    assert block_id == native_state_id(block) and path.read_bytes() == encode_native_state(block)
    decoded = []
    original = store_module._array
    monkeypatch.setattr(store_module, "_array", lambda buffer, dtype, shape: (decoded.append(dtype), original(buffer, dtype, shape))[1])
    state = store_module.read_native_state_file(path, block_id)
    assert decoded == ["I64", "I64"]   # token_ids and positions only
    assert state.id == block_id and state.meta() == {"id": block_id, **block.metadata()}
    assert state.token_ids.tolist() == [11, 12] and state.layer_kinds == ("attention", "finite_convolution")
    state.require_native(checkpoint_sha256="2" * 64, tokenizer_fingerprint="1" * 64, embedding_weight_sha256="3" * 64,
                         architecture="fixture", layer_kinds=("attention", "finite_convolution"))
    with pytest.raises(ValueError, match="foreign native state checkpoint_sha256"):
        state.require_native(checkpoint_sha256="4" * 64, tokenizer_fingerprint="1" * 64,
                             embedding_weight_sha256="3" * 64, architecture="fixture")
    with pytest.raises(ValueError, match="layer layout differs"):
        state.require_native(checkpoint_sha256="2" * 64, tokenizer_fingerprint="1" * 64,
                             embedding_weight_sha256="3" * 64, architecture="fixture", layer_kinds=("attention",))
    assert len(decoded) == 2
    loaded = state.load()
    assert native_state_id(loaded) == block_id and torch.equal(loaded.layers[0]["raw_k"], block.layers[0]["raw_k"])


def test_native_state_file_verification_rejects_corruption_nonfinite_values_and_bad_layouts(tmp_path):
    from natlang_neuralese.serve.store import read_native_state_file, write_native_state_file
    block = _block()
    path, block_id = _stored_file(tmp_path, block)
    body = bytearray(path.read_bytes())
    body[-1] ^= 1
    path.write_bytes(bytes(body))
    with pytest.raises(ValueError, match="holds native state"):
        read_native_state_file(path)
    tensors = dict(block.tensors())
    tensors["layers.1.gated_inputs"] = torch.tensor([[1.0, float("nan"), 0.0], [0.0, 0.0, 0.0]])
    write_native_state_file(path, block.metadata(), tensors)
    with pytest.raises(ValueError, match="non-finite"):
        read_native_state_file(path)
    tensors = dict(block.tensors())
    del tensors["layers.1.gated_inputs"]
    write_native_state_file(path, block.metadata(), tensors)
    with pytest.raises(ValueError, match="inventory differs"):
        read_native_state_file(path)


def test_adopting_a_native_state_file_moves_it_into_the_store(tmp_path):
    """The capture -> store -> import path: a file written in the store format is verified and renamed into the
    store; imports hand out its stored path; the learner and computing engines read the same file."""
    block = _block()
    store = TensorStore(tmp_path / "store")
    incoming = tmp_path / "store" / "capture"
    incoming.mkdir()
    path, block_id = _stored_file(incoming, block)
    with pytest.raises(ValueError, match="metadata differs"):
        store.adopt_native_state_file(path, metadata={**block.metadata(), "text": "other"})
    with pytest.raises(ValueError, match="token IDs differ"):
        store.adopt_native_state_file(path, token_ids=[11, 13])
    state = store.adopt_native_state_file(path, "owner-a", metadata=block.metadata(), token_ids=[11, 12])
    assert state.id == block_id and not path.exists() and state.path == store._path(block_id)
    assert state.path.read_bytes() == encode_native_state(block) and store.holders(block_id) == {"owner-a"}
    assert store.native_state_file(block_id) is state and store.resident == 0   # nothing decoded
    assert native_state_id(store.get_native_state(block_id).block) == block_id
    again, _ = _stored_file(incoming, block)
    assert store.adopt_native_state_file(again).id == block_id and not again.exists()
    store.close()
    reopened = TensorStore(tmp_path / "store")
    assert reopened.native_state_file(block_id).path == state.path
    state.path.unlink()
    assert reopened.native_state_file(block_id) is None and not reopened.contains(block_id)
    reopened.close()
