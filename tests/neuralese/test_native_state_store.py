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

