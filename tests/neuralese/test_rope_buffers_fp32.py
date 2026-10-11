"""RoPE frequency buffers stay FP32 in BF16 loads (a BF16 inv_freq puts a position-dependent phase error into every
attention layer). Mellum/Maple: the port rebuilds rotary_emb in FP32 after its dtype cast; LFM: from_pretrained."""
import pytest
import torch


def rope_buffers(module):
    return {name: buffer.dtype for name, buffer in module.named_buffers() if "inv_freq" in name}


def test_tiny_mellum_bf16_load_keeps_fp32_rope_tables(tmp_path):
    test_mellum_port = pytest.importorskip("test_mellum_port")
    from natlang_neuralese.maple.model import load_maple
    test_mellum_port.tiny_mellum(tmp_path)
    model = load_maple(tmp_path, device="cpu", dtype=torch.bfloat16, ternary_attention=False)
    buffers = rope_buffers(model)
    assert buffers and all(dtype == torch.float32 for dtype in buffers.values()), buffers


def test_lfm_bf16_load_keeps_fp32_rope_tables():
    from pathlib import Path
    from natlang_neuralese.model.lfm2_port import load_backbone, resolve_base
    if not Path(resolve_base(None)).exists():
        pytest.skip("LFM2.5-350M is not in the local HF cache")
    model, _ = load_backbone(dtype=torch.bfloat16, device="cpu")
    buffers = rope_buffers(model)
    assert buffers and all(dtype == torch.float32 for dtype in buffers.values()), buffers
