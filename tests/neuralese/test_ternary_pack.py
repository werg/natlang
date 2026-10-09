import torch

from natlang_neuralese.maple import ternary_pack
from natlang_neuralese.maple.ternary import ternarize


def test_ternary_tensors_pack_exactly_and_others_are_refused():
    torch.manual_seed(0)
    weight = ternarize((torch.randn(3, 7, 33) * 0.02).to(torch.bfloat16))  # odd sizes: padding path
    packed = ternary_pack.pack_tensor(weight)
    assert packed is not None and torch.equal(ternary_pack.unpack_tensor(packed), weight)
    assert packed["codes"].numel() * 4 >= weight.numel() and packed["codes"].numel() < weight.numel()
    assert ternary_pack.pack_tensor(torch.randn(4, 8)) is None  # not ternary
    assert ternary_pack.pack_tensor(torch.ones(8)) is None  # 1-D (norms) stays raw


def test_a_packed_directory_unpacks_bit_exactly(tmp_path):
    from safetensors.torch import load_file, save_file

    torch.manual_seed(1)
    src = tmp_path / "export"
    src.mkdir()
    tensors = {"layer.q_proj.weight": ternarize((torch.randn(16, 32) * 0.02).to(torch.bfloat16)),
               "embed.weight": torch.randn(10, 32).to(torch.bfloat16), "norm.weight": torch.ones(32)}
    save_file(tensors, str(src / "model-00001-of-00001.safetensors"))
    (src / "config.json").write_text("{}")
    report = ternary_pack.pack_dir(src, tmp_path / "packed")
    assert report["packed_tensors"] == 1 and report["packed_gb"] < report["raw_gb"]
    ternary_pack.unpack_dir(tmp_path / "packed", tmp_path / "back")
    back = load_file(str(tmp_path / "back" / "model-00001-of-00001.safetensors"))
    assert all(torch.equal(back[k], v) for k, v in tensors.items())
    assert (tmp_path / "back" / "config.json").exists()
