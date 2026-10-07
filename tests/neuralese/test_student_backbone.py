"""A port trained on a merged crisp student records it (`backbone` in the checkpoint); servers rebuild that backbone
from the checkpoint alone, and refuse a student adapter that changed since."""

import pytest
import torch

peft = pytest.importorskip("peft")


def _student(tmp_path, model):
    config = peft.LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj"], lora_dropout=0.0)
    wrapped = peft.get_peft_model(model, config)
    torch.manual_seed(0)
    for name, param in wrapped.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(param, std=0.05)
    out = tmp_path / "student"
    wrapped.save_pretrained(out)
    wrapped.unload()
    return out


def test_a_checkpoint_rebuilds_its_student_backbone(tmp_path, device):
    from natlang_neuralese.model.lfm2_port import backbone_identity, load_backbone
    from natlang_neuralese.serve import load_engine

    base, _ = load_backbone(dtype=torch.float32)
    student = _student(tmp_path, base)
    identity = backbone_identity(str(student))
    assert identity["student_lora_sha256"] and identity["revision"]
    plain = load_engine(device=device, dtype=torch.float32)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"port_config": {"cutoff": plain.heads.cutoff, **plain.heads.port_config()},
                "heads": plain.heads.state_dict(), "control_rows": plain.backbone.control_rows.detach(),
                "backbone": identity}, checkpoint)
    rebuilt = load_engine(heads_checkpoint=str(checkpoint), device=device, dtype=torch.float32)
    name = "model.layers.2.self_attn.q_proj.weight"
    plain_q, rebuilt_q = (dict(e.backbone.hf.named_parameters())[name] for e in (plain, rebuilt))
    assert not torch.allclose(plain_q, rebuilt_q), "the student LoRA is merged"
    # A changed student adapter is refused rather than silently served under the trained heads.
    weights = student / "adapter_model.safetensors"
    weights.write_bytes(weights.read_bytes()[:-4] + b"\0\0\0\0")
    with pytest.raises(ValueError, match="changed"):
        load_engine(heads_checkpoint=str(checkpoint), device=device, dtype=torch.float32)
