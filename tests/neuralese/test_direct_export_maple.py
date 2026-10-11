"""Pop's strict Maple direct export (train/direct_export.py) round-trips on a tiny Mellum: a direct-stage checkpoint
with the complete latent policy exports with its proofs and restores into the clean BF16 base with the exact trained
tensors. (The Mellum E2E smoke's load first failed on the coverage names' "backbone." prefix.)"""
import json

import pytest
import torch

from natlang_neuralese.common.hashing import canonical_json_sha256_hex, sha256_file_hex


def test_maple_direct_export_round_trips_the_trained_tensors(tmp_path, monkeypatch):
    test_mellum_port = pytest.importorskip("test_mellum_port")
    from natlang_neuralese.maple import foundation_heads, student
    from natlang_neuralese.maple.model import load_maple
    from natlang_neuralese.model.neuralese_projection import NeuraleseProjection
    from natlang_neuralese.serve import load_engine
    from natlang_neuralese.train import direct_export as dx
    from natlang_neuralese.train.backbone_policy import latent_backbone_parameters

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    test_mellum_port.tiny_mellum(model_dir)

    class Vocab:
        eos_token_id = 2
        from natlang_neuralese.serve.chat import declared_history_reasoning
        natlang_history_reasoning = {"policy": declared_history_reasoning("mellum")}

        def get_vocab(self):
            return {"<|extra_token_7|>": 33, "<|extra_token_8|>": 34}

        def convert_tokens_to_ids(self, token):
            return self.get_vocab().get(token)

    def tiny_student(model, state=None, device="cpu", **_):
        loaded = load_maple(model, device="cpu", dtype=torch.float32, ternary_attention=False)
        loaded.config.model_type = "mellum"
        loaded.natlang_precision = "bf16"
        return loaded, Vocab()
    monkeypatch.setattr(foundation_heads, "load_student", tiny_student)
    monkeypatch.setattr(student, "load_student", tiny_student)

    receipt = tmp_path / "init-gate.json"
    receipt.write_text(json.dumps({"passed": False, "artifact": "none"}))
    layers = json.loads((model_dir / "config.json").read_text())["num_hidden_layers"]
    heads = tmp_path / "heads.pt"
    foundation_heads.main(["--model", str(model_dir), "--precision", "bf16", "--init-receipt", str(receipt),
                           "--profile", "raw-token-v1", "--cutoff", str(layers), "--stop-source", "final",
                           "--out", str(heads)])
    # A direct-stage checkpoint: the complete latent policy, moved off its base values.
    engine = load_engine(heads_checkpoint=str(heads), device="cpu")
    named = latent_backbone_parameters(engine.backbone)
    torch.manual_seed(0)
    trained = {"backbone." + n: (q.detach() + 0.01 * torch.randn_like(q)).clone() for n, q in named}
    width = int(engine.backbone.hf.config.hidden_size)
    projection = NeuraleseProjection(width, hidden=8)
    package = __import__("pathlib").Path(dx.__file__).resolve().parents[1]
    code = {str(p.relative_to(package)): sha256_file_hex(p) for p in sorted(package.rglob("*.py"))}
    identity = {"schema": dx.CHECKPOINT_SCHEMA, "options": {"heads": str(heads.resolve()), "quantization": None,
                                                           "projection_blocks": 2, "projection_hidden": 8,
                                                           "projection_kernel": 4, "token_choice_projection": False},
                "inputs": {str(heads.resolve()): sha256_file_hex(heads)}, "code": code}
    stage = tmp_path / "direct_sft"
    stage.mkdir()
    checkpoint = stage / "checkpoint.pt"
    torch.save({"schema": dx.CHECKPOINT_SCHEMA, "identity": identity, "step": 7, "student_parameters": trained,
                "neuralese_projection": projection.state_dict()}, checkpoint)
    checkpoint_sha = sha256_file_hex(checkpoint)
    anchor = torch.load(heads, weights_only=False)
    files = dx._model_input_inventory(model_dir)
    catalog = lambda d: {k: [list(v.shape), str(v.dtype)] for k, v in sorted(d.items())}
    proof = tmp_path / "schema-proof.json"
    proof.write_text(json.dumps({"checkpoint_sha256": checkpoint_sha, "step": 7, "identity": identity,
                                 "heads": {"path": str(heads.resolve()), "sha256": sha256_file_hex(heads)},
                                 "student_parameters_catalog": catalog(trained),
                                 "neuralese_projection_catalog": catalog(projection.state_dict()),
                                 "base_model": {"path": str(model_dir.resolve()), "files": files}}))
    confirmation = tmp_path / "confirmation.json"
    confirmation.write_text(json.dumps({"schema": "natlang.maple-training-base-confirmation/1",
                                        "checkpoint_sha256": checkpoint_sha, "model_path": str(model_dir.resolve()),
                                        "files_sha256": canonical_json_sha256_hex(files),
                                        "same_bytes_used_during_training": True}))
    source = {"backbone_adapter": dx.MAPLE_BF16_ADAPTER, "files": files, "model_path": str(model_dir.resolve()),
              "anchor_sha256": sha256_file_hex(heads), "anchor_backbone_identity": anchor["backbone"],
              "schema_proof_path": str(proof), "schema_proof_sha256": sha256_file_hex(proof),
              "source_temporal_relation": "owner-confirmed-same-bytes-used-at-training",
              "training_base_confirmation_path": str(confirmation),
              "training_base_confirmation_sha256": sha256_file_hex(confirmation)}
    out = tmp_path / "export"
    dx.export_direct_checkpoint(checkpoint, out, base_snapshot=model_dir, base_source_proof=source,
                                backbone_adapter=dx.MAPLE_BF16_ADAPTER)
    with pytest.raises(ValueError, match="base snapshot"):
        dx.load_direct_artifact(out, base_snapshot=None)
    restored = dx.load_direct_artifact(out, base_snapshot=model_dir)
    from natlang_neuralese.train.backbone_policy import plain_named_tensors
    values = plain_named_tensors(restored.engine.backbone.hf)
    for name, value in trained.items():
        assert torch.equal(values[name.removeprefix("backbone.")].float(), value.float()), name
    for key, value in projection.state_dict().items():
        assert torch.equal(restored.neuralese_projection.state_dict()[key], value)
