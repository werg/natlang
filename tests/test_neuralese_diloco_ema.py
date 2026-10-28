"""DiLoCo coordinator EMA of global versions, and the EMA-weights export."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from natlang_neuralese.train.diloco.coordinator import Coordinator, global_ema, global_masters, read_manifest
from natlang_neuralese.train.diloco.protocol import ema_decay_at, save_tensors

from test_neuralese_diloco import _init, _model


def _merge(coordinator, root, session, sequence, scale):
    for fragment in coordinator.fragments:
        update = Path(root) / f"u-{sequence}-{fragment.index}.safetensors"
        save_tensors(update, {"delta": torch.full((fragment.length,), scale)})
        coordinator.submit(fragment.index, 0, session, sequence, sequence - 1, 4, 100, update)


def test_coordinator_keeps_an_ema_of_global_versions_with_warmup(tmp_path):
    root = tmp_path / "c"
    _init(root, _model(), workers=1, quorum=1, max_staleness=4, ema={"decay": 0.9, "warmup": 4})
    seed, _ = global_masters(root)
    expected = {name: value.clone() for name, value in seed.items()}
    coordinator = Coordinator(root)
    session = "2" * 8 + "-2222-2222-2222-" + "2" * 12
    coordinator.lease(0, session)
    latest = None
    for sequence, scale in enumerate([0.05, -0.02, 0.03], start=1):
        _merge(coordinator, tmp_path, session, sequence, scale)
        latest, _ = global_masters(root)
        weight = 1 - ema_decay_at(0.9, 4, sequence - 1)
        expected = {name: expected[name] + weight * (latest[name] - expected[name]) for name in latest}
        averaged, info = global_ema(root)
        assert info["versions"] == [sequence] * len(coordinator.fragments)
        for name in averaged:
            assert torch.allclose(averaged[name], expected[name], atol=1e-6)
    # The EMA lags the weights (it is not the weights) and survives a coordinator restart.
    assert any(not torch.equal(averaged[name], latest[name]) for name in averaged)
    Coordinator(root)
    restarted, _ = global_ema(root)
    assert all(torch.equal(restarted[name], averaged[name]) for name in averaged)


def test_coordinator_without_an_ema_is_unchanged_and_refuses_its_export(tmp_path):
    _init(tmp_path / "plain", _model(), workers=1, quorum=1)
    assert "ema" not in read_manifest(tmp_path / "plain")
    with pytest.raises(ValueError, match="no EMA"):
        global_ema(tmp_path / "plain")
    with pytest.raises(ValueError, match="EMA"):
        _init(tmp_path / "bad", _model(), workers=1, quorum=1, ema={"decay": 1.5})


def test_diloco_export_writes_the_coordinator_ema_as_a_model_only_checkpoint(tmp_path, monkeypatch):
    import json

    from test_native_serve import _tiny_source
    from test_native_state_sft import graph
    from natlang_neuralese.model.native_model import load_native_checkpoint, load_native_model
    from natlang_neuralese.train import native_diloco
    from natlang_neuralese.train import native_state_sft as trainer
    from natlang_neuralese.train.diloco.coordinator import initialize_coordinator

    base = tmp_path / "base"
    _tiny_source(base)
    _, _, _, config = load_native_model(base, dtype=torch.bfloat16, rank=4)
    paths = {}
    for split in ("train", "test"):
        row = dict(graph(split), split=split, source_groups=[split], tokenizer_fingerprint=config["tokenizer_fingerprint"])
        paths[split] = tmp_path / (split + ".jsonl")
        paths[split].write_text(json.dumps(row) + "\n")
    out = tmp_path / "learner"
    assert trainer.main(["--base", str(base), "--dtype", "bfloat16", "--rank", "4", "--device", "cpu",
                         "--native-graphs", str(paths["train"]), "--held-native-graphs", str(paths["test"]),
                         "--epochs", "1", "--eval-every", "100", "--out", str(out)]) == 0
    model, named = native_diloco._trainable(out / "checkpoint.pt", base)
    root = tmp_path / "coordinator"
    masters = {name: parameter.detach().float() for name, parameter in named}
    initialize_coordinator(root, run_id="ema", masters=masters, identity={"parent_checkpoint_sha256": "p" * 64},
                           workers=1, min_quorum=1, fragment_bytes=1 << 14, grace_seconds=0.0, max_staleness=4,
                           ema={"decay": 0.5})
    coordinator = Coordinator(root)
    session = "3" * 8 + "-3333-3333-3333-" + "3" * 12
    coordinator.lease(0, session)
    for sequence, scale in enumerate([0.01, 0.02], start=1):
        _merge(coordinator, tmp_path, session, sequence, scale)
    averages, _ = global_ema(root)
    latest, _ = global_masters(root)
    assert any(not torch.equal(averages[name], latest[name]) for name in averages)
    args = SimpleNamespace(root=root, ema_out=tmp_path / "ema-out")
    native_diloco.export_ema(args, read_manifest(root), model, named, out / "checkpoint.pt", 7)
    exported, *_ = load_native_checkpoint(tmp_path / "ema-out" / "checkpoint.pt", base=base, allow_weights_only=True)
    from natlang_neuralese.train.native_state_sft import native_trainable_parameters
    for name, parameter in native_trainable_parameters(exported):
        assert torch.equal(parameter, averages[name].to(parameter.dtype)), name
    state = torch.load(tmp_path / "ema-out" / "checkpoint.pt", weights_only=False)
    assert "training" not in state and state["ema_export"]["source"] == "diloco-coordinator-global-ema"
    assert state["ema_export"]["ema"] == {"decay": 0.5, "warmup": 0}


def test_coordinator_cli_keeps_an_ema_by_default_with_an_explicit_opt_out():
    from natlang_neuralese.train.native_diloco import DEFAULT_COORDINATOR_EMA, parse_coordinator_ema

    assert parse_coordinator_ema(None) == DEFAULT_COORDINATOR_EMA == {"decay": 0.99, "warmup": 0}
    assert parse_coordinator_ema("off") is None
    assert parse_coordinator_ema('{"decay": 0.9, "warmup": 3}') == {"decay": 0.9, "warmup": 3}
