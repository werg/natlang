import json
import os

import pytest
import torch
from safetensors.torch import save_file

from natlang_neuralese import artifacts


STATE = "nzs1_" + "a" * 52


def nz(path, dialect="native-state/test", width=8, tensors=0):
    """A native `.nz` file (spec/NEURALESE_FILES.md): one state tensor of `width` bytes, `tensors` named tensors."""
    native_tensors = {f"nnt1_{chr(ord('a') + i) * 52}": {"bytes": 4} for i in range(tensors)}
    header = {"format": "natlang.neuralese-file/1", "dialect": dialect, "types": "",
              "exports": {"piece": {"type": "Neuralese<string>", "value": {"$neuralese": {"type": "Neuralese<string>", "id": STATE}}}},
              "native_states": {STATE: {"bytes": width}}, "native_tensors": native_tensors}
    payload = {STATE: torch.zeros(width, dtype=torch.uint8), **{name: torch.zeros(4, dtype=torch.uint8) for name in native_tensors}}
    save_file(payload, str(path), metadata={"natlang": json.dumps(header)})
    return path


def repo_with_corpus(tmp_path):
    (tmp_path / "training").mkdir()
    (tmp_path / "training/neuralese_corpora.json").write_text(json.dumps({"corpora": [{"id": "corpus-a"}]}))
    return tmp_path


def item(identity="bank-test-v1", **extra):
    return {"id": identity, "kind": "prompt-bank", "owner": "dgx", "dialect": "native-state/test",
            "backbone": {"model": "test/model", "revision": "abc"}, "init": {"method": "text", "source": "x.ts"},
            "training": None, "qualification": {"status": "unqualified", "evidence": []}, **extra}


def test_register_verify_resolve_and_refuse_other_spaces(tmp_path):
    repo = repo_with_corpus(tmp_path)
    manifest = artifacts.register(repo, item(), {"bank.nz": nz(tmp_path / "src.nz")})
    assert manifest["files"][0]["nz"] == {"exports": ["piece"], "native_states": 1, "native_tensors": 0,
                                          "dialects": ["native-state/test"]}
    assert artifacts.verify(repo, "bank-test-v1")["id"] == "bank-test-v1"
    path, sha = artifacts.resolve("bank-test-v1", repo=repo, dialect="native-state/test", backbone_model="test/model")
    assert path.name == "bank.nz" and len(sha) == 64
    with pytest.raises(artifacts.ArtifactError, match="dialect"):
        artifacts.resolve("bank-test-v1", repo=repo, dialect="native-state/other")
    with pytest.raises(artifacts.ArtifactError, match="belongs to"):
        artifacts.resolve("bank-test-v1", repo=repo, backbone_model="other/model")
    with pytest.raises(artifacts.ArtifactError, match="already registered"):
        artifacts.register(repo, item(), {"bank.nz": nz(tmp_path / "src2.nz")})


def test_tampered_bytes_and_wrong_dialects_are_refused(tmp_path):
    repo = repo_with_corpus(tmp_path)
    artifacts.register(repo, item(), {"bank.nz": nz(tmp_path / "src.nz")})
    stored = repo / artifacts.STORE / "bank-test-v1/bank.nz"
    os.chmod(stored, 0o644)
    nz(stored, width=4)
    with pytest.raises(artifacts.ArtifactError):
        artifacts.verify(repo, "bank-test-v1")
    with pytest.raises(artifacts.ArtifactError, match="hash mismatch"):
        artifacts.resolve("bank-test-v1", repo=repo)
    with pytest.raises(artifacts.ArtifactError, match="differs from"):
        artifacts.register(repo, item("bank-test-v2"), {"bank.nz": nz(tmp_path / "other.nz", dialect="native-state/other")})
    with pytest.raises(artifacts.ArtifactError, match="semantic-state"):
        artifacts.register(repo, item("bank-test-v3", dialect="nd:test@1"), {"bank.nz": nz(tmp_path / "old.nz")})


def test_trained_artifacts_need_registered_corpora_parent_and_commit(tmp_path):
    repo = repo_with_corpus(tmp_path)
    artifacts.register(repo, item(), {"bank.nz": nz(tmp_path / "src.nz")})
    trained = item("bank-test-trained-v1", init={"method": "trained", "parent": "bank-test-v1"},
                   training={"corpora": ["corpus-a"], "trainer": "train.decision", "commit": "deadbeef"})
    artifacts.register(repo, trained, {"bank.nz": nz(tmp_path / "t.nz")})
    unknown = item("bank-test-trained-v2", init={"method": "trained", "parent": "bank-test-v1"},
                   training={"corpora": ["nope"], "trainer": "train.decision", "commit": "deadbeef"})
    with pytest.raises(artifacts.ArtifactError, match="not registered"):
        artifacts.register(repo, unknown, {"bank.nz": nz(tmp_path / "u.nz")})
    orphan = item("bank-test-trained-v3", init={"method": "trained", "parent": "missing"},
                  training={"corpora": [], "trainer": "t", "commit": "c"})
    with pytest.raises(artifacts.ArtifactError, match="unknown artifact"):
        artifacts.register(repo, orphan, {"bank.nz": nz(tmp_path / "o.nz")})


def test_bundles_hold_native_tensors_and_retired_formats_are_refused(tmp_path):
    repo = repo_with_corpus(tmp_path)
    manifest = artifacts.register(repo, item("bundle-test-v1", kind="bundle"), {"a.nz": nz(tmp_path / "bundle.nz", tensors=2)})
    assert manifest["files"][0]["nz"]["native_tensors"] == 2
    # Native adapters are named tensor artifacts without a dialect, so extra_dialects is retired.
    with pytest.raises(artifacts.ArtifactError, match="extra_dialects is retired"):
        artifacts.register(repo, item("bundle-test-v2", kind="bundle", extra_dialects=["adapter/1;base=x;kind=xs"]),
                           {"a.nz": nz(tmp_path / "b2.nz")})
    # A projected-vector file (a flat `blocks` inventory) is refused, as the runtime loader refuses it.
    old = tmp_path / "old.nz"
    header = {"exports": {}, "blocks": {"a": {"dialect": "native-state/test"}}}
    save_file({"a": torch.zeros(2, 8)}, str(old), metadata={"natlang": json.dumps(header)})
    with pytest.raises(artifacts.ArtifactError, match="blocks"):
        artifacts.register(repo, item("bundle-test-v3", kind="bundle"), {"a.nz": old})


def test_trainer_outputs_register_with_their_parent_found_by_content(tmp_path):
    repo = repo_with_corpus(tmp_path)
    source = nz(tmp_path / "src.nz")
    artifacts.register(repo, item(), {"bank.nz": source})
    assert artifacts.find_by_sha(artifacts.digest(source), repo) == ("bank-test-v1", "bank.nz")
    trained = nz(tmp_path / "system-prompts.nz", width=6)
    manifest = artifacts.register_output(trained, identity="bank-test-trained-v1", kind="prompt-bank",
                                         dialect="native-state/test", backbone={"model": "test/model", "revision": "abc"},
                                         trainer="natlang_neuralese.train.decision", commit="c0ffee",
                                         corpora=["corpus-a"], parent="bank-test-v1", repo=repo)
    entry = artifacts.entry(artifacts.load_registry(repo), "bank-test-trained-v1")
    assert entry["init"] == {"method": "trained", "parent": "bank-test-v1"}
    assert manifest["files"][0]["nz"]["dialects"] == ["native-state/test"]
    snapshot = tmp_path / "hub/models--x/snapshots/0123abcd"
    assert artifacts.backbone_identity(str(snapshot)) == {"model": str(snapshot), "revision": "0123abcd"}
    with pytest.raises(artifacts.ArtifactError, match="cannot pin"):
        artifacts.backbone_identity(str(tmp_path / "unpinned-model"))


def test_a_finished_run_registers_its_bank_with_parent_and_corpora(tmp_path):
    repo = repo_with_corpus(tmp_path)
    source = nz(tmp_path / "src.nz")
    artifacts.register(repo, item(), {"bank.nz": source})
    records = tmp_path / "records.jsonl"
    records.write_text('{"r": 1}\n')
    manifests = repo / "training/corpus-manifests"
    manifests.mkdir(parents=True)
    (manifests / "corpus-a.json").write_text(json.dumps({"id": "corpus-a", "files": [
        {"path": "records.jsonl", "sha256": artifacts.digest(records)}]}))
    run = tmp_path / "run"
    run.mkdir()
    nz(run / "system-prompts.nz", width=5)
    (run / "summary.json").write_text(json.dumps({"options": {"records": str(records), "bank": str(source),
                                                              "base": str(tmp_path / "hub/snapshots/feed01")}}))
    artifacts.register_run(run, "bank-run-v1", commit="c0ffee", repo=repo)
    entry = artifacts.entry(artifacts.load_registry(repo), "bank-run-v1")
    assert entry["init"]["parent"] == "bank-test-v1" and entry["training"]["corpora"] == ["corpus-a"]
    assert entry["backbone"]["revision"] == "feed01" and entry["qualification"]["status"] == "unqualified"


def test_a_declared_local_replica_is_used_only_when_its_bytes_match(tmp_path, monkeypatch):
    repo = repo_with_corpus(tmp_path)
    artifacts.register(repo, item(), {"bank.nz": nz(tmp_path / "src.nz")})
    replica = tmp_path / "nvme"
    replica.mkdir()
    (replica / "bank.nz").write_bytes((tmp_path / "src.nz").read_bytes())
    table = tmp_path / "replicas.json"
    table.write_text(json.dumps({"bank-test-v1": str(replica)}))
    monkeypatch.setattr(artifacts, "REPLICAS", table)
    path, sha = artifacts.resolve("bank-test-v1", repo=repo)
    assert path == replica / "bank.nz"
    (replica / "bank.nz").write_bytes(b"other bytes")  # a stale copy falls back to the registered bytes
    path, fallback = artifacts.resolve("bank-test-v1", repo=repo)
    assert path != replica / "bank.nz" and fallback == sha


def test_projected_artifact_kind_is_retired():
    registry = artifacts.empty_registry()
    with pytest.raises(artifacts.ArtifactError, match="artifact kind"):
        artifacts.validate_entry(item("projection-test-v1", kind="projection"), registry)
