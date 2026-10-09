import json
import os

import pytest
import torch
from safetensors.torch import save_file

from natlang_neuralese import artifacts


def nz(path, dialect="nd:test@1", width=8):
    header = {"exports": {"piece": {"type": "Neuralese<string>", "value": {"$neuralese": {"id": "b1"}}}},
              "blocks": {"b1": {"dialect": dialect, "dtype": "F32", "length": 3}}}
    save_file({"b1": torch.zeros(3, width)}, str(path), metadata={"natlang": json.dumps(header)})
    return path


def repo_with_corpus(tmp_path):
    (tmp_path / "training").mkdir()
    (tmp_path / "training/neuralese_corpora.json").write_text(json.dumps({"corpora": [{"id": "corpus-a"}]}))
    return tmp_path


def item(identity="bank-test-v1", **extra):
    return {"id": identity, "kind": "prompt-bank", "owner": "dgx", "dialect": "nd:test@1",
            "backbone": {"model": "test/model", "revision": "abc"}, "init": {"method": "text", "source": "x.ts"},
            "training": None, "qualification": {"status": "unqualified", "evidence": []}, **extra}


def test_register_verify_resolve_and_refuse_other_spaces(tmp_path):
    repo = repo_with_corpus(tmp_path)
    manifest = artifacts.register(repo, item(), {"bank.nz": nz(tmp_path / "src.nz")})
    assert manifest["files"][0]["nz"] == {"exports": ["piece"], "blocks": 1, "dialects": ["nd:test@1"], "widths": [8]}
    assert artifacts.verify(repo, "bank-test-v1")["id"] == "bank-test-v1"
    path, sha = artifacts.resolve("bank-test-v1", repo=repo, dialect="nd:test@1", backbone_model="test/model")
    assert path.name == "bank.nz" and len(sha) == 64
    with pytest.raises(artifacts.ArtifactError, match="dialect"):
        artifacts.resolve("bank-test-v1", repo=repo, dialect="nd:other@1")
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
    with pytest.raises(artifacts.ArtifactError, match="differ from"):
        artifacts.register(repo, item("bank-test-v2"), {"bank.nz": nz(tmp_path / "other.nz", dialect="nd:other@1")})


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


def test_bundles_declare_adapter_dialects(tmp_path):
    repo = repo_with_corpus(tmp_path)
    path = tmp_path / "bundle.nz"
    header = {"exports": {}, "blocks": {"a": {"dialect": "nd:test@1"}, "b": {"dialect": "adapter/1;base=x;kind=xs"}}}
    save_file({"a": torch.zeros(2, 8), "b": torch.zeros(2, 4)}, str(path), metadata={"natlang": json.dumps(header)})
    with pytest.raises(artifacts.ArtifactError, match="differ from"):
        artifacts.register(repo, item("bundle-test-v1", kind="bundle"), {"a.nz": path})
    manifest = artifacts.register(repo, item("bundle-test-v2", kind="bundle", extra_dialects=["adapter/1;base=x;kind=xs"]),
                                  {"a.nz": path})
    assert manifest["extra_dialects"] == ["adapter/1;base=x;kind=xs"]
    with pytest.raises(artifacts.ArtifactError, match="adapter dialects"):
        artifacts.register(repo, item("bundle-test-v3", kind="bundle", extra_dialects=["nd:other@1"]), {"a.nz": path})
