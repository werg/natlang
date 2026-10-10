"""v4 of the view/ask corpus builder: SWE-rebench tool outputs take the harness bench's repository placements
(S1's split where S1 places the repository, `split_of` otherwise), so view, S1 and the harness bench agree."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "training" / "neuralese"))

from neuralese_data import view_corpus as vc  # noqa: E402
from natlang_neuralese.harness_bench import records as hb  # noqa: E402

REPOS = ("acme/lib", "other/one", "free/repo", "mixed/up")


def _row(n: int, repo: str) -> dict:
    cwd = f"/workspace/{repo.replace('/', '__')}__1.0"
    log = "\n".join(f"tests/test_mod{k}.py::test_case_{k} PASSED [{k:3d}%]" for k in range(40)) + (
        "\n============================== 40 passed in 1.23s ==============================\n"
        "[Command finished with exit code 0]")
    return {"trajectory_id": f"t{n}", "instance_id": f"{repo.replace('/', '__')}-1", "repo": repo, "resolved": 1,
            "model_patch": "diff", "trajectory": [
                {"role": "system", "content": "You are OpenHands agent."},
                {"role": "user", "content": f"<uploaded_files>\n{cwd}\n</uploaded_files>\nConsider the following issue:\n"
                                            "<issue_description>\nTests fail\n</issue_description>\n"},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "e", "type": "function", "function": {"name": "execute_bash",
                                                                 "arguments": json.dumps({"command": "pytest -v"})}}]},
                {"role": "tool", "tool_call_id": "e", "name": "execute_bash", "content": log},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "g", "type": "function", "function": {"name": "finish", "arguments": json.dumps({"message": "ok"})}}]},
                {"role": "tool", "tool_call_id": "g", "name": "finish", "content": "done"}]}


def test_swe_tool_outputs_take_the_harness_bench_placements(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = [_row(n, REPOS[n % len(REPOS)]) for n in range(400)]
    parquet = tmp_path / "trajectories.parquet"
    pq.write_table(pa.Table.from_pylist(rows), parquet, row_group_size=50)
    own = {repo: hb.split_of(repo, 5) for repo in REPOS}
    flip = {"train": "test", "test": "train"}
    index = tmp_path / "s1"
    index.mkdir()
    groups = {"repo:acme/lib": flip[own["acme/lib"]], "repo:other/one": flip[own["other/one"]],
              "repo:mixed/up": "train", "swe-instance:mixed__up-1": "test"}
    (index / "index.json").write_text(json.dumps({"corpus_id": "s1", "records": len(groups)}))
    (index / "groups.json").write_text(json.dumps(groups))
    monkeypatch.setattr(vc, "NEBIUS", parquet)
    monkeypatch.setattr(hb, "PLACED_BY", (str(index),))
    ctx = vc.Ctx(tmp_path, {"swe_tool_outputs": 1000}, 32_000, 50, 0)
    docs = list(vc.swe_tool_outputs(ctx))
    assert docs, "fixture trajectories must yield tool-output documents"
    report = ctx.info["swe_tool_outputs_placement"]
    assert report["indexes"][0]["corpus_id"] == "s1" and report["unplaceable"] == ["mixed/up"]
    seen = set()
    for doc in docs:
        repo = next(r for r in REPOS if doc.groups[0] == f"swe-rebench-repo:{r}")
        seen.add(repo)
        rule = doc.notes["split_placement"]
        if repo in ("acme/lib", "other/one"):
            assert doc.split == flip[own[repo]] and rule == "published"
        elif repo == "free/repo":
            assert doc.split == own[repo] and rule == "split_of"
        else:  # unplaceable: keeps split_of; the cross-corpus closure drops it
            assert doc.split == own[repo] and rule == "unplaceable"
    assert seen == set(REPOS)


def test_v4_preset_keeps_v3_caps():
    src = (ROOT / "scripts" / "neuralese_view_corpus.py").read_text()
    assert '"v4": view_corpus.V3_CAPS' in src
    assert vc.CONVERTER.endswith("@4")


def test_registered_view_v4_supersedes_v3_and_is_pinned_by_the_recipe():
    entries = {e["id"]: e for e in json.loads((ROOT / "training/neuralese_corpora.json").read_text())["corpora"]}
    v4, v3 = entries["view-ask-20261010-v4"], entries["view-ask-20261010-v3"]
    assert v4["training_admission"] is False and v4["supersedes"] == "view-ask-20261010-v3"
    assert v3["admission"].startswith("superseded-by-view-ask-20261010-v4")
    held = " ".join(str(v4.get(k, "")) for k in ("admission", "training_admission_reason")).lower()
    assert "licen" not in held or "facts only" in held
    ids = [ix["registry_id"] for ix in v4["build"]["cross_corpus_indexes"]]
    assert ids == ["cross-corpus-index-s1-full-final-20261003-v1",
                   "cross-corpus-index-harness-bench-swe-rebench-openhands-pi-records-20261010-v4-v1"]
    assert all(entries[i]["kind"] == "cross-corpus-index" for i in ids) and v4["build"]["builder"].endswith("@4")
    recipe = json.loads((ROOT / "training/neuralese/recipes/raw-recurrence-v4.json").read_text())["overrides"]
    binding = recipe["input_bindings"]["view-stage-records"]
    assert binding["path"] == v4["path"] + "/view-stage/records.jsonl"
    assert binding["sha256"] in v4["build"]["view_stage_conversion"]
    manifest = json.loads((ROOT / "training/corpus-manifests/view-ask-20261010-v4.json").read_text())
    assert {f["path"]: f["sha256"] for f in manifest["files"]}["view-stage/records.jsonl"] == binding["sha256"]
    assert recipe["view_operator"]["corpus"]["id"] == v4["id"] and recipe["view_operator"]["admitted"] is False
