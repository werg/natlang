import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_luna_skill_overlay.py"
SPEC = importlib.util.spec_from_file_location("build_luna_skill_overlay", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _source(tmp_path, existing_skill):
    skill_path = "module/skills/judge-against-criteria/SKILL.md"
    row = {
        "id": "program:one",
        "split": "train",
        "source_groups": ["group:one"],
        "source_revisions": ["source:v1"],
        "curriculum": {"reference": {"root": [[None, {"code": "export const main = 1;"}]]}},
        "semantics": {"files": {"module.nl": "export const main = 1;", **(
            {skill_path: existing_skill} if existing_skill is not None else {})},
            "expected": {"result": 1}, "folder_files": {"input.json": "{}"}},
    }
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps(row, ensure_ascii=False) + "\n")
    skill = tmp_path / "new-skill.md"
    skill.write_text("New generic skill instructions.\n")
    return source, skill, row, skill_path


def test_replaces_only_hash_bound_existing_skill(tmp_path):
    source, skill, original, skill_path = _source(tmp_path, "Old skill.\n")
    output, receipt = tmp_path / "overlay.jsonl", tmp_path / "receipt.json"
    old_sha = MODULE.sha(b"Old skill.\n")
    result = MODULE.build(source, [{"label": "case", "index": 0, "seed": 7}], skill,
                          output, receipt, replace_existing_skill_sha256=old_sha)
    modified = json.loads(output.read_text())
    assert modified["semantics"]["files"][skill_path] == "New generic skill instructions.\n"
    modified["semantics"]["files"][skill_path] = "Old skill.\n"
    assert modified == original
    assert result["rows"][0]["skill_change"] == {"kind": "replace-existing-skill", "prior_sha256": old_sha}


def test_existing_skill_requires_exact_replacement_hash(tmp_path):
    source, skill, _, _ = _source(tmp_path, "Old skill.\n")
    try:
        MODULE.build(source, [{"label": "case", "index": 0, "seed": 7}], skill,
                     tmp_path / "overlay.jsonl", tmp_path / "receipt.json",
                     replace_existing_skill_sha256="0" * 64)
    except ValueError as exc:
        assert "existing skill hash mismatch" in str(exc)
    else:
        raise AssertionError("a different existing skill body must not be replaced")
