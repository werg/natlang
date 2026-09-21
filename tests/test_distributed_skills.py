"""Relocatable authoring resources and executable examples shipped to agents."""
from pathlib import Path
import re

import pytest
import yaml

from natlang.install_skills import NAMES, bundled_skills, install
from natlang.host import load
from natlang.runtime import Runtime
from natlang.values import dump


def test_installation_is_complete_relocatable_and_preserves_existing_skills(tmp_path):
    installed = install(tmp_path / "agent-skills")
    for folder in installed:
        source = bundled_skills() / folder.name
        for path in source.rglob("*"):
            if path.is_file():
                assert (folder / path.relative_to(source)).read_bytes() == path.read_bytes()
        for document in folder.rglob("*.md"):
            for link in re.findall(r"\]\(([^)]+)\)", document.read_text()):
                if "://" not in link and not link.startswith("#"):
                    assert (document.parent / link.split("#")[0]).exists(), (document, link)
        metadata = yaml.safe_load((folder / "SKILL.md").read_text().split("---", 2)[1])
        assert metadata["name"] == folder.name
    marker = installed[0] / "SKILL.md"
    marker.write_text("User maintained content")
    with pytest.raises(FileExistsError):
        install(tmp_path / "agent-skills")
    assert marker.read_text() == "User maintained content"
    with pytest.raises(ValueError):
        install(tmp_path / "elsewhere", ["../escape"])
    assert not (tmp_path / "elsewhere").exists()


@pytest.mark.parametrize("observations", [[], ["one", "two", "three"]])
def test_relocated_authoring_example_runs_real_helpers_with_scripted_semantic_leaf(tmp_path, observations):
    folder = install(tmp_path, [NAMES[0]])[0]
    root = load(folder / "assets/review/review.nl", {"observations": observations, "criterion": "Fixture criterion"})

    class Fixture:
        def __init__(self, lam):
            self.lam = lam

        def run(self, session):
            if self.lam.fn_name == "assess":
                verdict = {"one": "supported", "two": "contradicted", "three": "uncertain"}[self.lam.in_["observation"]]
                result = session.apply("write", {"path": "return", "type": "Assessment",
                    "value": {"verdict": verdict, "reason": "Scripted wiring fixture"}})
                assert result.kind == "ok", result.text
            else:
                for args in [
                    {"function": "assess", "to": "let/assessments", "over": "args/observations",
                     "inputs": {"criterion": "args/criterion"}},
                    {"function": "summarize", "to": "return", "inputs": {"assessments": "let/assessments"}},
                ]:
                    result = session.apply("call", args)
                    assert result.kind == "done", result.text
            assert session.finish()

    outcome, value = Runtime(lambda lam: Fixture(lam)).run_root(root)
    assert outcome.kind == "done"
    report = dump(value)
    assert len(report["assessments"]) == len(observations)
    assert [report[k] for k in ("supported", "contradicted", "uncertain")] == ([1, 1, 1] if observations else [0, 0, 0])
