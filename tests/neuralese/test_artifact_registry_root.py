"""A copy of natlang_neuralese outside the checkout (a recipe lineage's frozen runtime, a code snapshot) still resolves
the registered artifacts: the registry is read from the machine's repo root (common/paths.py), not from the copy's
location (run-v9's handoff to a snapshot failed with 'unknown artifact' on the preserve teacher, 2026-10-10)."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[2] / 'training' / 'neuralese' / 'natlang_neuralese'


def test_a_package_copy_reads_the_repo_root_registry(tmp_path):
    frozen = tmp_path / 'run' / 'runtime' / 'natlang_neuralese'
    shutil.copytree(PACKAGE, frozen, ignore=shutil.ignore_patterns('__pycache__'))
    repo = tmp_path / 'repo'
    (repo / 'training').mkdir(parents=True)
    (repo / 'training' / 'neuralese_artifacts.json').write_text(json.dumps(
        {'schema': 'natlang.neuralese-artifacts/1', 'artifacts': [{'id': 'probe-teacher-1'}]}))
    script = ('import natlang_neuralese.artifacts as a, json; '
              'print(json.dumps([a.__file__, str(a.REPO), [x["id"] for x in a.load_registry()["artifacts"]]]))')
    env = {k: v for k, v in os.environ.items() if k != 'PYTHONPATH'}
    env.update(PYTHONPATH=str(frozen.parent), NATLANG_REPO=str(repo))
    out = subprocess.run([sys.executable, '-c', script], cwd=tmp_path, env=env, capture_output=True, text=True,
                         check=True).stdout
    module, root, ids = json.loads(out)
    assert module.startswith(str(frozen)) and root == str(repo) and ids == ['probe-teacher-1']


def test_the_default_repo_root_holds_the_registry_the_recipes_name():
    from natlang_neuralese import artifacts
    from natlang_neuralese.common.paths import root
    assert artifacts.REPO == root('repo')
    if (root('repo') / artifacts.REGISTRY).exists():
        ids = {item['id'] for item in artifacts.load_registry()['artifacts']}
        assert 'mellum21-teacher-v3-top64-20261010' in ids
