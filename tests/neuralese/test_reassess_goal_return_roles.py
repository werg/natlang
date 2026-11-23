"""Run the native-evidence reassessor's Node regressions in the normal suite."""
from pathlib import Path
import subprocess


def test_goal_return_roles_native_evidence():
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ['node', '--test', str(root / 'scripts/neuralese_data/reassess_goal_return_roles.test.mjs')],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
