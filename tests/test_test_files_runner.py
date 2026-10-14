"""A failed startup must retain its reason alongside the test name."""
from pathlib import Path
import shutil
import subprocess


def test_first_named_failure_keeps_the_hook_diagnostic(tmp_path):
    root = tmp_path / 'ts-host'
    (root / 'scripts').mkdir(parents=True)
    (root / 'test').mkdir()
    source = Path(__file__).resolve().parents[1] / 'ts-host/scripts/test-files.mjs'
    shutil.copy(source, root / 'scripts/test-files.mjs')
    (root / 'test/startup.test.mjs').write_text(
        "import { before, test } from 'node:test';\n"
        "before(() => { throw new Error('fixture startup failed: CUDA allocation'); });\n"
        "test('named learning assertion', () => {});\n")
    result = subprocess.run(['node', str(root / 'scripts/test-files.mjs')],
                            text=True, capture_output=True)
    assert result.returncode == 1
    assert 'test/startup.test.mjs' in result.stdout
    assert 'named learning assertion' in result.stdout
    assert 'fixture startup failed: CUDA allocation' in result.stdout
