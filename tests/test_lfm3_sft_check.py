"""A failed trainer must never become a successful distributed seed."""
import json
import os
from pathlib import Path
import subprocess

import pytest


@pytest.mark.parametrize('trainer_exit,checkpoint,expected', [(23, False, 23), (0, False, 1), (0, True, 0)])
def test_seed_requires_success_and_real_checkpoint(tmp_path, trainer_exit, checkpoint, expected):
    root = Path(__file__).resolve().parents[1]
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    def executable(name, text):
        p = bin_dir / name
        p.write_text('#!/bin/bash\n' + text)
        p.chmod(0o755)
    executable('nvidia-smi', 'echo "fixture,0,0"\n')
    executable('sleep', '/usr/bin/sleep .05\n')
    # Only the HF base download is replaced. Argument/progress JSON uses real Python.
    executable('python3', 'if [ "$#" = 1 ] && [ "$1" = - ]; then cat >/dev/null; echo /fixture/base; else exec /usr/bin/python3 "$@"; fi\n')
    write = ''
    if checkpoint:
        write = """mkdir -p "$TEST_OUT/train"; echo state > "$TEST_OUT/train/checkpoint.pt"; echo '{"checkpoint_step":1,"global_step":1}' > "$TEST_OUT/train/progress.json"
"""
    executable('python', write + f'exit {trainer_exit}\n')
    args = tmp_path / 'args.json'
    args.write_text(json.dumps(['--base', '@@BASE@@']))
    out = tmp_path / 'out'
    env = dict(os.environ, PATH=str(bin_dir) + ':' + os.environ['PATH'], CHECK_PHASE_B='0', TEST_OUT=str(out))
    result = subprocess.run(['bash', str(root / 'scripts/coreweave/lfm3_sft_check.sh'), str(out), str(args)],
                            cwd=root, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == expected, (result.stdout, result.stderr)
    phases = (out / 'phases.log').read_text()
    assert f'phase A exit {trainer_exit}' in phases
    assert ('phase B skipped' in phases) == (expected == 0)
