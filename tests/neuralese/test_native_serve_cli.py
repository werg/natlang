"""The public server entrypoint accepts complete native checkpoints only."""

import pytest
import os
import subprocess
import sys
from pathlib import Path

from natlang_neuralese.serve.__main__ import main


@pytest.mark.parametrize("retired", [
    "--heads", "--lora", "--direct-artifact", "--direct-base-snapshot",
    "--recurrence-checkpoint", "--runtime-qualification", "--cutoff", "--max-block",
])
def test_public_server_rejects_retired_projected_inputs(retired):
    with pytest.raises(SystemExit) as error:
        main([retired, "old-checkpoint"])
    assert error.value.code == 2


def test_public_server_requires_native_checkpoint():
    with pytest.raises(SystemExit) as error:
        main([])
    assert error.value.code == 2


def test_native_http_import_does_not_load_projected_gradient_or_writer_modules():
    package_root = Path(__file__).resolve().parents[2] / "training" / "neuralese"
    code = (
        "import sys; from natlang_neuralese.serve.http import make_handler; "
        "from natlang_neuralese.serve import native_learning, native_recorded_objective; "
        "retired = {'natlang_neuralese.serve.grad', 'natlang_neuralese.write', "
        "'natlang_neuralese.model.heads', 'natlang_neuralese.model.neuralese_projection'}; "
        "loaded = sorted(retired & sys.modules.keys()); "
        "assert not loaded, f'projected implementation imported by native HTTP surface: {loaded}'"
    )
    env = dict(os.environ, PYTHONPATH=str(package_root))
    subprocess.run([sys.executable, "-c", code], env=env, check=True)


def test_native_server_clis_expose_prefill_chunk_binding():
    package_root = Path(__file__).resolve().parents[2] / "training" / "neuralese"
    env = dict(os.environ, PYTHONPATH=str(package_root))
    checkpoint_server = Path(__file__).resolve().parents[2] / "scripts" / "neuralese_data" / "serve_native_checkpoint.py"
    direct_server = subprocess.run([sys.executable, str(checkpoint_server), "--help"], env=env,
        capture_output=True, text=True, check=True)
    direct_help = " ".join(direct_server.stdout.split())
    assert "--prefill-tokens" in direct_help
    assert "2048 on GPUs <=12 GiB" in direct_help
    with pytest.raises(SystemExit) as error:
        main(["--help"])
    assert error.value.code == 0
