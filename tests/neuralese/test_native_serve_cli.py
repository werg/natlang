"""The public server entrypoint accepts complete native checkpoints only."""

import pytest

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
