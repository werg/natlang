"""The env-gated update profiler (common/profiling.py): a no-op when unset; on CUDA it attributes forward, checkpoint
recompute, backward (by the forward scope of each gradient) and the MuonSR Newton-Schulz step."""

import pytest

from natlang_neuralese.common import profiling


def test_scopes_are_shared_noops_when_no_profile_records(tmp_path):
    assert not profiling.ENABLED
    assert profiling.scope('norm') is profiling.scope('moe') is profiling.region('update/forward')
    assert profiling.backward_phase() is profiling.scope('x')
    profiler = profiling.UpdateProfiler(tmp_path, value='')
    profiler.begin(1)
    profiler.maybe_finish(1)
    profiler.close(1)
    assert not profiler.active and not list(tmp_path.iterdir())
    assert profiling.parse_range('5:8') == (5, 8) and profiling.parse_range('3') == (3, 3)
    assert all(profiling.parse_range(v) is None for v in ('', None, '0', '0:0', 'off', 'none'))  # launchers' "no profile"
    with pytest.raises(ValueError):
        profiling.parse_range('8:5')
