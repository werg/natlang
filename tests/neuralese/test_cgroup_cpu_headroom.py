import pytest
from scripts.coreweave.cgroup_memory import cpu_headroom_bytes


def stats(**changes):
    value=dict(active_file=77,inactive_file=2,file=80,shmem=1,file_dirty=0,file_writeback=0)
    value.update(changes)
    return value


def test_clean_active_cache_is_usable_for_cpu_work():
    assert cpu_headroom_bytes(90,96,90,stats()) == 85


@pytest.mark.parametrize('changes,expected',[
    (dict(file_dirty=70),15),
    (dict(shmem=78),8),
    (dict(file_dirty=40,file_writeback=30),15),
    (dict(file_dirty=100),6),
    (dict(active_file=0,inactive_file=2),8),
])
def test_dirty_shared_or_unaccounted_cache_is_not_free(changes,expected):
    assert cpu_headroom_bytes(90,96,90,stats(**changes)) == expected


def test_host_and_limit_bounds_and_unlimited_group():
    assert cpu_headroom_bytes(3,96,90,stats()) == 3
    assert cpu_headroom_bytes(90,96,200,stats()) == 0
    assert cpu_headroom_bytes(90,'max',0,{}) == 90


def test_missing_accounting_is_not_a_silent_grant():
    with pytest.raises(KeyError):cpu_headroom_bytes(90,96,90,{})
