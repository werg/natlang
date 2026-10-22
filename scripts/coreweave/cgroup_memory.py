"""Conservative cgroup-v2 CPU headroom including clean active file cache."""


def cpu_headroom_bytes(host_available, limit, current, stats):
    """Estimate CPU allocation headroom without counting tmpfs or dirty cache."""
    if limit == 'max':
        return max(0, int(host_available))
    lru = max(0, int(stats['inactive_file'])) + max(0, int(stats['active_file']))
    clean_file = max(0, int(stats['file']) - int(stats['shmem'])
                     - int(stats['file_dirty']) - int(stats['file_writeback']))
    reclaimable = min(lru, clean_file)
    return max(0, min(int(host_available), int(limit) - int(current) + reclaimable))


def available_cpu_memory_bytes():
    """Read actual CPU availability; malformed accounting fails to the caller."""
    from pathlib import Path
    host = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                    if line.startswith('MemAvailable:'))) * 1024
    cg = Path('/sys/fs/cgroup')
    limit = (cg / 'memory.max').read_text().strip()
    if limit == 'max':
        return max(0, host)
    stats = dict(line.split() for line in (cg / 'memory.stat').read_text().splitlines())
    return cpu_headroom_bytes(host, limit, (cg / 'memory.current').read_text().strip(), stats)
