# CPU packaging and preparation headroom

Policy: keep useful training running while bounded optional CPU work uses genuinely available memory. Reclaimable clean active file cache is not anonymous learner memory. Retain existing available-memory floors and anonymous scratch limits; never count shared-memory state, dirty pages or writeback as freely reclaimable.

1. Read host MemAvailable and the actual cgroup-v2 limit/current/stat files.
2. Count clean disk-backed file cache conservatively: the smaller of active_file plus inactive_file and file minus shmem minus dirty minus writeback, clamped at zero.
3. Add this reclaimable cache to limit minus current, clamped at zero, and cap the result by host availability. An unlimited cgroup uses host availability.
4. Existing source-host optional preparers and actor-view followers use this single crisp arithmetic helper. Their policy floors, source ownership, resource sampling and anonymous caps remain unchanged. Missing/malformed accounting fails visibly instead of silently granting memory.

This is CPU host-memory headroom, not CUDA memory capacity or a guarantee that reclaim has zero performance cost. It does not change learner math or alter the DGX unified-memory ledger.
