# Actual update resource observation

This extends the existing shared backbone observer; it creates no new training policy or model run.

1. The existing declared first/every-N observation cadence remains. Record resource brackets on the first five useful updates and at that same later cadence.
2. Read exact process I/O counters and process/guest/cgroup memory metadata immediately around the original update. Sample host memory every 200ms within the bracket. Label sampled host peaks and lifetime RSS separately.
3. On the actual optimizer CUDA device only, reset allocator peak counters and record allocated/reserved/peak bytes. Do not synchronize, allocate model tensors, or initialize CUDA for CPU updates.
4. Record whether the original update returned. Observation failure disables diagnostics and preserves the original update result/error, optimizer and RNG. No exposure credit is assigned.

All implementation is crisp outside-world measurement and exact serialization. I/O read counters include bounded observer proc metadata reads; writes are sampled before observer output. GPU allocator peaks are distinct from driver-wide GPU usage.
