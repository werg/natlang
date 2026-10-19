# Engine global memory reserve

Policy: bounded external CPU/inference work yields before either its cgroup or the shared guest loses the declared free-memory reserve. Training is not stopped by this mechanism.

1. Crisp sampler reads exact kernel cgroup counters and guest MemAvailable.
2. Existing guard compares the minimum available bound against its explicit reserve and observes OOM events. This reserve is free headroom, not a worker memory claim.
3. Existing session custody drains only the guard-owned worker and descendants, preserving durable reasons and completed outputs.

The kernel accounting and process identity operations require crisp outside-world plumbing. No model-facing policy or text changes.
