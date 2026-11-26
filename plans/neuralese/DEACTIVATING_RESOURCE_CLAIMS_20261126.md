# Retain resource accounting while a unit stops

Inventory before editing (base 6ef3bfba0442305ca7b42b893cb1b8811d556752):

- `scripts/memory_ledger.py:418-434` measures host/cgroup and declared/discovered container memory; `538-549` measures discrete CUDA. Both excluded systemd `deactivating` despite a live cgroup.
- `measure_claims` (561), shared usage cache (579-628), `live_claims` (631-655), host headroom (688), and CUDA headroom (339) consume those measurements. `live_claims` released the claim on a missing measurement after startup grace. Status, admission and guard share this projection.
- `head_blocked` (791), `may_backfill` (756), backfill allowance (742), and preemption planning (879) use retained measured resources and outstanding reservations. Losing a stopping claim can incorrectly admit fresh backfill while a head waits for that resource lifetime to finish.
- Relaunch eligibility (837), `unit_exit` (1150), guard pending shutdown (1747) already distinguish stopping from finished. Keep their completion/cancellation semantics. Launch-failure claim removals (1140,1731) are separate failed-start paths, not this state filter.
- All local, DGX and Pop guards and `run`/status clients use this same script. Container and CUDA measurements use the same state predicate; no engine-specific model/runtime code implements ledger state filtering.

Default: `resource_bearing_state` includes active, activating, reloading and deactivating. Diagnostic opt-out `NATLANG_LEDGER_COUNT_DEACTIVATING=0` restores the former measurement rule, without changing owner cancellation or restart eligibility. Finished services release normally. No additional memory, reduced reserve, claim changes or forced process stops.

Controls exercise all four live states, inactive release, head/backfill scheduling while shutdown retains memory, CUDA ownership and opt-out. Existing fixed claims, cancellation, guard and continuation tests remain unchanged.

Historical limitation: Source57 events show claim release during pending shutdown and fresh backfills before/soon after kill. Exact historical ActiveState/cgroup statistics were not captured. Kill events sample MemAvailable immediately after signaling, not after resource join. This fixes a demonstrated lifetime defect; it does not attribute the entire observed reclaim gap. Shared/cache/kernel reclaim and outside-ledger fluctuations remain unmeasured for those instants.
