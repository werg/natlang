# Publish completed quality feedback during the next transfer

Policy: expose each durable completed evaluation promptly, even while the
supervisor waits for the next separately admitted checkpoint transfer. Preserve
its exact historical/current-weight flags and checkpoint identity. A transfer's
failure or success is a separate fact and cannot confer evaluation credit.

The existing natlang supervision stages remain unchanged: evaluate, persist
the result, choose the next available checkpoint, transfer and dispatch. The
crisp supervision loop must mirror the already-persisted result while it waits
for its worker. This is exact durability/plumbing, not a new model-facing task.

Observed trigger: Mellum step20,515's result was durable at11:22 UTC, while the
visible status still showed20,513 because the controller was waiting for a
subsequent checkpoint download. An administrative atomic copy exposed the
existing result without touching the active download or evaluator.

Do not replace the canonical service source while an admitted worker still
binds that source file. Prepare and validate the change separately, then adopt
it at a verified worker boundary. Tests must show an already-completed result
appearing before the next transfer finishes, with historical flags unchanged,
and preserve the worker's original result/custody/failure checks.
